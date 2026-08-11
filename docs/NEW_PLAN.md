# 잔여 실험 지시문 (guard-lab)

발표(D-3)까지 수행할 추가 실험 5건. 우선순위 순으로 배치되어 있으며, EXP-1·2가 필수(지시문 커버리지), EXP-3·4가 발표 방어용, EXP-5는 선택이다. 뒤 실험이 잘려도 앞 실험만으로 발표가 성립하도록 독립적으로 설계했다.

## 공통 규칙

- 기존 코드베이스(`models.py`, `eval.py`, `app.py`)를 재사용한다. 새 추상화 계층을 만들지 말 것. 실험별 스크립트는 `exp/` 아래 단일 파일로 작성한다.
- 모든 결과는 `results/final/` 아래 CSV로 저장하고, 각 CSV 첫 행 주석 또는 동명 `.meta.json`에 실행 조건(모델 id, dtype, 샘플 수, 날짜, git commit hash)을 기록한다.
- 추론 조건 고정: bf16, greedy(`do_sample=False`), `max_new_tokens=64`, `tokenizer.padding_side="left"`.
- 기존 예측 CSV(`results/*.csv`)가 있는 실험은 **재실행하지 말고 재활용**한다. GPU 재실행은 EXP-1과 EXP-2의 LG3 응답축뿐이다.
- 대상 모델 3종 고정: `ToxicityPrompts/PolyGuard-Qwen-Smol`, `meta-llama/Llama-Guard-3-1B`, `SamsungSDS-Research/SGuard-ContentFilter-2B-v1`.
- 판단이 필요한 지점(데이터셋 필드명 불일치, 파싱 포맷 불명 등)은 임의로 추정해서 진행하지 말고 raw 출력을 print한 뒤 멈추고 보고할 것.

---

## EXP-1. Latency 벤치마크

**목적**: 3모델의 추론 지연시간 비교. "판단 차이와 latency 비교"라는 원 지시문의 미충족 항목을 메꾼다.

**설계**
- 입력: ko probe의 base 30문장 + PGPrompts en 20문장 = 50건 고정 세트. 세트 구성을 `exp/latency_inputs.json`으로 저장해 재현 가능하게 한다.
- 측정 대상은 **모델 forward + generate 시간만**이다. FastAPI/HTTP 경유 측정 금지. 토크나이즈 시간은 별도 컬럼으로 분리 기록.
- 절차: 모델당 warmup 5회(측정 제외) → 50건을 배치 1로 순차 실행 → 건별 wall time 기록. `torch.cuda.synchronize()`를 측정 전후에 반드시 호출한다(비동기 커널 때문에 이거 없으면 수치가 무의미하다).
- 3모델을 한 프로세스에서 순차 측정하되, 모델 전환 시 이전 모델을 `del` + `torch.cuda.empty_cache()` 후 로드한다.

**산출물**
- `results/final/latency_raw.csv`: `model, input_id, input_len_tokens, output_len_tokens, tokenize_ms, generate_ms`
- `results/final/latency_summary.csv`: 모델별 median / p95 / mean±std

**verify**
- [ ] 모델별 50행 × 3 = 150행, 결측 없음
- [ ] median과 p95가 함께 보고되고, warmup 제외가 코드상 확인됨
- [ ] 출력 토큰 수가 모델별로 크게 다르면(예: PolyGuard 4-field vs LG3 한 줄) summary에 `ms_per_output_token` 컬럼을 추가해 정규화 수치도 병기

---

## EXP-2. Response Harmfulness 재현

**목적**: 지금까지 prompt harmfulness 축만 평가했다. PolyGuard 지시문에 명시된 response harmfulness 축을 추가한다.

**설계**
- 데이터: PGPrompts en/ko 기존 평가 서브셋 그대로. 정답 필드는 response harm 라벨을 사용한다(정확한 필드명은 데이터셋에서 확인 후 `.meta.json`에 기록).
- **PolyGuard**: 재실행 불필요. 기존 예측 CSV에 4-field raw 출력이 저장되어 있다면 파서에 response harmfulness 필드 추출을 추가해 재채점만 한다. raw 출력이 저장 안 돼 있으면 이때만 재실행.
- **LG3-1B**: 재실행 필요. chat template의 conversation에 user + assistant turn을 모두 넣는다. 기존에 잡았던 버그와 동일하게 content는 반드시 `[{"type":"text","text":...}]` 형식으로 감쌀 것. Agent 모드(응답 평가) 프롬프트가 별도 포맷인지 모델 카드에서 확인 후 진행.
- **SGuard**: ContentFilter가 (prompt, response) 쌍 입력을 지원하는지 모델 카드 샘플 코드로 확인. 미지원이면 이 실험에서 제외하고 그 사실만 기록(무리하게 우회하지 말 것).
- response가 빈 문자열인 샘플은 평가에서 제외하고 제외 건수를 기록한다.

**산출물**
- `results/final/response_harm_{model}_{lang}.csv`: `id, text, response, pred, label, raw_output`
- 채점 요약: prompt축 기존 결과와 나란히 놓은 F1 표 (en/ko × prompt/response)

**verify**
- [ ] 파싱 실패율 <5%, 실패 건은 raw_output과 함께 별도 파일로 격리
- [ ] F1이 unsafe=positive 기준으로 계산됨 (기존 채점 코드와 동일 함수 사용)
- [ ] LG3 재실행 전에 스모크 1건으로 raw 출력을 확인하고 파서를 맞춘 뒤 전량 실행

---

## EXP-3. Confidence Calibration

**목적**: demo가 출력하는 confidence가 실제로 믿을 만한 숫자인지 검증. reliability curve + ECE.

**설계**
- **GPU 재실행 없음.** Phase 7에서 구현한 logprob confidence 추출을 전체 평가 세트(PGPrompts en/ko 전량 + ko probe 150행)의 기존 예측에 적용한다. 기존 예측 시 logprob을 저장 안 했다면, 이때만 `output_scores=True`로 재실행하되 이 비용을 먼저 보고하고 진행 여부를 확인받을 것.
- confidence 정의를 모델별로 명시한다: safe/unsafe 결정 토큰 위치에서 두 토큰 logit만 뽑아 2-way softmax한 max 확률. SGuard는 자체 confidence를 제공하므로 그걸 사용하고, 정의가 다름을 기록.
- 계산: 예측 confidence를 10개 bin(0.5~1.0 균등)으로 나눠 bin별 (mean confidence, accuracy) → reliability diagram. ECE는 bin별 |conf−acc|의 샘플 가중 평균.
- bin에 샘플 30개 미만이면 해당 bin을 회색 처리하고 각주로 표기(적은 표본으로 과해석 금지).

**산출물**
- `results/final/calibration_{model}.csv`: `id, pred, label, confidence`
- `results/final/calibration_summary.csv`: 모델별 ECE
- `figures/reliability_{model}.png`: matplotlib, 대각선 기준선 포함, 3모델 동일 축 범위

**verify**
- [ ] ECE 수치와 diagram의 bin 값이 상호 일치 (같은 CSV에서 생성)
- [ ] confidence 분포 히스토그램을 diagram 하단에 병기 (쏠림 여부 확인용)
- [ ] SGuard의 confidence 정의가 타 모델과 다름이 figure 캡션에 명시됨

---

## EXP-4. McNemar 유의성 검정

**목적**: 주요 비교쌍의 성능 차이에 p-value를 부착. 발표 질문 "그 차이가 유의한가?"에 대한 사전 답변.

**설계**
- **GPU 재실행 없음.** 기존 예측 CSV만 사용.
- 비교쌍 (동일 샘플 위에서 paired):
  1. PGPrompts en: PolyGuard vs LG3
  2. PGPrompts ko: PolyGuard vs LG3
  3. ko probe (전체 150행): 3모델 pairwise 3쌍
- 각 쌍에 대해 2×2 분할표(둘 다 정답 / A만 정답 / B만 정답 / 둘 다 오답) 작성 → `statsmodels.stats.contingency_tables.mcnemar` (exact=True if 불일치 셀 합 <25, else chi-square with correction).
- ko probe pairwise 3쌍에는 Bonferroni 보정(α=0.05/3)을 적용하고 보정 전후 p를 모두 보고.
- 함께 Cohen's κ도 계산해 모델 간 판단 일치도를 기록 (t2i-lab에서 쓰던 구현 재사용 가능하면 재사용).

**산출물**
- `results/final/mcnemar.csv`: `pair, dataset, n, b, c, p_raw, p_adjusted, kappa`

**verify**
- [ ] 각 쌍의 분할표 4셀 합계 = 해당 데이터셋 샘플 수
- [ ] exact/chi-square 선택 기준이 코드에 조건문으로 구현됨
- [ ] p-value와 함께 방향(누가 우세한지: b vs c)이 표에 표기됨

---

## EXP-5. (선택) 영어 변형 프로브

**목적**: ko probe에서 발견한 변형 취약성이 한국어 특유인지, 언어 무관인지 분리. 시간이 남을 때만 수행하고, 착수 전 보고할 것.

**설계**
- ko probe의 base 30문장을 영어로 번역해 `data/en_probe_base.csv` 생성. 번역은 의미 보존을 사람이 검수해야 하므로 **자동 번역 후 검수 대기 상태로 멈추고 보고**한다. 라벨은 base에서 그대로 상속.
- 변형 5종을 영어에 맞게 대응 적용: 원문 / leet-speak 치환(a→4, e→3 등, ko의 우회 표기 대응) / 글자 간 공백·기호 삽입 / 한국어 직역투 영어(ko의 번역투 대응) / code-switching(영한 혼용, ko probe와 역방향).
- 변형 생성 규칙을 `exp/en_variants.py`에 함수로 작성하고, 생성된 150행을 사람이 훑을 수 있게 CSV로 뽑는다.
- 평가·채점·flip rate 계산은 ko probe 파이프라인을 그대로 재사용한다. 새로 짜지 말 것.

**산출물**
- `data/en_probe.csv` (150행, ko probe와 동일 스키마)
- `results/final/flip_rate_en.csv` + ko와 나란히 비교한 variant × model × lang 표

**verify**
- [ ] ko/en probe가 base_id로 1:1 대응되어 언어 간 직접 비교 가능
- [ ] flip rate 계산 함수가 ko와 동일 코드 경로임 (diff 없음)

---

## 실행 순서와 동결

```
Day 1: EXP-1 → EXP-2        (지시문 커버리지 완성)
Day 2: EXP-3 → EXP-4 → (EXP-5)  (발표 방어 실험)
Day 3 오전: results/final/ 동결. 이후 어떤 수치도 재생성 금지.
```

동결 시점에 `results/final/` 전체를 커밋하고 태그(`presentation-freeze`)를 남긴다. 이후 발견되는 문제는 수치 수정이 아니라 발표 자료의 각주로 처리한다.
