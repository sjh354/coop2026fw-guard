# guard-lab

Safety guard 모델(Llama Guard, PolyGuard) 재현 + 서빙 실습.
자세한 실행 계획은 [docs/PLAN.md](docs/PLAN.md) 참고 — Phase별 세부 스텝과 verify 조건이
필요할 때 그쪽을 열어볼 것. t2i 이미지 생성 작업(`../t2i-lab/`)은 이 트랙이 끝날 때까지 보류.

## 구조

```
guard-lab/
├── CLAUDE.md
├── README.md
├── models.py      # 모델 로드 + 프롬프트 빌드 + 출력 파싱
├── eval.py        # PGPrompts 서브셋 평가 → CSV
├── app.py         # FastAPI 서빙
├── docs/
│   ├── PLAN.md      # Phase 0~7 전체 계획
│   ├── NEW_PLAN.md  # 잔여 실험 EXP-1~5 지시문
│   ├── handoff_multilingual.md  # 7개 언어 확장 데이터셋 추출 지시서 + 실행 결과
│   └── results_summary.html  # 발표용 EXP-1~5 결과 요약 (단일 HTML, 브라우저로 열기)
├── exp/           # EXP-1~5 실험별 단일 스크립트 + extract_multilingual_base.py
├── data/          # ko_probe / en_probe / multilingual_base.csv (base + variant 문장)
│   └── multilingual_base.csv: Track A(PG 홈그라운드, es/hi/th/ar/ru, 순위 금지) +
│     Track B(OOD 공정 비교, 7개 언어, 순위 메인) — `track` 컬럼으로 구분, 상세는
│     docs/handoff_multilingual.md
├── results/       # {model}_{lang}.csv 평가 결과 (+ _conf.csv: confidence 포함 버전)
├── results/final/ # EXP-1~5 산출물 CSV + .meta.json
└── figures/       # EXP-3 reliability diagram 등 plot 산출물
```

로컬에서 다국어 데이터 추출만 할 때는 GPU가 필요 없어 `guard` conda env 대신 `.venv`
(homebrew python 3.14 + pandas/datasets/huggingface_hub)를 씀 — `python3 -m venv .venv &&
.venv/bin/pip install pandas datasets pyarrow huggingface_hub`. `guard` env(3.11)와는 버전이
다르니 모델 추론이 필요한 작업(eval.py/app.py)은 반드시 서버의 `guard` env를 쓴다.

## 진행 상태

- [x] Phase 0 — conda env(`guard`) 세팅
- [x] Phase 1 — Llama Guard / PolyGuard / SGuard-v1 논문 리딩, 세 taxonomy 정렬
      (`docs/taxonomy_map.md`) — S1-S14가 Llama Guard 원문(O1-O6)이 아니라 LG3부터 MLCommons
      정렬로 재정의된 것이었다는 점 정정. S11/S13/S14는 SGuard-v1에 대응 카테고리 없음 확인
- [x] Phase 2 — 스모크 테스트 (raw 출력 확보) — LG3-1B chat template 버그(문자열 content 시
      전부 "safe") 발견/수정, S14 미지원 확인
- [x] Phase 3 — PGPrompts 재현 평가 en/ko 300샘플 (PolyGuard en 0.9711 / ko 0.9424,
      LG3-1B en 0.8993 / ko 0.6862, parse 실패율 0%) — 파이프라인 자체는 닫혔고 표본 확대·
      혼동행렬 분해가 남음, 아래 Phase 5 참고
- [x] Phase 4 — FastAPI 서빙 (`/moderate`, 동시 5개 요청/OOM 없음 확인)
- [x] Phase 5 — 한국어 변형 스위트 구축 (원문/우회표기/띄어쓰기/번역투/code-switching, flip rate) —
      `data/ko_probe.csv` 150행(base 30×5) 완성, 3모델(PolyGuard/LG3-1B/SGuard-v1) 예측 완료
      (parse 실패 전부 0%). SGuard-v1은 `models.py`에 통합(logit 기반 5카테고리 파서).

      | variant_type | PolyGuard | LG3-1B | SGuard-v1 |
      |---|---|---|---|
      | 우회표기 | 23.3% | 6.7% | 20.0% |
      | 띄어쓰기 | 10.0% | 6.7% | 3.3% |
      | 번역투 | 26.7% | 10.0% | 23.3% |
      | code-switching | 13.3% | 10.0% | 16.7% |

      세 모델 다 번역투에서 가장 흔들리고, LG3-1B가 전 variant에서 가장 안정적
- [x] Phase 6 — PGPrompts 전량(1699) 재실행 + 실패 사례 분석 (`docs/failure_cases.md`) — 300샘플
      대비 F1 5~15%p 하락 확인(표본 편향), **LG3 ko F1 저하는 precision(0.7680)이 아니라
      recall(0.4523) 문제**로 확정(진짜 다국어 갭, template 버그 잔존 아님), adversarial 슬라이스
      낙폭은 LG3 ko가 최대(recall -0.194p), 세 모델 공통으로 허구/공인 인물+개인정보 키워드
      조합과 반문형 혐오 표현에 취약, 오탐/미탐 18건 원인 추정 정리
- [x] Phase 7 — Demo 스키마 구현 (`risk → category → confidence → reason`, `models.py`의
      `moderate()` 함수 + `/moderate` 응답 확장). `app.py`에 SGuard 추가로 3모델 동시 서빙
      (VRAM 8.9GB/24GB), 동시 요청 5개 OOM 없음 확인. confidence는 결정 토큰 2개 후보 softmax
      (LG3/PolyGuard) 또는 SGuard 네이티브 5카테고리 확률. reason은 카테고리→한국어 템플릿
      매핑이며 응답에 `reason_source: "category_template"`로 명시(모델 실제 근거 아님)

Phase 5~7은 정량 재현이 아니라 정성 분석 축이다. 세 번째 비교 모델로 SGuard-v1(한국어 특화,
confidence 네이티브 제공) 추가 완료 — 상세 계획은 `docs/PLAN.md` Phase 5~7, 실패 사례는
`docs/failure_cases.md` 참고.

### 잔여 실험 (`docs/NEW_PLAN.md`)

발표 전 추가 실험 5건(EXP-1~5) 전부 완료, `results/final/`은 최초 `presentation-freeze`
태그(commit `4096476`)로 동결됐었으나 **2026-08-12 사용자 승인으로 동결 해제 후 D6/D8/D9
관련 결과 갱신**(EXP-7, 아래 참고) — 이 시점 이후 `results/final/`을 인용할 때는 갱신본
기준. 진행 상태:

- [x] EXP-1 — Latency 벤치마크. ko probe 원문 30 + PGPrompts en 20 = 50건 × 3모델, batch=1
      순차 측정(`torch.cuda.synchronize()` 전후 호출, warmup 5회 제외). median/p95:
      LG3-1B 46.6/89.4ms, PolyGuard 677.2/722.2ms, SGuard 297.7/299.3ms. 출력 토큰 수 차이가
      커서(3.18 vs 31.6 vs 5) `ms_per_output_token`도 병기 — LG3-1B가 절대 latency는 가장
      빠르지만 토큰당으로는 PolyGuard가 더 느림. `exp/latency_bench.py`, `results/final/latency_raw.csv`,
      `results/final/latency_summary.csv`
- [x] EXP-2 — Response harmfulness 재현. **계획 수정**: 기존 `results/{model}_{lang}.csv`는
      PolyGuard/SGuard도 response=""(빈 응답)로 실행됐던 게 확인돼 재활용 불가 판정 →
      3모델 전부 재실행(en/ko 각 1689건, `response_harm_label` 정답 필드). LG3-1B는 user+assistant
      두 턴을 넣으면 chat template이 자동으로 Agent 평가 모드로 전환됨(별도 포맷 불필요).
      PGPrompts response 필드에 40k+ 토큰 이상치가 섞여 있어 1차 실행이 CUDA OOM으로 중단됨
      — `models.py`에 `truncation=True, max_length=2048` 추가(영향 샘플 <1%)로 해결.

      | model | axis | en F1 | ko F1 |
      |---|---|---|---|
      | PolyGuard | prompt | 0.8729 | 0.8158 |
      | PolyGuard | response | 0.7392 | 0.6333 |
      | LG3-1B | prompt | 0.7478 | 0.5693 |
      | LG3-1B | response | 0.4530 | 0.3669 |
      | SGuard-v1 | prompt | 0.8877 | 0.7538 |
      | SGuard-v1 | response | 0.7957 | 0.4748 |

      세 모델 다 response 축이 prompt 축보다 F1이 낮다 — response harmfulness가 prompt harmfulness보다
      어려운 판정임을 시사. LG3-1B는 response 축 낙폭이 가장 커서(en 0.75→0.45) response 평가에
      특히 취약. `exp/response_harm.py`, `results/final/response_harm_{model}_{lang}.csv`,
      `results/final/harm_axis_summary.csv`
- [x] EXP-3 — Confidence calibration. 기존 예측 CSV에 logprob/confidence가 저장돼 있지 않아
      (재사용 불가 확인) `models.py`의 `moderate()` confidence 로직(safe/unsafe 결정 토큰 2개
      softmax, SGuard는 5카테고리 네이티브 확률)을 `generate_batch()`로 확장, GPU 재실행
      필요성을 보고 후 승인받아 진행. PGPrompts en/ko 전량(1699건) + ko_probe(150행) 3모델
      confidence 포함 재실행(`results/{model}_{lang}_conf.csv`, `results/ko_probe_{model}_conf.csv`),
      F1은 기존 결과와 오차범위 내 일치(<1%p). `exp/calibration.py`로 모델별 예측을 합쳐
      10-bin(0.5~1.0) reliability diagram + ECE 계산:

      | model | ECE | n | low-sample bins(n<30) |
      |---|---|---|---|
      | LG3-1B | 0.1614 | 3547 | 0 |
      | PolyGuard | 0.1017 | 3548 | 1 |
      | SGuard-v1 | 0.1065 | 3548 | 0 |

      세 모델 다 과신(overconfident) — accuracy per bin이 대각선(perfect calibration) 아래에
      계속 위치. LG3-1B가 가장 심하게 과신(ECE 0.16, confidence 0.9대인데 accuracy는 0.6~0.8대).
      `results/final/calibration_{model}.csv`, `calibration_summary.csv`, `figures/reliability_*.png`
- [x] EXP-4 — McNemar 유의성 검정. GPU 재실행 없음, 기존 예측 CSV만 사용. 비교쌍: PGPrompts
      en/ko는 PolyGuard vs LG3-1B, ko_probe(150행)는 3모델 pairwise 3쌍(Bonferroni 보정,
      alpha=0.05/3). 불일치 셀 합(b+c)이 25 미만이면 exact binomial, 아니면 continuity-corrected
      chi-square:

      | pair | dataset | n | b | c | test | p_raw | p_adjusted | kappa | 우세 |
      |---|---|---|---|---|---|---|---|---|---|
      | PolyGuard vs LG3-1B | PGPrompts en | 1699 | 223 | 60 | chi2 | 5.98e-22 | 5.98e-22 | 0.651 | PolyGuard |
      | PolyGuard vs LG3-1B | PGPrompts ko | 1699 | 341 | 86 | chi2 | 1.00e-34 | 1.00e-34 | 0.439 | PolyGuard |
      | PolyGuard vs LG3-1B | ko_probe | 150 | 50 | 10 | chi2 | 4.78e-07 | 1.43e-06 | 0.125 | PolyGuard |
      | PolyGuard vs SGuard-v1 | ko_probe | 150 | 37 | 12 | chi2 | 6.07e-04 | 1.82e-03 | 0.300 | PolyGuard |
      | LG3-1B vs SGuard-v1 | ko_probe | 150 | 4 | 19 | exact | 2.60e-03 | 7.80e-03 | 0.385 | SGuard-v1 |

      전 쌍이 Bonferroni 보정 후에도 p<0.01 — 관측된 성능 차이가 우연이 아님. PolyGuard가
      LG3-1B에는 en/ko/ko_probe 전부에서 유의하게 우세하지만, SGuard-v1과는 ko_probe에서
      LG3-1B가 되레 우세(변형 프롬프트에 SGuard가 더 강건). 모델 간 판단 일치도(kappa)는
      전반적으로 낮음(0.13~0.65) — 세 모델이 서로 다른 기준으로 판정하고 있음을 시사.
      `exp/mcnemar.py`, `results/final/mcnemar.csv`
- [x] EXP-5 (선택) — 영어 변형 프로브. ko_probe base 30문장 중 문화특이적 카테고리(age 3개,
      origin 중 홍어/b09 1개 — 영어에 대응 슬러가 없어 confound 우려)를 제외한 26문장을 자동
      번역(`data/en_probe_base.csv`) 후 5-variant 적용(`exp/en_variants.py`,
      `data/en_probe.csv`, 26×5=130행). 3모델 GPU 재실행, parse 실패 0%. **번역투(학습 분포
      이탈) 취약성은 한국어 특유였다**: PolyGuard 26.67%→11.54%, SGuard 23.33%→3.85%로 영어에서
      크게 떨어짐. 반면 **우회표기·code-switching은 언어 무관하거나 오히려 영어에서 더 취약**:
      SGuard code-switching 16.67%→26.92%, LG3-1B 우회표기 6.67%→11.54%로 상승. 즉 변형
      취약성은 단일 원인(언어 특이 vs 언어 무관)으로 설명 안 되고 variant type마다 다른 메커니즘을
      시사(번역투=학습 분포 이탈은 언어별, leet-speak/code-switching=표층 패턴 의존은 언어 공통
      경향). base가 26개뿐이라 variant당 표본이 작아(en=ko probe 대비 4문장 적음) 해석에 주의.
      `exp/en_variants.py`, `exp/flip_compare_exp5.py`, `results/final/flip_rate_en_vs_ko.csv`
- [x] EXP-6 — 다국어(7개 언어) Track A/B 평가. `data/multilingual_base.csv`(360행) 3모델
      실행, parse 실패 0%. **Track A**(PG 홈그라운드 5개 언어, 순위 금지, n=150) F1:
      llamaguard 0.4789 / polyguard 0.7701 / sguard 0.8398. **Track B**(OOD 공정 비교
      7개 언어, 순위 메인, n=210) F1: llamaguard 0.6696 / polyguard 0.7939 / sguard 0.7815.
      **예상과 반대되는 결과**: PolyGuard가 자기 이름을 딴 데이터(Track A)에서 유리할
      것이라는 우려(`docs/handoff_multilingual.md` §Track A/B 재설계 배경)와 달리, llamaguard는
      오히려 Track A에서 더 낮고(0.48→0.67, +0.19p) polyguard는 두 트랙에서 거의 동일(0.77→0.79,
      +0.02p) — PG 데이터 자체가 특별히 유리하게 작용하지 않음. sguard만 A>B(0.84→0.78, -0.06p)로
      약한 홈 이점을 보임. Track B 언어별로는 세 모델 다 id(인도네시아어)에서 가장 약함(F1
      0.40~0.69) — RTP-LX id의 continuation-stem 형태 문제(§`docs/handoff_multilingual.md`
      태스크 형태 캐비어트) 영향 가능성, 후속 분석 필요. `results/multilingual_{model}.csv`
      (`eval.py --model {model} --multilingual`)
- [x] EXP-6 후속 — id(인도네시아어) 오탐/미탐 정성 분석 (`docs/failure_cases_id.md`). id의
      낮은 F1은 순수 언어 갭이 아니라 **taxonomy mismatch**가 더 크게 기여: 세 모델 공통
      미탐 7건 중 4건이 개인정보 캐묻기(Information Hazards, 예 "내 동생 혈액형이 뭐야"),
      3건이 지역 문화·종교 민감성(Region-specific Sensitivity) — 둘 다 표면 언어가 중립적이라
      폭력/증오/성 위주 taxonomy로는 안 걸림. `Information Hazards` recall은 3모델 전부 0%.
      반면 `Malicious Uses`(고전적 유해 요청)는 polyguard recall 1.00 — 카테고리가 학습
      taxonomy와 겹치면 정상 작동. FP는 대부분 RTP-LX id의 continuation-stem 형태 + 경계선
      toxicity 라벨링 문제로 확인(모델 결함이라기보다 데이터 이슈).
- [x] EXP-7 — 발표 자료 집계(`results_summary.md`) 중 발견된 결손 보강. `results/final/`
      동결 해제(§잔여 실험 상단 참고) 후: (1) McNemar를 PGPrompts en/ko에도 3모델 pairwise
      3쌍 전부로 확장 — **PolyGuard vs SGuard-v1이 PGPrompts en에서 Bonferroni 보정 후
      유의하지 않음**(p_adjusted=0.124) 발견, LG3 vs SGuard는 en/ko/ko_probe 전부 SGuard
      우세로 일관. (2) GPU 모델명(RTX 3090) 확인. (3) 7개 언어(es/hi/ar/ru/th/vi/id)
      variant flip rate 신규 생성 — 우회표기/띄어쓰기/번역투/code-switching을 문자 체계별
      규칙(Latin leetspeak, Cyrillic homoglyph, Arabic tatweel, Devanagari ZWJ, Thai ZWSP)
      + Google Translate 왕복번역으로 만듦(**원어민 미검증**, `exp/multilingual_variants.py`).
      태국어 code-switching은 띄어쓰기 부재로 56.7%가 no-op. (4) Track A 5개 언어(es/hi/th/ar/ru)
      response harmfulness 추가 — en/ko와 동일하게 response F1 < prompt F1 패턴 재확인,
      llamaguard는 ar/es에서 response F1 0.0으로 다국어에서 낙폭이 더 극단적. 상세는
      `results_summary.md`, `docs/journal/2026-08-12.md` 참고.

## 빠른 시작

```bash
conda create -n guard python=3.11 -y && conda activate guard
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets fastapi uvicorn pandas scikit-learn
```

t2i-lab conda env는 재사용하지 않는다 (diffusers 핀 때문에 transformers 버전 충돌 가능).
서버(`172.10.5.23`)에 `/data` 마운트가 없어 `HF_HOME`은 지정하지 않고 기본 캐시
(`~/.cache/huggingface`)를 t2i-lab과 공유해서 쓴다 — 자세한 내용은 `docs/PLAN.md` Phase 0 참고.

## 참고 논문

`../papers/`에 이미 받아둔 관련 PDF: `polyguard.pdf`, `limaguard.pdf`, `mrguard.pdf`,
`sealguard.pdf`, `sguard.pdf`. **`limaguard.pdf`가 파일명과 달리 Llama Guard 원문(2312.06674)이다**
— 확인 완료. `sguard.pdf`는 Phase 5~7에서 세 번째 비교 모델(SGuard-v1)로 실제 사용 예정. 세 모델
taxonomy 정렬은 `docs/taxonomy_map.md` 참고.
