# Guard-Lab 다국어 확장 — 데이터셋 다운로드 & 샘플 추출 핸드오프

> 에이전트용 지시서. 목적: 7개 언어에 대해 언어별 base 문장 30개(unsafe 20 + safe 10)를
> 실존 데이터셋에서 추출해 통일 CSV로 저장한다. **harmful 문장을 생성하지 말 것** —
> 이미 라벨링된 공개 데이터셋에서 가져오기만 한다.

---

## 0. 대상 언어 (7개)

| 코드 | 언어 | 문자 | 지역 | 1순위 소스 |
|---|---|---|---|---|
| es | Spanish | Latin | 남미/유럽 | PolyGuardPrompts |
| hi | Hindi | Devanagari | 남아시아 | PolyGuardPrompts |
| th | Thai | Thai | 동남아 | PolyGuardPrompts |
| ar | Arabic | Arabic (RTL) | 서아시아 | PolyGuardPrompts |
| ru | Russian | Cyrillic | 유럽 | PolyGuardPrompts |
| id | Indonesian | Latin | 동남아 | RTP-LX / IndoSafety |
| vi | Vietnamese | Latin(+diacritics) | 동남아 | RTP-LX / MultiJail |

es/hi/th/ar/ru → PolyGuard 하나로 스키마 통일. id/vi만 PolyGuard 미포함이라 별도 소스.

---

## 1. 추출 목표 스펙 (출력 형식)

언어별로 아래를 만족하는 **base 문장(원문)** 만 추출한다. 변형(우회표기/띄어쓰기/번역투/
code-switching)은 데이터셋에 없음 — 이건 추출 이후 별도 스크립트로 거는 결정론적 변환이므로
**이 작업 범위 밖**이다. 여기서는 "원문 variant"에 해당하는 base만 확보한다.

- 언어당 **unsafe 20개 + safe 10개 = 30개** (기존 bench 비율 유지)
- 최종 출력: 단일 CSV `multilingual_base.csv`, 컬럼:

```
base_id        # 통일 식별자. PolyGuard는 원본 id 재사용, 그 외는 dataset명_원본id
lang           # es / hi / th / ar / ru / id / vi
text           # 프롬프트 원문 (base)
label          # "unsafe" 또는 "safe" (이진 통일 라벨)
source_dataset # 예: polyguard / rtp-lx / indosafety / multijail / aya
source_id      # 원본 데이터셋의 행 id
source_label   # 원본 라벨 원문 (매핑 전, 감사용으로 보존)
```

**중요:** `source_label`(원본)을 반드시 보존할 것. 데이터셋마다 라벨 의미가 달라서
(§5 참고) 나중에 매핑 규칙을 재검토해야 할 수 있음.

---

## 2. Tier 1 — PolyGuardPrompts (es, hi, th, ar, ru)

### 설명
17개 언어 다국어 안전 벤치마크. WildGuardMix(영어)를 human-verified 기계번역 + 자연발생
human-LLM 대화로 확장. prompt harmfulness / response harmfulness / response refusal 3축
라벨 보유. **재훈이 en/ko로 이미 돌린 PGPrompts 파이프라인과 동일 스키마** → 무마찰.

### 출처
- HF Dataset: `ToxicityPrompts/PolyGuardPrompts`
- Split: `test` 하나 (29.3k rows), Format: parquet, License: **CC-BY-4.0**
- Paper: arXiv 2504.04377 (COLM 2025)

### 스키마 (확인 완료)
```
prompt (str)                     # ← 프롬프트 원문. 이걸 text로 사용
response (str)                   # LLM 응답 (prompt축만 볼 거면 미사용)
prompt_harm_label (str)          # "harmful" / "unharmful"  ← 이걸 gold 이진 라벨로 사용
prompt_label (str)               # "safe" / "unsafe" (동일 정보의 다른 표기)
adversarial (bool)               # 적대적 프롬프트 여부 (분석용으로 함께 보존 권장)
subcategory (str, 15종)          # 세부 카테고리
prompt_categories (str)          # 위반 카테고리
language (str, 17종)             # 전체 이름 표기: "Spanish","Hindi","Thai","Arabic","Russian"
id (int)                         # ★ 17개 언어에 걸쳐 공유되는 parallel id
response_harm_label, response_refusal_label, *_agreement ...  # 이번엔 미사용
```

### ★ Parallel id 활용 (권장 추출 전략)
`id`가 언어 간 공유됨 = 같은 id는 같은 문장의 언어별 번역. 따라서:

1. **한 언어(예: Spanish)에서 먼저 id를 30개 뽑는다** — `prompt_harm_label=="harmful"` 20개
   + `"unharmful"` 10개.
2. **그 30개 id를 나머지 4개 언어(hi/th/ar/ru)에서 그대로 필터링**해서 가져온다.
3. 결과: 5개 언어가 **content-controlled**(동일 문장 집합의 번역) → 언어 간 F1 차이가
   순수하게 언어 효과임을 보장. 기존 bench_v1의 fixed-benchmark 철학과 동일.

---

## 3. Tier 2 — id, vi (PolyGuard 미포함)

단일 소스로 id+vi를 둘 다, safe+unsafe 모두, 동일 스키마로 주는 데이터셋은 없다.

### 3-A. RTP-LX (기본값, id+vi 둘 다 커버)
- **설명:** RealToxicityPrompts를 원어민이 transcreation + 주석. toxicity continuation
  라벨(1~5)이라 prompt-harmfulness 이진이 아님 — 임계값 매핑 필요.
- **출처:** GitHub `microsoft/RTP-LX`, HF `adewynter/RTP-LX`(gated).

### 3-B. IndoSafety (id 고품질 대안)
- **설명:** 인도네시아어 원어민 작성 문화기반 안전 벤치마크.
- **출처:** arXiv 2506.02573. HF `GoToCompany` org 계열로 추정.

### 3-C. MultiJail (vi 보조, harmful만)
- **설명:** 315개 영어 harmful 프롬프트를 원어민 번역으로 10개 언어 확장(vi 포함, id는 없음).
- **출처:** GitHub `DAMO-NLP-SG/multilingual-safety-for-LLMs` (ICLR 2024).
- **한계:** unsafe만 존재. safe는 RTP-LX로 보충.

### 권장 조합
- **id:** IndoSafety(native) unsafe 20 + safe 10. 접근 안 되면 RTP-LX id.
- **vi:** MultiJail vi unsafe 20 (native) + RTP-LX vi safe 10.

---

## 4. (선택) Native 통제군 — Aya Redteaming

es/hi/ar/ru 원어민 작성 harmful 프롬프트(HF `CohereLabs/aya_redteaming`). "번역 vs 네이티브"
recall 비교용 별도 슬라이스. 선택 항목.

---

## 5. 라벨 의미론 통일 규칙 (⚠️ 최대 리스크)

| 소스 | 원본 라벨 | → 통일 라벨 매핑 |
|---|---|---|
| PolyGuard | prompt_harm_label: harmful/unharmful | harmful→unsafe, unharmful→safe (1:1) |
| RTP-LX | toxicity 주석/점수 | 임계값 매핑 필요 (상위=unsafe, benign=safe, 중간 제외) |
| MultiJail | (전부 harmful) | 전부 unsafe |

**리포팅 분리 원칙:** Tier 1(es/hi/th/ar/ru)끼리는 라벨·스키마 동일 → 한 표에서 직접 비교
OK. Tier 2(id/vi)는 소스가 달라 라벨 의미가 이질적 → 별도 표/섹션으로 리포팅. **Tier 1과
Tier 2를 하나의 F1 랭킹 표에 합치지 말 것.**

---

## 6. 에이전트 실행 순서 (체크리스트)

```
1. PolyGuardPrompts 로드 → es 앵커 30 id 선정 → 5개 언어 추출
2. RTP-LX(또는 IndoSafety/MultiJail) 로드 → id/vi 각 30 추출
3. (선택) Aya redteaming → es/hi/ar/ru harmful 슬라이스 추출
4. 전부 §1 통일 스키마로 concat → multilingual_base.csv 저장
5. 요약 리포트: 언어별 unsafe/safe 카운트, 소스 분포, 라이선스 목록 출력
```

---

## 7. 파이프라인 밖(다음 단계, 참고용)

이 문서 범위는 **base 문장 확보까지**. 이후는 별도 작업:
- 5개 variant 변환 스크립트(언어별 재정의)
- 3개 모델 재실행
- variant별 flip rate + 언어별 오탐/미탐 분석

---

## 실행 결과 v1 (2026-08-12, 폐기됨)

최초 실행은 Tier1(PolyGuard 5개 언어) + Tier2(id/vi, PolyGuard 미포함 언어만 별도 소스)
구조였음. 사용자가 리뷰 후 "Tier2가 공정하지 않다"고 판단 — PolyGuard 자체가 3모델 중
하나(PolyGuard-Qwen-Smol)의 이름을 딴 데이터셋이라 Tier1 F1로 순위를 매기면 그 모델에
유리하게 편향됨. 이 구조는 폐기하고 아래 v2(Track A/B)로 대체.

## 실행 결과 v2 — Track A/B 병행 (2026-08-12)

`exp/extract_multilingual_base.py` → `data/multilingual_base.csv` (360행: TrackA 150 + TrackB
210) 재구성 완료. 스키마에 `track`(A/B) 컬럼 추가.

**Track A (PG 홈그라운드, 5개 언어, 순위 금지)**: PolyGuardPrompts 그대로, v1의 Tier1과 동일
(parallel id로 content-controlled, 영어 미번역 통과 없음 확인).

**Track B (OOD 공정 비교, 7개 언어, 순위 메인)**: unsafe는 언어별 native harmful 소스, safe는
전 언어 RTP-LX로 통일.

| 언어 | unsafe 소스 | safe 소스 |
|---|---|---|
| es, hi, ar, ru | Aya Redteaming (HF `CohereLabs/aya_redteaming`, 언어별 split) | RTP-LX |
| th, vi | MultiJail (같은 20개 id → th/vi가 서로도 content-controlled됨, 우연한 보너스) | RTP-LX |
| id | IndoSafety | RTP-LX |

**v1 대비 바뀐 점**:
- v1은 하나의 소스 조합(PolyGuard 5개 + RTP-LX/MultiJail 2개)이었고 모델 순위용 표에 PG가
  섞여 있었음. v2는 PG를 Track A로 완전히 분리하고, Track B는 세 모델 모두에게 "제3자"인
  소스로만 구성.
- **IndoSafety를 실제로 찾음**: HF에는 없지만(검색 실패) 논문 저자 GitHub
  `falensiazmi/IndoSafety`의 `dataset/IndoSafety-Eval-1.xlsx`(2514행, 원어민 작성 formal
  Indonesian harmful prompt)를 사용. safe 라벨이 없는 데이터셋이라 id의 safe는 사용자 지시대로
  RTP-LX id로 보충. **주의**: 비공식 배포본(논문은 ACL Rolling Review 심사 중, 정식 공개 링크
  없음) — 라이선스 명시 안 됨, 발표/공개 전 저자에게 확인 권장.
- **RTP-LX 역할 축소**: v1에서는 id/vi 콘텐츠 자체의 절반이었지만(continuation stem이라
  태스크 형태 문제), v2에서는 전 언어 safe 보충 전용으로만 씀. safe 문장은 "harmful 여부"만
  중요해서 stem 형태여도 문제가 상대적으로 적음(대신 §5 라벨 의미론 차이는 여전히 남음 — 각주
  필수).
- **adversarial 컬럼**: Track A만 값 보존, Track B는 원본에 해당 필드 없어 공란.

**리포팅 원칙 확정**: Track A와 Track B는 별도 표. 메인 결과는 **세 모델을 A/B 양쪽에서 돌려
모델별 delta(A 대비 B 낙폭) 비교** — "PG 우위가 OOD에서도 유지되는가"가 핵심 질문.
