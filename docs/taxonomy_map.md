# Taxonomy Map — Llama Guard 3 vs PolyGuard vs SGuard-v1

Phase 1 산출물. PLAN.md Phase 5~7(정성 분석)에서 "실패"를 실패라고 부르려면 세 모델이 애초에
무엇을 unsafe로 정의했는지 알아야 한다. 이 문서는 세 모델의 카테고리 정의를 정렬한 표다.

## 중요 정정: "S1–S14가 Llama Guard 원문의 taxonomy"라는 이전 가정은 틀렸다

`limaguard.pdf`(2312.06674, 2023-12-07)를 읽어보니 이 논문이 정의한 taxonomy는 **O1–O6 6종**
(Violence & Hate, Sexual Content, Guns & Illegal Weapons, Regulated/Controlled Substances,
Suicide & Self-Harm, Criminal Planning)이다. 우리가 실제로 서빙 중인 `meta-llama/Llama-Guard-3-1B`
와 PolyGuard가 쓰는 **S1–S14 taxonomy는 이후 버전(Llama Guard 2/3)에서 MLCommons Safety
Taxonomy에 맞춰 재정의된 것**이고, 원문 논문에는 나오지 않는다. `docs/PLAN.md` Phase 1의
"S1–S14가 이후 모든 논문의 표준" 서술은 이 문서 기준으로 정정한다 — 정확히는 "**Llama Guard 2/3
이후 MLCommons 정렬 taxonomy가 표준이 됐다**"이다.

## 세 taxonomy 계보

```
MLCommons Safety Taxonomy (Ghosh et al. 2025a, AILuminate)  — 12종 baseline
        │
        ├── Llama Guard 3 / PolyGuard: 12종 + Elections + Code Interpreter Abuse = S1–S14
        │     (LG3-1B는 8B의 축소판이라 S14 미지원 — Phase 2에서 확인)
        │
        └── SGuard-v1 ContentFilter: 12종을 5개 상위 카테고리로 통합
```

## 카테고리 정렬표

| S-code | LG3 / PolyGuard (S1–S14) | MLCommons 12종 원본 | SGuard-v1 5종 (label) |
|---|---|---|---|
| S1 | Violent Crimes | Violent Crimes | Violence and Hate (`Violence`) |
| S2 | Non-Violent Crimes | Non-Violent Crimes | Illegal and Criminal Activities (`Crime`) |
| S3 | Sex Crimes | Sex-Related Crimes | Sexual Content and Exploitation (`Sexual`) |
| S4 | Child Exploitation | Child Sexual Exploitation | Sexual Content and Exploitation (`Sexual`) |
| S5 | Defamation | Defamation | Manipulation and Societal Harm (`Manipulation`) |
| S6 | Specialized Advice | Specialized Advice | Privacy and Sensitive Information Misuse (`Privacy`)† |
| S7 | Privacy | Privacy | Privacy and Sensitive Information Misuse (`Privacy`) |
| S8 | Intellectual Property | Intellectual Property | Privacy and Sensitive Information Misuse (`Privacy`)† |
| S9 | Indiscriminate Weapons | Indiscriminate Weapons | Illegal and Criminal Activities (`Crime`) |
| S10 | Hate | Hate | Violence and Hate (`Violence`) |
| S11 | Self-Harm | Suicide & Self-Harm | **매핑 불명확** (아래 참고) |
| S12 | Sexual Content | Sexual Content | Sexual Content and Exploitation (`Sexual`) |
| S13 | Elections | *(LG3/PolyGuard 자체 추가, MLCommons 12종에 없음)* | **대응 카테고리 없음** |
| S14 | Code Interpreter Abuse | *(LG3/PolyGuard 자체 추가)* | **대응 카테고리 없음**, LG3-1B 자체도 미지원 |

† SGuard-v1 Table 1 정의문에 "provides unqualified guidance on health, legal, or financial
matters"(Specialized Advice에 해당)와 "leaks proprietary or confidential data"(Intellectual
Property에 해당)가 Privacy 카테고리 정의 안에 명시적으로 포함돼 있어 그쪽으로 매핑했다.

### 매핑이 안 되는 지점 (Phase 6 분석 소재)

- **S11 Self-Harm** — SGuard-v1 논문 Table 1의 5개 카테고리 정의 어디에도 자해가 명시되지 않는다.
  "Violence and Hate"는 정의상 "타인(others)"을 향한 위해로 한정돼 있어 자해가 깔끔하게 들어가지
  않는다. SGuard-v1 실제 서빙 결과를 봐야 어느 카테고리로 흡수됐는지(혹은 아예 unsafe로 못 잡는지)
  확인 가능 — Phase 6 관찰 대상.
- **S13 Elections, S14 Code Interpreter Abuse** — LG3/PolyGuard가 MLCommons 12종에 자체
  추가한 카테고리라 SGuard-v1(MLCommons 12종 기반)에는 대응 카테고리가 원천적으로 없다. 세 모델
  비교 시 이 두 카테고리에 해당하는 입력은 SGuard-v1이 구조적으로 다른 카테고리에 넣거나 safe로
  낼 가능성이 있다 — taxonomy 설계 차이지 버그가 아니다.

## K-MHaS(Phase 5 ko_probe 소스) 카테고리와의 관계

`data/ko_probe.csv`의 `category` 컬럼(K-MHaS 8종: profanity, age, origin, religion, physical,
gender, politics, race)은 **toxicity/abuse 분류**지 위 세 모델의 **safety 분류(S1-S14 계열)**와
축 자체가 다르다. PLAN.md Phase 1에 이미 적어둔 예시("이 나쁜 녀석아!"는 Abuse/Slang이지 S1-S14
safety 카테고리가 아님)와 같은 구조 — K-MHaS의 `age`(틀딱 등 연령 비하), `physical`(외모 비하)은
S1-S14 어디에도 깔끔히 대응하지 않고, `gender`/`race`/`religion`/`origin`은 S10 Hate에 가장
가깝게 걸치지만 완전히 겹치지는 않는다. **즉 ko_probe의 harmful 라벨 20개 중 상당수가 세 모델
전부에서 safe로 판정되더라도 그 자체가 "safety guardrail이 애초에 toxicity/abuse를 커버하지
않는다"는 결과이지, flip이 아니다** — Phase 6 실패 사례 분석에서 flip(원문 대비 예측 변화)과
이 구조적 미탐(원문부터 safe)을 반드시 구분해서 봐야 한다.

## 모델별 출력 포맷 차이 (참고)

| | Llama Guard 3 | PolyGuard | SGuard-v1 ContentFilter |
|---|---|---|---|
| 형식 | `safe` \| `unsafe`\n`S1,S3,...` | 4~5필드 텍스트(Harmful request, 위반 카테고리, 거부 여부, 응답 유해성 등) | 5개 카테고리 각각 safe/unsafe + confidence score |
| 공통 비교축 | prompt harmfulness 이진 분류 (`models.py`의 `parse_prompt_harm`) | 좌동 | 좌동 — 통합 시 5카테고리 중 하나라도 unsafe면 harmful로 정규화 필요 |

## 모델 레퍼런스

- Llama Guard 3-1B: `meta-llama/Llama-Guard-3-1B`
- PolyGuard-Qwen-Smol: `ToxicityPrompts/PolyGuard-Qwen-Smol`
- SGuard-v1 ContentFilter: `SamsungSDS-Research/SGuard-ContentFilter-2B-v1` (Granite-3.3-2B-Instruct
  베이스, Apache-2.0, 영어+한국어 bilingual 학습 — 12개 언어 지원 베이스 모델이지만 논문이 성능을
  검증한 언어는 영어/한국어뿐)
- (JailbreakFilter `SamsungSDS-Research/SGuard-JailbreakFilter-2B-v1`는 별도 컴포넌트, 이번
  Phase 5~7 범위 밖 — 필요시 언급만)

## 출처

- Llama Guard: Inan et al., *Llama Guard: LLM-based Input-Output Safeguard for Human-AI
  Conversations* (arXiv:2312.06674, 2023-12-07) — 로컬 `../papers/limaguard.pdf`
- PolyGuard: Kumar et al., *PolyGuard: A Multilingual Safety Moderation Tool for 17 Languages*
  (COLM 2025, arXiv:2504.04377) — 로컬 `../papers/polyguard.pdf`
- SGuard-v1: Lee et al., *SGuard-v1: Safety Guardrail for Large Language Models*
  (arXiv:2511.12497, 2025-11-16, Samsung SDS) — 로컬 `../papers/sguard.pdf`
