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
│   └── PLAN.md    # Phase 0~5 전체 계획
└── results/       # {model}_{lang}.csv 평가 결과
```

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
- [ ] Phase 7 — Demo 스키마 구현 (`risk → category → confidence → reason`, `/moderate` 응답 확장)

Phase 5~7은 정량 재현이 아니라 정성 분석 축이다. 세 번째 비교 모델로 SGuard-v1(한국어 특화,
confidence 네이티브 제공) 추가 완료 — 상세 계획은 `docs/PLAN.md` Phase 5~7, 실패 사례는
`docs/failure_cases.md` 참고.

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
