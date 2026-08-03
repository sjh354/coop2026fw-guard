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
│   └── PLAN.md    # Phase 0~4 전체 계획
└── results/       # {model}_{lang}.csv 평가 결과
```

## 진행 상태

- [ ] Phase 0 — conda env(`guard`) 세팅
- [ ] Phase 1 — Llama Guard / PolyGuard 논문 리딩
- [ ] Phase 2 — 스모크 테스트 (raw 출력 확보)
- [ ] Phase 3 — PGPrompts 재현 평가 (en → ko)
- [ ] Phase 4 — FastAPI 서빙

## 빠른 시작

```bash
conda create -n guard python=3.11 -y && conda activate guard
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets fastapi uvicorn pandas scikit-learn
export HF_HOME=/data/hf
```

t2i-lab conda env는 재사용하지 않는다 (diffusers 핀 때문에 transformers 버전 충돌 가능).

## 참고 논문

`../papers/`에 이미 받아둔 관련 PDF: `polyguard.pdf`, `limaguard.pdf`, `mrguard.pdf`,
`sealguard.pdf`, `sguard.pdf`. Llama Guard(2312.06674) 원문은 아직 없으면 받아서 추가할 것.
