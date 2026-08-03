# CLAUDE.md

This file guides Claude Code when working in this repo.

## What this is

Safety guard 모델(Llama Guard, PolyGuard) 재현 + 서빙 실습. t2i-lab(`../t2i-lab/`)과는 별개
프로젝트/별개 conda env — 이미지 생성 작업은 이 트랙 동안 보류 중.

## Commands

`guard` conda env를 쓴다 (t2i-lab의 `t2i-*` env는 재사용하지 않는다 — diffusers 버전 핀 때문에
transformers 버전이 묶여 있어서 Llama Guard 3 로드가 깨질 수 있다). 세팅 커맨드는
`docs/PLAN.md` Phase 0 참고.

```bash
conda activate guard
python eval.py          # PGPrompts 서브셋 평가 → results/{model}_{lang}.csv
python app.py           # FastAPI 서빙 (uvicorn --workers 1 고정, 워커 늘리면 VRAM 배로 나감)
```

There is no build/lint/test step — this is a script-driven eval/serving harness, not a library.

## Where things live

- **`docs/PLAN.md`** — Phase 0~4 전체 실행 계획(환경 세팅 / 논문 리딩 / 스모크 테스트 / 재현 평가 /
  FastAPI 서빙). 지금 뭘 해야 하는지, 각 단계의 verify 조건이 뭔지 확인할 때 참조.
- 새로 만드는 문서 파일(.md)은 전부 `docs/` 아래에 쓴다. 최상위에는 `CLAUDE.md`와 `README.md`만
  둔다 — 둘 다 문서 목록이 늘어나도 얇게 유지하고, 실제 내용/세부 계획은 `docs/`로 위임한다.
- 코드 4파일(`models.py`, `eval.py`, `app.py`) + `results/`는 최상위에 그대로 둔다 — 실습 규모상
  더 쪼갤 필요 없음(`docs/PLAN.md` 참고).

## Results

`results/{model}_{lang}.csv` 형식으로 평가 결과를 저장한다. 재현 시도마다 덮어쓰지 말고
구분해서 남길 것 — 발표 자료가 여기서 나온다.

## GPU server & keeping everything in sync

GPU 작업은 t2i-lab과 같은 서버 **`ubuntu@172.10.5.23`**을 쓴다 (`ssh -i /Users/sjh354/.ssh/id_ed25519
ubuntu@172.10.5.23`) — 현재 유일한 GPU 서버이고, t2i 생성/채점/리라이팅 작업과 디스크(97GB)를
공유한다. **큰 모델(Llama Guard, PolyGuard 가중치)을 받거나 `guard` env를 새로 만들기 전에는
반드시 `df -h /`로 여유 공간부터 확인**할 것.

이 repo는 서버 홈 디렉토리 기준 `guard/` 폴더에 clone되어 있다 — 이후에는
t2i-lab과 동일한 동기화 규칙을 따른다:

**로컬에서 코드 변경 후** (`models.py`/`eval.py`/`app.py`/`docs/PLAN.md` 수정 등):
1. 로컬에서 commit & push.
2. 서버에 ssh로 접속해 `git pull`로 반영 (실행 전에 최신 코드가 있어야 함).

**서버에서 실행한 뒤** (평가/서빙 세션 실행 등):
1. 서버에서 결과(`results/*.csv` 등) commit & push.
2. 로컬에서 `git pull`로 받아와서 두 체크아웃을 동일하게 유지.

목표: 새 작업을 시작하기 전에 로컬과 서버가 항상 같은 최신 커밋을 가리키고 있어야 한다.
