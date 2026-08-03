# CLAUDE.md

This file guides Claude Code when working in this repo.

## What this is

Safety guard 모델(Llama Guard, PolyGuard) 재현 + 서빙 실습. t2i-lab(`../t2i-lab/`)과는 별개
프로젝트/별개 conda env — 이미지 생성 작업은 이 트랙 동안 보류 중.

## Where things live

- **`docs/PLAN.md`** — Phase 0~4 전체 실행 계획(환경 세팅 / 논문 리딩 / 스모크 테스트 / 재현 평가 /
  FastAPI 서빙). 지금 뭘 해야 하는지, 각 단계의 verify 조건이 뭔지 확인할 때 참조.
- 새로 만드는 문서 파일(.md)은 전부 `docs/` 아래에 쓴다. 최상위에는 `CLAUDE.md`와 `README.md`만
  둔다 — 둘 다 문서 목록이 늘어나도 얇게 유지하고, 실제 내용/세부 계획은 `docs/`로 위임한다.
- 코드 4파일(`models.py`, `eval.py`, `app.py`) + `results/`는 최상위에 그대로 둔다 — 실습 규모상
  더 쪼갤 필요 없음(`docs/PLAN.md` 참고).

## Conda env

`guard` env를 쓴다. t2i-lab의 `t2i-*` env를 재사용하지 않는다 — diffusers 버전 핀 때문에
transformers 버전이 묶여 있어서 Llama Guard 3 로드가 깨질 수 있다. 세팅 커맨드는
`docs/PLAN.md` Phase 0 참고.

## Results

`results/{model}_{lang}.csv` 형식으로 평가 결과를 저장한다. 재현 시도마다 덮어쓰지 말고
구분해서 남길 것 — 발표 자료가 여기서 나온다.
