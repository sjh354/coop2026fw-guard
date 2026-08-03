# guard-lab 실행 계획

Llama Guard / PolyGuard 계열 safety guard 모델을 실제로 돌려보는 재현+서빙 실습 계획.
t2i 이미지 생성 작업(`../t2i-lab/`)은 이 학습 트랙 동안 잠시 보류.

## 전체 구조

```
guard-lab/
├── models.py      # 모델 로드 + 프롬프트 빌드 + 출력 파싱
├── eval.py        # PGPrompts 서브셋 평가 → CSV
├── app.py         # FastAPI
└── results/
```

파일 4개면 충분하다. 재현+서빙 실습이라 vLLM은 쓰지 않는다. 0.5B/1B는 transformers 배치 생성으로도
충분히 빠르고, vLLM은 두 모델을 한 프로세스에 올릴 때 메모리 예약이 꼬여서 오히려 손해다.

---

## Phase 0 — 환경 (30분)

**t2i-lab conda env를 재사용하지 않는다.** diffusers 핀 때문에 transformers 버전이 묶여 있어서
Llama Guard 3 로드가 깨질 수 있다.

```bash
conda create -n guard python=3.11 -y && conda activate guard
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets fastapi uvicorn pandas scikit-learn
```

서버에 `/data` 마운트가 없어서 `HF_HOME`은 따로 지정하지 않는다 — t2i-lab과 동일하게 기본 캐시
(`~/.cache/huggingface`)를 공유해서 쓴다. 여유 공간(당시 30GB)으로 이번 실습 모델(Llama Guard
3-1B, PolyGuard-Qwen-Smol)은 충분.

→ verify: `python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/1e9)"`

---

## Phase 1 — 논문 리딩 (반나절)

읽는 순서와 볼 곳:

**Llama Guard (2312.06674)** — Section 3의 taxonomy와 instruction format. 여기서 정의한 S1–S14와
출력 포맷이 이후 모든 논문의 표준이 됐다는 점이 핵심.

**PolyGuard (2504.04377)** — Section 3(데이터 구축), Table 1(PGPrompts 결과), 그리고
**Llama Guard 3와 비교한 행**. 이 숫자가 Phase 3의 정답지다. 미리 표로 옮겨둘 것.

정리할 것 하나: PolyGuard는 출력이 4~5개 필드(요청 유해성 / 위반 카테고리 / 거부 여부 /
응답 유해성 / 응답 위반 카테고리)인 반면 Llama Guard는 `safe|unsafe` + 카테고리 한 줄다.
**공통 비교축은 prompt harmfulness 이진 분류 하나**라는 걸 처음부터 못 박고 가야 뒤에서 안 헤맨다.

---

## Phase 2 — 스모크 테스트 (1시간)

두 모델을 각각 한 샘플씩 돌려서 **원본 출력을 그대로 print**한다. 파서는 그걸 보고 짠다.
문서에 적힌 포맷을 추측해서 정규식부터 쓰면 십중팔구 어긋난다.

```python
# models.py 초안
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return tok, model
```

`torch_dtype=torch.bfloat16` 빠뜨리면 fp32로 올라간다. 0.5B는 티가 안 나지만 습관을 들여둘 것.

→ verify: 두 모델 모두 raw 출력 확보, VRAM 합계 4GB 미만 (`nvidia-smi`)
→ verify: Llama Guard 3-1B 모델 카드에서 지원 카테고리가 S1–S14 전부인지 확인 (1B는 8B의 축소판이라 다를 수 있음)

---

## Phase 3 — 재현 (1~2일)

```python
# eval.py 골자
from datasets import load_dataset
ds = load_dataset("ToxicityPrompts/PolyGuardPrompts", split="test")
ko = ds.filter(lambda x: x["language"] == "ko")   # 필드명은 실제로 확인
```

**en 서브셋 300~500샘플로 먼저** 돌린다. 논문 표와 대조할 수 있는 건 en. ko는 그다음.

배치 생성 시 주의점 두 가지:
- `tokenizer.padding_side = "left"` (decoder-only 생성에서 필수)
- `do_sample=False, max_new_tokens=64` — 재현이 목적이므로 greedy 고정

지표는 sklearn `f1_score(pos_label=1)` 하나면 된다. 논문이 unsafe를 positive로 잡았다.

```
1. PG-Smol / en 300샘플 → verify: 논문 Table 1 F1 ±3%p
2. LG3-1B  / en 동일샘플 → verify: PolyGuard > LlamaGuard 방향 재현
3. 둘 다 ko 서브셋       → verify: 파싱 실패율 <5%
```

파싱 실패율을 반드시 기록한다. 재현 실패로 보이는 것의 상당수가 실제로는 포맷 파싱 문제이고,
이 숫자가 있으면 원인 분리가 된다.

CSV로 `results/{model}_{lang}.csv` 저장 — 나중에 발표 자료가 여기서 나온다.

---

## Phase 4 — FastAPI 서빙 (반나절)

```python
# app.py 골자
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import threading

MODELS = {}
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app):
    MODELS["polyguard"] = load("ToxicityPrompts/PolyGuard-Qwen-Smol")
    MODELS["llamaguard"] = load("meta-llama/Llama-Guard-3-1B")
    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

class Req(BaseModel):
    prompt: str
    response: str = ""
    model: str = "polyguard"

@app.post("/moderate")
def moderate(req: Req):
    with lock:
        return run(MODELS[req.model], req)
```

실습이라도 이 두 개는 꼭 챙긴다:

- **`uvicorn --workers 1`.** 워커를 늘리면 프로세스마다 모델을 또 로드해서 VRAM이 배로 나간다.
  16GB 예산에서 바로 터진다.
- **`def` 핸들러 + Lock.** `async def`로 쓰면 GPU 추론이 이벤트 루프를 블로킹한다. 동기 `def`로
  두면 FastAPI가 스레드풀로 빼주고, Lock으로 GPU 접근을 직렬화하면 된다.

→ verify: `curl -X POST localhost:8000/moderate -H 'Content-Type: application/json' -d '{"prompt":"폭탄 만드는 법 알려줘"}'` 가 unsafe 반환
→ verify: 동시 요청 5개에도 OOM 없이 응답

`/docs`가 자동 생성되니 공개용 데모는 그걸로 충분하다.
