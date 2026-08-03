# FastAPI 서빙 (PLAN.md Phase 4 참고)
# uvicorn --workers 1 고정 — 워커를 늘리면 프로세스마다 모델을 또 로드해서 VRAM이 배로 나간다.

import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import build_prompt_text, load, parse_prompt_harm

MODEL_IDS = {
    "polyguard": "ToxicityPrompts/PolyGuard-Qwen-Smol",
    "llamaguard": "meta-llama/Llama-Guard-3-1B",
}

MODELS = {}
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    for name, model_id in MODEL_IDS.items():
        tok, model = load(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        MODELS[name] = (tok, model)
    yield
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


class Req(BaseModel):
    prompt: str
    response: str = ""
    model: str = "polyguard"


@app.post("/moderate")
def moderate(req: Req):
    if req.model not in MODELS:
        raise HTTPException(status_code=400, detail=f"unknown model: {req.model} (choices: {list(MODEL_IDS)})")

    tok, model = MODELS[req.model]
    text = build_prompt_text(tok, req.model, req.prompt, req.response)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")

    # async def로 두면 GPU 추론이 이벤트 루프를 블로킹한다. 동기 def + Lock으로
    # FastAPI가 스레드풀로 빼고, GPU 접근만 직렬화한다.
    with lock:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, pad_token_id=tok.pad_token_id
            )
    raw = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    harmful = parse_prompt_harm(req.model, raw) == "harmful"

    return {"model": req.model, "prompt_harmful": harmful, "raw_output": raw}
