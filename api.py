# PolyGuard 단일 모델 FastAPI 서빙 — 엔드포인트 1개짜리 최소 버전.
# text -> risk -> category -> confidence -> reason 을 하나의 응답 스키마로 합쳐서 반환한다.
# 기존 app.py(멀티모델 서빙)와 별개로, 발표/데모용 단일 엔드포인트 예시.
#
# 실행: conda activate guard && uvicorn api:app --workers 1

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from models import load, moderate

MODEL_ID = "ToxicityPrompts/PolyGuard-Qwen-Smol"

# POLYGUARD_SYSTEM_PROMPT(models.py)의 S1-S14 정의를 그대로 축약한 표시용 이름.
CATEGORY_NAMES = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S3": "Sex Crimes",
    "S4": "Child Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tok"], state["model"] = load(MODEL_ID)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class ModerateRequest(BaseModel):
    text: str


class ModerateResponse(BaseModel):
    text: str
    risk_category_confidence: str
    reason: str


@app.post("/moderate", response_model=ModerateResponse)
def moderate_text(req: ModerateRequest):
    result = moderate("polyguard", state["tok"], state["model"], req.text)
    pct = round(result["confidence"] * 100) if result["confidence"] is not None else 0

    if result["risk"] == "unsafe" and result["category"]:
        label = CATEGORY_NAMES.get(result["category"][0], result["category"][0])
    else:
        label = "Safe"

    return ModerateResponse(
        text=req.text,
        risk_category_confidence=f"{label} (Confidence: {pct}%)",
        reason=result["reason"],
    )
