from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import os

from inference import predict_endpoint

app = FastAPI(title="SmartTurn API (ONNX)")

BASE_SILENCE_MS = int(os.getenv("SMARTTURN_BASE_SILENCE_MS", "800"))
MAX_SILENCE_MS  = int(os.getenv("SMARTTURN_MAX_SILENCE_MS", "4000"))
P_CONTINUE_TH   = float(os.getenv("SMARTTURN_P_CONTINUE", "0.5"))

class SmartTurnRequest(BaseModel):
    audio_b64: str
    format: str = "s16le"
    sample_rate: int = 16000
    channels: int = 1
    tail_ms: int = 8000
    silence_ms: int = BASE_SILENCE_MS

class SmartTurnResponse(BaseModel):
    p_continue: float
    p_end: float
    decision: str  # "continue" | "end"
    recommended_silence_ms: int

def decode_pcm_s16le(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    if len(raw) < 2:
        return np.zeros((0,), dtype=np.float32)

    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    audio_f32 = pcm_i16.astype(np.float32) / 32768.0
    return np.clip(audio_f32, -1.0, 1.0)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/smartturn/predict", response_model=SmartTurnResponse)
def smartturn_predict(req: SmartTurnRequest):
    if req.sample_rate != 16000:
        raise HTTPException(status_code=400, detail="Only sample_rate=16000 supported")
    if req.channels != 1:
        raise HTTPException(status_code=400, detail="Only channels=1 supported")
    if req.format.lower() != "s16le":
        raise HTTPException(status_code=400, detail="Only format=s16le supported")

    audio = decode_pcm_s16le(req.audio_b64)
    if audio.size == 0:
        return SmartTurnResponse(
            p_continue=0.0,
            p_end=1.0,
            decision="end",
            recommended_silence_ms=BASE_SILENCE_MS,
        )

    out = predict_endpoint(audio)

    # probability из smartturn = p_end (реплика завершена)
    p_end = float(out.get("probability", 0.0))
    p_end = max(0.0, min(1.0, p_end))
    p_continue = 1.0 - p_end

    decision = "continue" if p_continue >= P_CONTINUE_TH else "end"
    rec_ms = MAX_SILENCE_MS if decision == "continue" else BASE_SILENCE_MS

    return SmartTurnResponse(
        p_continue=p_continue,
        p_end=p_end,
        decision=decision,
        recommended_silence_ms=int(rec_ms),
    )
