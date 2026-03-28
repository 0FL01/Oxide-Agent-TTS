import asyncio
import io
import os
import wave
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

app = FastAPI(title="Oxide-Agent TTS (Silero CPU)")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FILE = os.getenv("SILERO_MODEL_FILE", "v5_ru.pt")
MODEL_PATH = MODELS_DIR / MODEL_FILE

DEFAULT_SPEAKER = os.getenv("SILERO_SPEAKER", "xenia")
DEFAULT_SAMPLE_RATE = int(os.getenv("SILERO_SAMPLE_RATE", "48000"))
OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))

torch.set_num_threads(OMP_NUM_THREADS)
torch.set_grad_enabled(False)

device = torch.device("cpu")
model_lock = Lock()


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл модели: {MODEL_PATH}. "
        f"Положи модель в ./models/{MODEL_FILE}"
    )

print(f"Loading Silero model from {MODEL_PATH} ...")
model = torch.package.PackageImporter(str(MODEL_PATH)).load_pickle("tts_models", "model")
model.to(device)


class TTSRequest(BaseModel):
    text: str
    speaker: str = DEFAULT_SPEAKER
    sample_rate: int = DEFAULT_SAMPLE_RATE
    format: str = Field(default="wav", pattern="^(wav)$")
    ssml: bool = False


def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
    audio = audio_tensor.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return buf.getvalue()


def synthesize(req: TTSRequest) -> bytes:
    kwargs = {
        "speaker": req.speaker,
        "sample_rate": req.sample_rate,
    }

    if req.ssml:
        kwargs["ssml_text"] = req.text
    else:
        kwargs["text"] = req.text

    with model_lock:
        audio = model.apply_tts(**kwargs)

    return tensor_to_wav_bytes(audio, req.sample_rate)


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": "cpu",
        "model_path": str(MODEL_PATH),
        "default_speaker": DEFAULT_SPEAKER,
        "default_sample_rate": DEFAULT_SAMPLE_RATE,
    }


@app.post("/v1/audio/speech")
async def generate_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой text")

    if req.sample_rate not in (8000, 24000, 48000):
        raise HTTPException(
            status_code=400,
            detail="Допустимые sample_rate: 8000, 24000, 48000"
        )

    try:
        wav_bytes = await asyncio.to_thread(synthesize, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {e}") from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
    )
