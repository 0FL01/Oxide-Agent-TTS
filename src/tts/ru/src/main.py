import asyncio
import io
import os
import shutil
import subprocess
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

DEFAULT_SPEAKER = os.getenv("SILERO_SPEAKER", "baya")
DEFAULT_SAMPLE_RATE = int(os.getenv("SILERO_SAMPLE_RATE", "48000"))
OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", "4"))

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
OGG_OPUS_BITRATE = os.getenv("OGG_OPUS_BITRATE", "32k")
OGG_OPUS_APPLICATION = os.getenv("OGG_OPUS_APPLICATION", "voip")

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
    format: str = Field(default="wav", pattern="^(wav|ogg)$")
    ssml: bool = False


def looks_like_ssml(text: str) -> bool:
    stripped = text.lstrip()
    ssml_markers = (
        "<speak",
        "<break",
        "<prosody",
        "<p>",
        "<s>",
        "<emphasis",
        "<say-as",
        "<sub",
        "<phoneme",
    )
    return any(marker in stripped for marker in ssml_markers)


def tensor_to_pcm16_bytes(audio_tensor: torch.Tensor) -> bytes:
    audio = audio_tensor.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def pcm16_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


def pcm16_to_ogg_bytes(pcm16_bytes: bytes, sample_rate: int) -> bytes:
    ffmpeg_path = shutil.which(FFMPEG_BIN)
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg не найден в PATH, ogg/opus недоступен")

    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-b:a", OGG_OPUS_BITRATE,
        "-vbr", "on",
        "-compression_level", "10",
        "-application", OGG_OPUS_APPLICATION,
        "-frame_duration", "20",
        "-f", "ogg",
        "pipe:1",
    ]

    completed = subprocess.run(
        cmd,
        input=pcm16_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg opus encode failed: {stderr or 'unknown error'}")

    if not completed.stdout:
        raise RuntimeError("ffmpeg вернул пустой ogg output")

    return completed.stdout


def prepare_tts_kwargs(req: TTSRequest) -> dict:
    text = req.text.strip()
    use_ssml = req.ssml or looks_like_ssml(text)

    kwargs = {
        "speaker": req.speaker,
        "sample_rate": req.sample_rate,
    }

    if use_ssml:
        if not text.lstrip().startswith("<speak"):
            text = f"<speak>{text}</speak>"
        kwargs["ssml_text"] = text
    else:
        kwargs["text"] = text

    return kwargs


def synthesize(req: TTSRequest) -> tuple[bytes, str, str]:
    kwargs = prepare_tts_kwargs(req)

    with model_lock:
        audio = model.apply_tts(**kwargs)

    pcm16_bytes = tensor_to_pcm16_bytes(audio)

    if req.format == "wav":
        return (
            pcm16_to_wav_bytes(pcm16_bytes, req.sample_rate),
            "audio/wav",
            "speech.wav",
        )

    if req.format == "ogg":
        return (
            pcm16_to_ogg_bytes(pcm16_bytes, req.sample_rate),
            "audio/ogg",
            "speech.ogg",
        )

    raise RuntimeError(f"Unsupported format: {req.format}")


def get_available_voices() -> list[str]:
    if hasattr(model, "speakers") and model.speakers:
        return sorted(set(map(str, model.speakers)))
    if hasattr(model, "speaker_to_id") and model.speaker_to_id:
        return sorted(set(map(str, model.speaker_to_id.keys())))
    return []


@app.get("/v1/audio/voices")
def list_voices():
    voices = get_available_voices()

    return {
        "default_speaker": DEFAULT_SPEAKER,
        "voices": voices,
    }


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": "cpu",
        "model_path": str(MODEL_PATH),
        "default_speaker": DEFAULT_SPEAKER,
        "default_sample_rate": DEFAULT_SAMPLE_RATE,
        "supported_formats": ["wav", "ogg"],
        "supports_ssml": True,
        "ffmpeg_available": shutil.which(FFMPEG_BIN) is not None,
        "ogg_opus_bitrate": OGG_OPUS_BITRATE,
        "ogg_opus_application": OGG_OPUS_APPLICATION,
    }


@app.post(
    "/v1/audio/speech",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/ogg": {},
            },
            "description": "Generated speech audio",
        }
    },
)
async def generate_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой text")

    if req.sample_rate not in (8000, 24000, 48000):
        raise HTTPException(
            status_code=400,
            detail="Допустимые sample_rate: 8000, 24000, 48000"
        )

    voices = get_available_voices()
    if voices and req.speaker not in voices:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported speaker: {req.speaker}. Use GET /v1/audio/voices"
        )

    try:
        audio_bytes, media_type, filename = await asyncio.to_thread(synthesize, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {e}") from e

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
