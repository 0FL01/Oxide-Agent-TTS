import asyncio
import html
import inspect
import io
import os
import re
import shutil
import subprocess
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

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
APP_VERSION = os.getenv("APP_VERSION", "dev")

torch.set_num_threads(OMP_NUM_THREADS)
torch.set_grad_enabled(False)

device = torch.device("cpu")
model_lock = Lock()

SUPPORTED_BREAK_STRENGTHS = {"x-weak", "weak", "medium", "strong", "x-strong"}
SUPPORTED_PROSODY_RATES = {"x-slow", "slow", "medium", "fast", "x-fast"}
SUPPORTED_PROSODY_PITCHES = {"x-low", "low", "medium", "high", "x-high"}
SUPPORTED_SSML_TAGS = {"speak", "break", "prosody", "p", "s"}

NON_BAYA_SAFE_PITCHES = {"low", "high"}


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
    format: Literal["wav", "ogg"] = "wav"
    ssml: bool = False

    strict_ssml: bool = False

    humanize: bool = True
    put_accent: Optional[bool] = None
    put_yo: Optional[bool] = None
    put_stress_homo: Optional[bool] = None
    put_yo_homo: Optional[bool] = None


def looks_like_ssml(text: str) -> bool:
    stripped = text.lstrip().lower()
    markers = (
        "<speak",
        "<break",
        "<prosody",
        "<p>",
        "<s>",
    )
    return any(marker in stripped for marker in markers)


def get_available_voices() -> list[str]:
    if hasattr(model, "speakers") and model.speakers:
        return sorted(set(map(str, model.speakers)))
    if hasattr(model, "speaker_to_id") and model.speaker_to_id:
        return sorted(set(map(str, model.speaker_to_id.keys())))
    return []


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
        wf.setsampwidth(2)
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


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def ssml_to_plain_text(text: str) -> str:
    value = html.unescape(text)
    value = re.sub(r"<break\b[^>]*\/?>", ", ", value, flags=re.IGNORECASE)
    value = re.sub(r"</?p\b[^>]*>", ". ", value, flags=re.IGNORECASE)
    value = re.sub(r"</?s\b[^>]*>", ". ", value, flags=re.IGNORECASE)
    value = re.sub(r"</?(speak|prosody)\b[^>]*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def ensure_no_ascii_digits(spoken_text: str) -> None:
    if re.search(r"[0-9]", spoken_text):
        raise ValueError(
            "Text contains Arabic numerals (0-9). "
            "Для RU Silero числа нужно писать словами."
        )


def normalize_break_strength_from_time(raw_time: Optional[str]) -> Optional[str]:
    if not raw_time:
        return None

    raw = raw_time.strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|s)", raw)
    if not match:
        return None

    amount = float(match.group(1))
    unit = match.group(2)
    ms = amount * 1000.0 if unit == "s" else amount

    if ms < 150:
        return "x-weak"
    if ms < 350:
        return "weak"
    if ms < 900:
        return "medium"
    if ms < 1800:
        return "strong"
    return "x-strong"


def normalize_rate(raw_rate: Optional[str]) -> Optional[str]:
    if not raw_rate:
        return None

    value = raw_rate.strip().lower()
    if value in SUPPORTED_PROSODY_RATES:
        return value

    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)", value)
    if match:
        num = float(match.group(1))
        if num <= 0.65:
            return "x-slow"
        if num < 0.90:
            return "slow"
        if num <= 1.15:
            return "medium"
        if num < 1.45:
            return "fast"
        return "x-fast"

    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", value)
    if match:
        pct = float(match.group(1))
        if pct <= -30:
            return "x-slow"
        if pct < -10:
            return "slow"
        if pct < 10:
            return "medium"
        if pct < 35:
            return "fast"
        return "x-fast"

    return "medium"


def normalize_pitch(raw_pitch: Optional[str], speaker: str) -> Optional[str]:
    if not raw_pitch:
        return None

    value = raw_pitch.strip().lower()

    if speaker == "baya":
        allowed = SUPPORTED_PROSODY_PITCHES
    else:
        allowed = NON_BAYA_SAFE_PITCHES

    if value in SUPPORTED_PROSODY_PITCHES:
        if value in allowed:
            return value
        if value in {"x-low", "low"} and "low" in allowed:
            return "low"
        if value in {"x-high", "high"} and "high" in allowed:
            return "high"
        return None

    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)", value)
    if match:
        num = float(match.group(1))
        if num < 0.95 and "low" in allowed:
            return "low"
        if num > 1.05 and "high" in allowed:
            return "high"
        return None

    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", value)
    if match:
        pct = float(match.group(1))
        if pct <= -10 and "low" in allowed:
            return "low"
        if pct >= 10 and "high" in allowed:
            return "high"
        return None

    return None


def sanitize_ssml(text: str, speaker: str) -> Optional[str]:
    raw = html.unescape(text).strip()
    if not raw:
        return None

    if not raw.lstrip().lower().startswith("<speak"):
        raw = f"<speak>{raw}</speak>"

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None

    def render_text(value: Optional[str]) -> str:
        if not value:
            return ""
        return html.escape(value, quote=False)

    def render_node(node: ET.Element) -> str:
        tag = strip_ns(node.tag)

        if tag not in SUPPORTED_SSML_TAGS:
            chunks: list[str] = []
            chunks.append(render_text(node.text))
            for child in node:
                chunks.append(render_node(child))
                chunks.append(render_text(child.tail))
            return "".join(chunks)

        if tag == "break":
            strength = node.attrib.get("strength", "").strip().lower()
            time_value = node.attrib.get("time")
            normalized_from_time = normalize_break_strength_from_time(time_value)

            if normalized_from_time:
                strength = normalized_from_time

            if strength not in SUPPORTED_BREAK_STRENGTHS:
                strength = "medium"

            return f'<break strength="{strength}"/>'

        chunks: list[str] = []
        chunks.append(render_text(node.text))

        for child in node:
            chunks.append(render_node(child))
            chunks.append(render_text(child.tail))

        inner = "".join(chunks)

        if tag == "prosody":
            attrs: list[str] = []

            normalized_rate = normalize_rate(node.attrib.get("rate"))
            normalized_pitch = normalize_pitch(node.attrib.get("pitch"), speaker)

            if normalized_rate and normalized_rate != "medium":
                attrs.append(f'rate="{normalized_rate}"')

            if normalized_pitch:
                attrs.append(f'pitch="{normalized_pitch}"')

            attr_text = f" {' '.join(attrs)}" if attrs else ""
            return f"<prosody{attr_text}>{inner}</prosody>"

        return f"<{tag}>{inner}</{tag}>"

    rendered = render_node(root).strip()
    if not rendered.lower().startswith("<speak"):
        rendered = f"<speak>{rendered}</speak>"

    return rendered


def build_humanize_flags(req: TTSRequest) -> dict:
    sig = inspect.signature(model.apply_tts)

    if req.humanize:
        desired_flags = {
            "put_accent": True if req.put_accent is None else req.put_accent,
            "put_yo": True if req.put_yo is None else req.put_yo,
            "put_stress_homo": True if req.put_stress_homo is None else req.put_stress_homo,
            "put_yo_homo": True if req.put_yo_homo is None else req.put_yo_homo,
        }
    else:
        desired_flags = {
            "put_accent": req.put_accent,
            "put_yo": req.put_yo,
            "put_stress_homo": req.put_stress_homo,
            "put_yo_homo": req.put_yo_homo,
        }

    kwargs = {}
    for name, value in desired_flags.items():
        if value is not None and name in sig.parameters:
            kwargs[name] = value

    return kwargs


def prepare_tts_kwargs(req: TTSRequest) -> tuple[dict, bool]:
    text = req.text.strip()
    use_ssml = req.ssml or looks_like_ssml(text)

    kwargs = {
        "speaker": req.speaker,
        "sample_rate": req.sample_rate,
        **build_humanize_flags(req),
    }

    if use_ssml:
        sanitized = sanitize_ssml(text, req.speaker)

        if sanitized is None:
            if req.strict_ssml:
                raise ValueError("Invalid or unsupported SSML")
            plain = ssml_to_plain_text(text)
            ensure_no_ascii_digits(plain)
            kwargs["text"] = plain
            return kwargs, False

        spoken_text = ssml_to_plain_text(sanitized)
        ensure_no_ascii_digits(spoken_text)

        kwargs["ssml_text"] = sanitized
        return kwargs, True

    plain_text = ssml_to_plain_text(text) if looks_like_ssml(text) else text
    ensure_no_ascii_digits(plain_text)
    kwargs["text"] = plain_text
    return kwargs, False


def synthesize(req: TTSRequest) -> tuple[bytes, str, str, dict]:
    kwargs, used_ssml = prepare_tts_kwargs(req)
    response_headers: dict[str, str] = {}

    try:
        with model_lock:
            audio = model.apply_tts(**kwargs)
    except Exception as first_error:
        if used_ssml and not req.strict_ssml:
            plain_text = ssml_to_plain_text(req.text)
            ensure_no_ascii_digits(plain_text)

            fallback_req = req.model_copy(update={
                "text": plain_text,
                "ssml": False,
            })
            fallback_kwargs, _ = prepare_tts_kwargs(fallback_req)

            with model_lock:
                audio = model.apply_tts(**fallback_kwargs)

            response_headers["X-TTS-Degraded"] = "ssml-to-plain-text"
            response_headers["X-TTS-Original-Error"] = str(first_error)[:200]
        else:
            raise

    pcm16_bytes = tensor_to_pcm16_bytes(audio)

    if req.format == "wav":
        return (
            pcm16_to_wav_bytes(pcm16_bytes, req.sample_rate),
            "audio/wav",
            "speech.wav",
            response_headers,
        )

    if req.format == "ogg":
        return (
            pcm16_to_ogg_bytes(pcm16_bytes, req.sample_rate),
            "audio/ogg",
            "speech.ogg",
            response_headers,
        )

    raise RuntimeError(f"Unsupported format: {req.format}")


@app.get("/v1/audio/voices")
def list_voices():
    return {
        "app_version": APP_VERSION,
        "default_speaker": DEFAULT_SPEAKER,
        "voices": get_available_voices(),
    }


@app.get("/healthz")
def healthz():
    sig = inspect.signature(model.apply_tts)

    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "device": "cpu",
        "model_path": str(MODEL_PATH),
        "default_speaker": DEFAULT_SPEAKER,
        "default_sample_rate": DEFAULT_SAMPLE_RATE,
        "supported_formats": ["wav", "ogg"],
        "supports_ssml": "ssml_text" in sig.parameters,
        "ffmpeg_available": shutil.which(FFMPEG_BIN) is not None,
        "ogg_opus_bitrate": OGG_OPUS_BITRATE,
        "ogg_opus_application": OGG_OPUS_APPLICATION,
        "humanize_defaults": {
            "put_accent": "put_accent" in sig.parameters,
            "put_yo": "put_yo" in sig.parameters,
            "put_stress_homo": "put_stress_homo" in sig.parameters,
            "put_yo_homo": "put_yo_homo" in sig.parameters,
        },
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
            detail="Допустимые sample_rate: 8000, 24000, 48000",
        )

    voices = get_available_voices()
    if voices and req.speaker not in voices:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported speaker: {req.speaker}. Use GET /v1/audio/voices",
        )

    try:
        audio_bytes, media_type, filename, extra_headers = await asyncio.to_thread(synthesize, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation error: {e}") from e

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-App-Version": APP_VERSION,
        **extra_headers,
    }

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers=headers,
    )
