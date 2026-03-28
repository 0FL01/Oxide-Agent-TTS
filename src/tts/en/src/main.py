import asyncio
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from misaki import en
from pydantic import BaseModel, Field
from pydub import AudioSegment

app = FastAPI(title="Oxide-Agent TTS (Kokoro-82M)")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "kokoro-v1.0.onnx"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"

# 1. Настройка ONNX Runtime
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 6
sess_opts.inter_op_num_threads = 1
sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

print("Загрузка ONNX модели...")
session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_opts,
    providers=["CPUExecutionProvider"],
)

# Кэшируем имя выходного тензора для оптимизации
OUTPUT_NAME = session.get_outputs()[0].name

# 2. Загрузка ресурсов
print("Загрузка словарей и голосов...")

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Не найден tokenizer: {TOKENIZER_PATH}")

with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

try:
    vocab = tokenizer_data["model"]["vocab"]
except KeyError as e:
    raise RuntimeError(
        f"Некорректный tokenizer.json: нет ключа {e}. "
        f"Нужен именно models/tokenizer.json, а не config.json."
    ) from e

PAD_TOKEN_ID = vocab["$"]

if not VOICES_PATH.exists():
    raise FileNotFoundError(f"Не найден файл голосов: {VOICES_PATH}")

with np.load(str(VOICES_PATH)) as voices_np:
    voices_db = {name: voices_np[name].astype(np.float32) for name in voices_np.files}

# 3. Инициализация G2P (только EN)
print("Инициализация EN пайплайна (misaki)...")
en_g2p = en.G2P(trf=False, british=False, fallback=None)

# Пул потоков для инференса
inference_executor = ThreadPoolExecutor(max_workers=1)


def phonemes_to_tokens(phonemes: str) -> list[int]:
    return [vocab[ch] for ch in phonemes if ch in vocab]


def text_to_tokens_en(text: str) -> list[int]:
    """EN: Misaki G2P -> токены"""
    phonemes, _ = en_g2p(text)
    tokens = phonemes_to_tokens(phonemes)
    if not tokens:
        raise ValueError(f"Не удалось получить токены для EN текста: {text!r}")
    return tokens


def get_style_vector(voice_name: str) -> np.ndarray:
    """Извлечение или смешивание голоса"""
    if "+" in voice_name and ":" in voice_name:
        voices_part, ratio_str = voice_name.split(":")
        v1_name, v2_name = voices_part.split("+")
        weight = float(ratio_str)

        v1 = voices_db.get(v1_name)
        v2 = voices_db.get(v2_name)
        if v1 is None or v2 is None:
            raise ValueError("Один из голосов для смешивания не найден")

        min_len = min(v1.shape[0], v2.shape[0])
        v1_aligned = v1[:min_len, :]
        v2_aligned = v2[:min_len, :]
        v_mixed = (v1_aligned * weight) + (v2_aligned * (1.0 - weight))

        norm_v1 = np.linalg.norm(v1_aligned, axis=-1, keepdims=True)
        norm_v2 = np.linalg.norm(v2_aligned, axis=-1, keepdims=True)
        mean_norm = (norm_v1 * weight) + (norm_v2 * (1.0 - weight))
        v_mixed_norm = v_mixed / (np.linalg.norm(v_mixed, axis=-1, keepdims=True) + 1e-8)
        return (v_mixed_norm * mean_norm).astype(np.float32)

    if voice_name not in voices_db:
        raise ValueError(f"Голос {voice_name} не найден")

    return voices_db[voice_name]


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"   # только "en" (RU отключен, см. ru_module.py)
    voice: str = "af_bella"
    speed: float = 1.0
    format: str = Field(default="pcm", pattern="^(pcm|ogg|mp3|wav)$")


def pcm_to_ogg(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Convert PCM 16-bit audio to OGG Opus format."""
    # Create AudioSegment from PCM data
    audio = AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit
        frame_rate=sample_rate,
        channels=1  # mono
    )
    
    # Export to OGG/Opus in memory
    output = io.BytesIO()
    audio.export(output, format="ogg", codec="libopus")
    return output.getvalue()


def pcm_to_audio_format(pcm_bytes: bytes, format: str, sample_rate: int = 24000) -> bytes:
    """Convert PCM 16-bit audio to specified format."""
    audio = AudioSegment(
        data=pcm_bytes,
        sample_width=2,  # 16-bit
        frame_rate=sample_rate,
        channels=1  # mono
    )
    
    output = io.BytesIO()
    
    if format == "ogg":
        audio.export(output, format="ogg", codec="libopus")
    elif format == "mp3":
        audio.export(output, format="mp3")
    elif format == "wav":
        audio.export(output, format="wav")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output.getvalue()


DEBUG_AUDIO = False  # Отключено для production

# Регулярное выражение для разделения по предложениям
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_text_smart(text: str, max_chars: int = 180) -> list[str]:
    """Умное разбиение текста: по предложениям с объединением коротких фрагментов.
    
    Оптимальный компромисс между latency и качеством: меньше вызовов модели и
    меньше швов на стыках между чанками.
    """
    parts = [p.strip() for p in SENT_RE.split(text.strip()) if p.strip()]
    if not parts:
        return []

    merged = []
    buf = []
    buf_len = 0

    for part in parts:
        extra = len(part) if not buf else len(part) + 1
        if buf and buf_len + extra > max_chars:
            merged.append(" ".join(buf))
            buf = [part]
            buf_len = len(part)
        else:
            buf.append(part)
            buf_len += extra

    if buf:
        merged.append(" ".join(buf))

    return merged


def process_chunk_sync(text_chunk: str, lang: str, voice_name: str, speed: float) -> bytes:
    if not text_chunk.strip():
        return b""

    if lang.lower() != "en":
        raise ValueError("Поддерживается только lang='en'")

    if speed <= 0:
        raise ValueError("speed должен быть > 0")

    tokens = text_to_tokens_en(text_chunk)
    tokens = [PAD_TOKEN_ID, *tokens, PAD_TOKEN_ID]

    style_vector = get_style_vector(voice_name)
    style_idx = min(len(tokens), style_vector.shape[0] - 1)
    ref_s_tensor = np.asarray(style_vector[style_idx], dtype=np.float32).reshape(1, -1)

    audio_fp32 = session.run(
        [OUTPUT_NAME],
        {
            "tokens": np.asarray([tokens], dtype=np.int64),
            "style": ref_s_tensor,
            "speed": np.asarray([speed], dtype=np.float32),
        },
    )[0]

    audio_fp32 = np.asarray(audio_fp32).reshape(-1)
    audio_pcm = np.clip(audio_fp32 * 32767.0, -32768, 32767).astype(np.int16, copy=False)

    if DEBUG_AUDIO:
        print(f"audio_fp32: shape={audio_fp32.shape}, range=[{audio_fp32.min():.4f}, {audio_fp32.max():.4f}]")
        print(f"audio_pcm: shape={audio_pcm.shape}, range=[{audio_pcm.min()}, {audio_pcm.max()}]")

    return audio_pcm.tobytes()


@app.post("/v1/audio/speech")
async def generate_speech_file(req: TTSRequest):
    """Generate audio and return as file (OGG, MP3, or WAV)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой text")
    
    if req.format == "pcm":
        raise HTTPException(status_code=400, detail="Use /v1/audio/speech/stream for PCM format")
    
    chunks = split_text_smart(req.text.strip())
    
    # Accumulate all PCM chunks
    all_pcm = bytearray()
    loop = asyncio.get_running_loop()
    
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            pcm_bytes = await loop.run_in_executor(
                inference_executor,
                process_chunk_sync,
                chunk,
                req.lang,
                req.voice,
                req.speed,
            )
            if pcm_bytes:
                all_pcm.extend(pcm_bytes)
        except Exception as e:
            print(f"Ошибка генерации чанка {chunk!r}: {e}")
            raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    
    if not all_pcm:
        raise HTTPException(status_code=500, detail="No audio generated")
    
    # Convert to requested format
    try:
        audio_data = pcm_to_audio_format(bytes(all_pcm), req.format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {e}")
    
    # Set appropriate content type
    content_types = {
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "wav": "audio/wav"
    }
    
    return Response(
        content=audio_data,
        media_type=content_types[req.format],
        headers={
            "Content-Disposition": f"attachment; filename=speech.{req.format}"
        }
    )


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "tts-en",
        "model": "kokoro-v1.0",
        "voices_available": len(voices_db),
    }


@app.post("/v1/audio/speech/stream")
async def generate_speech_stream(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Пустой text")
    
    # For streaming, format should be pcm (or use /v1/audio/speech)
    if req.format != "pcm":
        raise HTTPException(status_code=400, detail="Streaming only supports PCM format. Use /v1/audio/speech for OGG/MP3/WAV")

    chunks = split_text_smart(req.text.strip())

    async def audio_generator():
        loop = asyncio.get_running_loop()
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                pcm_bytes = await loop.run_in_executor(
                    inference_executor,
                    process_chunk_sync,
                    chunk,
                    req.lang,
                    req.voice,
                    req.speed,
                )
                if pcm_bytes:
                    yield pcm_bytes
            except Exception as e:
                print(f"Ошибка генерации чанка {chunk!r}: {e}")
                raise

    return StreamingResponse(audio_generator(), media_type="audio/pcm")


if __name__ == "__main__":
    print("Сервер готов к работе.")
