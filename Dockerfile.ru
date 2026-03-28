FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    OMP_WAIT_POLICY=PASSIVE \
    SILERO_MODEL_FILE=v5_4_ru.pt \
    SILERO_SPEAKER=xenia \
    SILERO_SAMPLE_RATE=48000 \
    FFMPEG_BIN=ffmpeg \
    OGG_OPUS_BITRATE=32k \
    OGG_OPUS_APPLICATION=voip

ARG TORCH_VERSION=2.7.1+cpu
ARG MODEL_FILE=v5_4_ru.pt
ARG MODEL_URL=https://models.silero.ai/models/tts/ru/v5_4_ru.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_ru.txt /app/requirements_ru.txt

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==${TORCH_VERSION} && \
    pip install -r /app/requirements_ru.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/tts/ru/src /app/src

RUN mkdir -p /app/models && \
    curl -fL "${MODEL_URL}" -o "/app/models/${MODEL_FILE}"

RUN useradd -r -u 10001 -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8010

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
