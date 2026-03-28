# Oxide-Agent TTS Modules

Dual TTS system for agent voice synthesis: English (Kokoro-82M) and Russian (Silero v5.4).

## Structure

```
oxide-agent-tts/
├── docker-compose.yml     # Unified compose for both services
├── Dockerfile.en           # EN TTS (Kokoro, port 8000)
├── Dockerfile.ru           # RU TTS (Silero, port 8010)
├── requirements_en.txt
├── requirements_ru.txt
├── KOKORO_TTS_API.md       # EN API reference
├── VOICES-KOKORO.md        # Available EN voices
└── src/tts/
    ├── en/                 # Kokoro TTS (EN)
    │   ├── src/main.py     # FastAPI app
    │   ├── scripts/        # Benchmark scripts
    │   └── tests/
    └── ru/                 # Silero TTS (RU)
        └── src/main.py     # FastAPI app
```

## Tech Stack

- **Languages**: Python 3.12 (both)
- **Frameworks**: FastAPI, uvicorn, asyncio
- **EN TTS**: ONNX Runtime, misaki (G2P), pydub
- **RU TTS**: PyTorch (CPU), Silero

## Services

| Service | Port | Model | Language |
|---------|------|-------|----------|
| tts-en  | 8000 | Kokoro-82M (ONNX) | English |
| tts-ru  | 8001 | Silero v5.4 | Russian |

## Commands

```bash
# Build and start both services
docker compose up -d

# Start specific service
docker compose up -d tts-en

# View logs
docker compose logs -f tts-en
docker compose logs -f tts-ru

# Stop
docker compose down
```

## Architecture

### EN TTS (Kokoro)
- ONNX Runtime inference (CPU, 6 threads)
- G2P via misaki library
- Streaming: chunks by sentence (.!?)
- Formats: OGG/Opus, MP3, WAV, PCM (24kHz)
- Voice mixing: `voice1+voice2:ratio`

### RU TTS (Silero)
- PyTorch CPU inference
- Fixed speaker: xenia
- Sample rate: 48000 Hz
- Endpoint: `/healthz`

## API Endpoints

### EN TTS
```
POST /v1/audio/speech          # File (ogg/mp3/wav)
POST /v1/audio/speech/stream    # Streaming PCM
```

### RU TTS
```
POST /v1/audio/speech          # File (webm/ogg)
GET  /healthz                  # Health check
```

## Development

Python code only in `src/tts/*/src/`. Docker files and requirements in root.

For EN tests: `src/tts/en/scripts/benchmark_tts.py`
