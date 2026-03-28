# Oxide-Agent TTS

Dual TTS modules: English (Kokoro-82M) and Russian (Silero v5.4).

## Quick Start

```bash
docker compose up -d
```

- EN TTS: http://127.0.0.1:8000
- RU TTS: http://127.0.0.1:8001

## EN TTS Usage

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o voice.ogg \
  -d '{
    "text": "Hello world",
    "lang": "en",
    "voice": "af_bella",
    "speed": 1.0,
    "format": "ogg"
  }'
```

Parameters: `text`, `voice` (see VOICES-KOKORO.md), `speed`, `format` (ogg/mp3/wav).

## RU TTS Usage

```bash
curl -s http://127.0.0.1:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет мир"}'
```

## Structure

```
src/tts/en/  # Kokoro TTS (EN)
src/tts/ru/  # Silero TTS (RU)
```

Python code only in `src/tts/*/src/`.
