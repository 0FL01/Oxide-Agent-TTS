# Oxide-Agent TTS (Kokoro-82M)

Локально задеплоенный Text-to-Speech сервис на базе open-weight модели Kokoro-82M с Apache-лицензией.

## Prerequisites

**Important:** Requires Python 3.12 or 3.11 (Debian 13 has Python 3.13, but `misaki` 0.9.4 requires `Python <3.13`).

### System Packages

```bash
apt update
apt install -y espeak-ng curl ffmpeg
```

Required:
- `espeak-ng` - required for Russian phonemization via `phonemizer`
- `curl`, `ffmpeg` - optional, for testing only

### Python Dependencies

Inside Python 3.12/3.11 venv:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install fastapi uvicorn onnxruntime numpy pydantic pydub "misaki[en]" ruaccent phonemizer
```

**Required:**
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `onnxruntime` - CPU inference
- `numpy` - array operations
- `pydantic` - data validation
- `misaki[en]` - English G2P
- `ruaccent` - Russian accent detection
- `phonemizer` - Russian phonemization via espeak-ng

**Not used in current code:**
- `espeakng`, `soundfile`, `phonemizer-fork` - not imported

### Required Model Files

Download model files to `models/` directory:

```bash
# Create models directory
mkdir -p models

# Download ONNX model (FP32, optimal for quality)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -P models

# Download voice database
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -P models

# Download config
wget https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json -P models

# Download tokenizer (via huggingface_hub)
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="onnx-community/Kokoro-82M-v1.0-ONNX",
    filename="tokenizer.json",
    local_dir="models",
    local_dir_use_symlinks=False,
)
print("tokenizer.json downloaded to models/")
PY
```

Required files:
- `models/config.json` - model configuration
- `models/tokenizer.json` - tokenizer vocab
- `models/kokoro-v1.0.onnx` (310MB) - ONNX model
- `models/voices-v1.0.bin` - voice style vectors

## Systemd Service Example

```ini
[Unit]
Description=Kokoro TTS API
After=network.target

[Service]
User=root
WorkingDirectory=/root/kokoro-tts

Environment="OMP_NUM_THREADS=6"
Environment="OMP_WAIT_POLICY=PASSIVE"
Environment="PYTHONUNBUFFERED=1"

MemoryMax=8G
MemorySwapMax=0

ExecStart=/opt/kokoro-venv/bin/python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1

Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## API Documentation

See [KOKORO_TTS_API.md](./KOKORO_TTS_API.md) for full API reference and usage examples.

## Available Voices

See [VOICES.md](./VOICES.md) for information about available American English voices.
