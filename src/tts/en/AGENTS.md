# Project: Oxide-Agent TTS (Kokoro-82M)

Локально задеплоенный Text-to-Speech сервис на базе open-weight модели Kokoro-82M с Apache-лицензией. Модель содержит 82М параметров и обеспечивает высокое качество генерации речи при низкой вычислительной стоимости. Работает только с английским языком (RU отключен), поддерживает разные голоса и регулировку скорости.

**Tech Stack:**
- Language: Python 3.12
- Frameworks: FastAPI, asyncio
- Key Libs: ONNX Runtime, misaki (EN G2P), numpy, pydub

## Branch
The default branch is `main`.

## 🏗 Project Structure

/root/kokoro-tts/
├── models/          # Файлы моделей: ONNX модель (325MB), голоса, токенизатор
├── src/             # Основной код сервиса
│   ├── main.py      # FastAPI приложение с TTS API (EN only)
│   └── ru_module.py # RU модуль (отключен, сохранен для совместимости)
├── scripts/         # Скрипты тестирования
│   └── benchmark_tts.py
├── tests/           # Тестовые данные и результаты
└── .gitignore       # Python стандартный игнор (pycache, build)

### Key Modules
- **TTS API**: FastAPI сервис с эндпоинтами `/v1/audio/speech` (файл) и `/v1/audio/speech/stream` (стрим)
- **G2P Pipeline**: EN графемо-фонемное преобразование через misaki
- **ONNX Inference**: Инференс через ONNX Runtime с оптимизацией (CPU, 6 потоков)
- **Voice Mixing**: Поддержка смешивания голосов (формат: voice1+voice2:ratio)
- **Audio Formats**: PCM (стриминг), OGG/Opus, MP3, WAV

## 🛠 Architecture & Rules

### 1. Patterns
- Streaming API: Генерация аудио чанками по предложениям (разделение по .!?)
- Thread Pool: Один воркер для синхронного инференса в async окружении
- Audio format: PCM 16-bit, 24000 Hz, float32 → int16 conversion

### 2. Conventions
- **Language Support**: Только "en" (RU отключен, сохранен в ru_module.py)
- **Error Handling**: HTTPException с деталями, валидация скорости > 0
- **Phonemes**: EN графемо-фонемное преобразование через misaki G2P

## 🚀 Deployment

### Systemd Service
Сервис запускается через `oxide-tts.service`:

```bash
# Проверить статус
systemctl status oxide-tts.service

# Перезапуск
systemctl restart oxide-tts.service

# Логи
journalctl -u oxide-tts.service -f
```

## 🧪 Testing

### Benchmark Test
Для измерения производительности генерации OGG Opus (время генерации vs длительность аудио):

```bash
python3 scripts/benchmark_tts.py
```

Скрипт:
- Генерирует OGG Opus из длинного английского текста (~18.65s)
- Измеряет общее время (TTS инференс + кодирование)
- Вычисляет RTF (Real-Time Factor) через ffprobe
- Сохраняет результат в `tests/benchmark_output.ogg`
- Ожидаемый RTF: ~0.37-0.40 (генерация в ~2.5-2.8 раза быстрее реального времени)

**Примечание:** Тест включает кодирование в OGG. Чистый TTS инференс быстрее на ~20-30%.

### Manual Test (OGG для Telegram)
```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o voice.ogg \
  -d '{"text": "Hello world", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "ogg"}'
```

### Streaming Test (PCM)
```bash
curl -s http://127.0.0.1:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "pcm"}' > voice.pcm
```
