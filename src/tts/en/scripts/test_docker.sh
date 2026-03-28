#!/bin/bash
# Скрипт для быстрого тестирования Docker контейнера

set -e

echo "=== Kokoro TTS Docker Test ==="
echo

# Проверяем, запущен ли контейнер
if ! docker ps | grep -q kokoro-tts; then
    echo "❌ Контейнер kokoro-tts не запущен"
    echo "Запустите: docker-compose up -d"
    exit 1
fi

echo "✅ Контейнер запущен"
echo

# Ждем, пока сервис будет готов
echo "⏳ Ожидание готовности API..."
for i in {1..30}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ API готов"
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo "❌ API не отвечает"
        exit 1
    fi
done
echo

# Тест 1: Генерация OGG
echo "🎵 Тест 1: Генерация OGG..."
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o /tmp/test_voice.ogg \
  -d '{"text": "Hello from Docker container", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "ogg"}'

if [ -f /tmp/test_voice.ogg ] && [ -s /tmp/test_voice.ogg ]; then
    echo "✅ OGG файл создан: $(ls -lh /tmp/test_voice.ogg | awk '{print $5}')"
    file /tmp/test_voice.ogg
else
    echo "❌ Ошибка генерации OGG"
fi
echo

# Тест 2: Генерация MP3
echo "🎵 Тест 2: Генерация MP3..."
curl -s http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -o /tmp/test_voice.mp3 \
  -d '{"text": "Testing MP3 format", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "mp3"}'

if [ -f /tmp/test_voice.mp3 ] && [ -s /tmp/test_voice.mp3 ]; then
    echo "✅ MP3 файл создан: $(ls -lh /tmp/test_voice.mp3 | awk '{print $5}')"
    file /tmp/test_voice.mp3
else
    echo "❌ Ошибка генерации MP3"
fi
echo

# Тест 3: Стриминг PCM
echo "🎵 Тест 3: Стриминг PCM..."
curl -s http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Streaming test", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "pcm"}' > /tmp/test_voice.pcm

if [ -f /tmp/test_voice.pcm ] && [ -s /tmp/test_voice.pcm ]; then
    echo "✅ PCM файл создан: $(ls -lh /tmp/test_voice.pcm | awk '{print $5}')"
else
    echo "❌ Ошибка стриминга PCM"
fi
echo

# Тест 4: Проверка ошибок
echo "🧪 Тест 4: Обработка ошибок..."
response=$(curl -s -w "%{http_code}" http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "", "lang": "en", "voice": "af_bella", "speed": 1.0, "format": "ogg"}')

if echo "$response" | grep -q "400$"; then
    echo "✅ Правильная обработка пустого текста (400 Bad Request)"
else
    echo "⚠️  Неожиданный ответ на пустой текст: $response"
fi
echo

# Очистка
echo "🧹 Очистка тестовых файлов..."
rm -f /tmp/test_voice.*

echo
echo "=== Тестирование завершено ==="
