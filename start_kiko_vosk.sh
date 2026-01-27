#!/bin/bash
# ==============================================
# KIKO Voice Assistant - Vosk STT Launcher
# ==============================================
# Запуск распознавания речи через Vosk
# и отправка в KIKO AI
# ==============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============ НАСТРОЙКИ ============
# RTSP URL камеры (измени на свой)
RTSP_URL="${RTSP_URL:-rtsp://admin:totem1988@10.42.0.102:554/h264Preview_01_main}"

# URL KIKO сервера (docker-compose с network_mode: host, PORT: 3001)
KIKO_URL="${KIKO_URL:-http://127.0.0.1:3001/ai}"

# Путь к модели Vosk
MODEL_PATH="${MODEL_PATH:-model}"

# ============ ПРОВЕРКИ ============
echo "=========================================="
echo "  KIKO Voice Assistant - Vosk STT"
echo "=========================================="

# Проверяем Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден!"
    exit 1
fi

# Проверяем ffmpeg (нужен для RTSP)
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg не найден! Установи: sudo apt install ffmpeg"
    exit 1
fi

# Проверяем модель
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Модель Vosk не найдена в '$MODEL_PATH'"
    echo ""
    echo "Скачай модель с https://alphacephei.com/vosk/models"
    echo "Рекомендуем: vosk-model-ru-0.42 (1.8GB, лучшее качество)"
    echo ""
    echo "Команды:"
    echo "  wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
    echo "  unzip vosk-model-ru-0.42.zip"
    echo "  mv vosk-model-ru-0.42 model"
    exit 1
fi

# Проверяем/создаём venv
if [ ! -d "venv" ]; then
    echo "📦 Создаём виртуальное окружение..."
    python3 -m venv venv
fi

# Активируем venv
source venv/bin/activate

# Проверяем зависимости
if ! python3 -c "import vosk" 2>/dev/null; then
    echo "📦 Устанавливаем зависимости..."
    pip install --upgrade pip
    pip install vosk sounddevice requests
fi

# ============ ЗАПУСК ============
echo ""
echo "🎤 Запуск распознавания..."
echo "   RTSP: ${RTSP_URL##*@}"
echo "   KIKO: $KIKO_URL"
echo "   Model: $MODEL_PATH"
echo ""
echo "Скажи 'Оптимус' + команду"
echo "Ctrl+C для выхода"
echo "=========================================="
echo ""

# Экспортируем переменные
export KIKO_URL

# Запускаем
python3 runva_vosk.py \
    --rtsp "$RTSP_URL" \
    --kiko-url "$KIKO_URL" \
    --model "$MODEL_PATH" \
    --samplerate 16000
