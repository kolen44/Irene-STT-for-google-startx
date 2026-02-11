FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Установка Python 3.11 и системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    libgomp1 \
    git \
  && ln -sf /usr/bin/python3.11 /usr/bin/python \
  && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
  && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# КЛЮЧЕВОЙ ШАГ: Загрузка модели из HuggingFace при сборке образа
# Это обеспечивает быстрый старт и работу offline
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download( \
        repo_id='pipecat-ai/smart-turn-v3', \
        filename='smart-turn-v3.2-gpu.onnx', \
        local_dir='/app/models', \
        local_dir_use_symlinks=False \
    )" && \
    echo "Модель загружена: $(ls -lh /app/models/*.onnx)"

# Копируем код приложения ПОСЛЕ загрузки модели
# (чтобы изменения в коде не инвалидировали кэш слоя с моделью)
COPY audio_utils.py /app/
COPY inference.py /app/
COPY main.py /app/

EXPOSE 8010

# Health check для проверки готовности
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
