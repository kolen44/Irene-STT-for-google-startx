import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from audio_utils import truncate_audio_to_last_n_seconds
import os
from pathlib import Path

# Путь к модели в контейнере (загружена при сборке)
MODEL_DIR = Path("/app/models")
ONNX_MODEL_PATH = MODEL_DIR / "smart-turn-v3.2-gpu.onnx"

# Fallback для локальной разработки
if not ONNX_MODEL_PATH.exists():
    ONNX_MODEL_PATH = Path(__file__).parent / "smart-turn-v3.2-gpu.onnx"

if not ONNX_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Модель не найдена: {ONNX_MODEL_PATH}\n"
        "В Docker модель загружается автоматически при сборке.\n"
        "Для локальной разработки запустите:\n"
        "  python -c \"from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id='pipecat-ai/smart-turn-v3', "
        "filename='smart-turn-v3.2-gpu.onnx', local_dir='.')\""
    )

def build_session(onnx_path):
    """Создает ONNX Runtime сессию с GPU поддержкой (fallback на CPU)."""
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Список провайдеров с приоритетом GPU
    # Если CUDA недоступна, автоматически fallback на CPU
    available_providers = ort.get_available_providers()

    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append(("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
            "cudnn_conv_algo_search": "DEFAULT",
        }))
        print("[SmartTurn] GPU (CUDA) доступна - будет использована для inference")

    # CPU провайдер всегда в fallback
    providers.append("CPUExecutionProvider")

    try:
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=providers,
        )
        return session
    except Exception as e:
        print(f"[SmartTurn] Ошибка создания сессии с GPU, fallback на CPU: {e}")
        # Fallback на CPU только
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

print(f"[SmartTurn] Загрузка модели: {ONNX_MODEL_PATH}")
print(f"[SmartTurn] Размер модели: {ONNX_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} МБ")
print(f"[SmartTurn] ONNX Runtime версия: {ort.__version__}")
print(f"[SmartTurn] Доступные провайдеры: {ort.get_available_providers()}")

feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)

active_providers = session.get_providers()
print(f"[SmartTurn] Активные провайдеры: {active_providers}")

# Проверяем используется ли GPU
if "CUDAExecutionProvider" in active_providers:
    print("[SmartTurn] ✓ GPU режим активен (CUDA)")
else:
    print("[SmartTurn] ⚠ CPU режим (GPU недоступна)")

print(f"[SmartTurn] ✓ Модель готова к использованию")

def predict_endpoint(audio_array):
    """
    Предсказывает завершенность высказывания.

    Args:
        audio_array: Numpy массив с аудио сэмплами (16kHz)

    Returns:
        Dictionary с результатами:
        - prediction: 1 (завершено), 0 (продолжается)
        - probability: Вероятность завершения (sigmoid output)
    """
    # Обрезка до 8 секунд или padding
    audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

    # Извлечение фич через Whisper feature extractor
    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=8 * 16000,
        truncation=True,
        do_normalize=True,
    )

    # Подготовка входа для ONNX
    input_features = inputs.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)

    # ONNX inference
    outputs = session.run(None, {"input_features": input_features})

    # Извлечение вероятности
    probability = outputs[0][0].item()

    # Бинарное предсказание
    prediction = 1 if probability > 0.5 else 0

    return {
        "prediction": prediction,
        "probability": probability,
    }

# Тестовый запуск при импорте
if __name__ == "__main__":
    # Dummy тест для проверки работоспособности
    dummy_audio = np.random.randn(16000).astype(np.float32)
    result = predict_endpoint(dummy_audio)
    print(f"\n[Test] Prediction: {result['prediction']}")
    print(f"[Test] Probability: {result['probability']:.4f}")
