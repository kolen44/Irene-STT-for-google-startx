import numpy as np

def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    """
    Обрезает аудио до последних N секунд или добавляет padding.
    
    Args:
        audio_array: Numpy массив с аудио
        n_seconds: Длительность в секундах
        sample_rate: Частота дискретизации
    
    Returns:
        Numpy массив длиной ровно n_seconds * sample_rate
    """
    max_samples = n_seconds * sample_rate
    
    if len(audio_array) > max_samples:
        # Берем последние N секунд
        return audio_array[-max_samples:]
    elif len(audio_array) < max_samples:
        # Padding нулями в начале
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0), mode='constant', constant_values=0)
    
    return audio_array
