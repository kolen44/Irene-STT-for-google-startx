"""
runva_kiko.py - Режим работы Irene Voice Assistant для KIKO

Этот скрипт:
1. Слушает микрофон через VOSK (локально)
2. Распознаёт речь
3. Отправляет распознанный текст в KIKO через WebSocket

Запуск:
    python runva_kiko.py

Настройки в options/kiko.json:
    - kiko_ws_url: WebSocket URL KIKO сервера
    - send_partials: отправлять ли промежуточные результаты

Микрофон берётся из KIKO .env файла:
    - MIC_DEVICE (Windows)
    - PULSE_SOURCE (Linux)
"""

import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import logging
import json
import asyncio
import threading
import websockets
from typing import Optional
import time
from pathlib import Path
import subprocess

from jaa import load_options

# Глобальные переменные
mic_blocked = False
ws_connection: Optional[websockets.WebSocketClientProtocol] = None
ws_connected = False
text_queue = queue.Queue()

# Настройки по умолчанию
default_options = {
    "kiko_ws_url": "ws://localhost:3000/irene-asr",  # WebSocket URL KIKO
    "kiko_http_url": "http://localhost:3000",        # HTTP URL KIKO (fallback)
    "kiko_env_path": "../KIKO/server/.env",          # Путь к .env KIKO для чтения настроек микрофона
    "send_partials": False,                           # Отправлять промежуточные результаты
    "reconnect_delay": 3,                             # Задержка переподключения (сек)
    "model_path": "model",                            # Путь к модели VOSK
    "sample_rate": None,                              # Sample rate (None = авто)
    "device": None,                                   # Аудио устройство (None = из KIKO .env)
    "rtsp_url": None,                                 # RTSP URL для получения аудио с камеры
    "audio_source": "auto",                           # "auto", "rtsp", "mic"
}

# Загружаем опции
options = load_options(py_file=__file__, default_options=default_options)


def load_kiko_env() -> dict:
    """Загружает переменные из KIKO .env файла"""
    env_vars = {}
    
    # Пробуем разные пути к .env KIKO
    possible_paths = [
        options.get("kiko_env_path", "../KIKO/server/.env"),
        "../KIKO/server/.env",
        "../../KIKO/server/.env",
        os.path.join(os.path.dirname(__file__), "..", "KIKO", "server", ".env"),
    ]
    
    env_path = None
    for p in possible_paths:
        full_path = os.path.abspath(p)
        if os.path.exists(full_path):
            env_path = full_path
            break
    
    if not env_path:
        print("[KIKO ENV] Файл .env KIKO не найден, используются настройки по умолчанию")
        return env_vars
    
    print(f"[KIKO ENV] Загрузка настроек из: {env_path}")
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Убираем кавычки если есть
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    env_vars[key] = value
    except Exception as e:
        print(f"[KIKO ENV] Ошибка чтения .env: {e}")
    
    return env_vars


def get_microphone_device(kiko_env: dict) -> Optional[str]:
    """Получает устройство микрофона из настроек KIKO"""
    
    # Windows: MIC_DEVICE
    if sys.platform == 'win32':
        mic_device = kiko_env.get('MIC_DEVICE')
        if mic_device:
            print(f"[Микрофон] Из KIKO: MIC_DEVICE = {mic_device}")
            return mic_device
    
    # Linux: PULSE_SOURCE
    else:
        pulse_source = kiko_env.get('PULSE_SOURCE')
        if pulse_source:
            print(f"[Микрофон] Из KIKO: PULSE_SOURCE = {pulse_source}")
            return pulse_source
    
    return None


def find_device_by_name(device_name: str) -> Optional[int]:
    """Находит устройство по имени и возвращает его индекс"""
    if not device_name:
        return None
    
    devices = sd.query_devices()
    device_name_lower = device_name.lower()
    
    # Сначала ищем точное совпадение
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:  # Только input устройства
            if dev['name'].lower() == device_name_lower:
                print(f"[Микрофон] Найдено устройство (точное): [{i}] {dev['name']}")
                return i
    
    # Потом ищем частичное совпадение
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            if device_name_lower in dev['name'].lower():
                print(f"[Микрофон] Найдено устройство (частичное): [{i}] {dev['name']}")
                return i
    
    print(f"[Микрофон] Устройство '{device_name}' не найдено")
    print("[Микрофон] Доступные устройства:")
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")
    
    return None


def block_mic():
    """Блокировка микрофона во время обработки"""
    global mic_blocked
    mic_blocked = True


async def send_to_kiko(text: str, is_partial: bool = False):
    """Отправляет распознанный текст в KIKO через WebSocket"""
    global ws_connection, ws_connected
    
    if not text or not text.strip():
        return
        
    # Пропускаем partials если отключено
    if is_partial and not options.get("send_partials", False):
        return
    
    message = json.dumps({
        "type": "transcription",
        "text": text.strip(),
        "partial": is_partial,
        "source": "irene_vosk",
        "timestamp": time.time()
    })
    
    if ws_connection and ws_connected:
        try:
            await ws_connection.send(message)
            if not is_partial:
                print(f"[KIKO] Отправлено: {text}")
        except Exception as e:
            print(f"[KIKO] Ошибка отправки: {e}")
            ws_connected = False
    else:
        # Кладём в очередь для отправки позже
        if not is_partial:
            text_queue.put(message)


async def websocket_handler():
    """Обработчик WebSocket соединения с KIKO"""
    global ws_connection, ws_connected
    
    ws_url = options.get("kiko_ws_url", "ws://localhost:3000/irene-asr")
    reconnect_delay = options.get("reconnect_delay", 3)
    
    while True:
        try:
            print(f"[KIKO] Подключение к {ws_url}...")
            
            async with websockets.connect(ws_url) as websocket:
                ws_connection = websocket
                ws_connected = True
                print(f"[KIKO] Подключено к {ws_url}")
                
                # Отправляем накопленные сообщения
                while not text_queue.empty():
                    try:
                        msg = text_queue.get_nowait()
                        await websocket.send(msg)
                    except:
                        break
                
                # Слушаем ответы от KIKO (опционально)
                while True:
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(), 
                            timeout=30.0
                        )
                        data = json.loads(response)
                        
                        # Обработка команд от KIKO
                        if data.get("type") == "command":
                            cmd = data.get("command")
                            if cmd == "mute":
                                block_mic()
                            elif cmd == "unmute":
                                global mic_blocked
                                mic_blocked = False
                                
                    except asyncio.TimeoutError:
                        # Ping для поддержания соединения
                        await websocket.ping()
                    except websockets.ConnectionClosed:
                        break
                        
        except Exception as e:
            print(f"[KIKO] Ошибка WebSocket: {e}")
            ws_connected = False
            ws_connection = None
            
        print(f"[KIKO] Переподключение через {reconnect_delay} сек...")
        await asyncio.sleep(reconnect_delay)


def run_websocket_loop():
    """Запускает WebSocket loop в отдельном потоке"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler())


def send_text_sync(text: str, is_partial: bool = False):
    """Синхронная обёртка для отправки текста"""
    if not text or not text.strip():
        return
        
    if is_partial and not options.get("send_partials", False):
        return
    
    message = json.dumps({
        "type": "transcription", 
        "text": text.strip(),
        "partial": is_partial,
        "source": "irene_vosk",
        "timestamp": time.time()
    })
    
    if ws_connected and ws_connection:
        # Отправляем через asyncio
        try:
            future = asyncio.run_coroutine_threadsafe(
                ws_connection.send(message),
                asyncio.get_event_loop()
            )
            future.result(timeout=1.0)
            if not is_partial:
                print(f"[→ KIKO] {text}")
        except Exception as e:
            text_queue.put(message)
    else:
        if not is_partial:
            text_queue.put(message)
            print(f"[Очередь] {text}")


def rtsp_audio_stream(rtsp_url: str, sample_rate: int, audio_queue: queue.Queue):
    """
    Получает аудио из RTSP потока через ffmpeg и кладёт в очередь.
    """
    print(f"[RTSP] Подключение к {rtsp_url}")
    
    # ffmpeg команда для извлечения аудио из RTSP
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-vn',  # без видео
        '-acodec', 'pcm_s16le',  # 16-bit PCM
        '-ar', str(sample_rate),  # sample rate
        '-ac', '1',  # моно
        '-f', 's16le',  # raw PCM
        '-loglevel', 'error',
        '-'  # вывод в stdout
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=8000 * 2  # 16-bit = 2 bytes per sample
        )
        
        print(f"[RTSP] Подключено, получаю аудио...")
        
        # Читаем аудио блоками
        block_size = 8000 * 2  # 8000 samples * 2 bytes
        
        while True:
            if mic_blocked:
                time.sleep(0.1)
                continue
                
            data = process.stdout.read(block_size)
            if not data:
                # Проверяем, жив ли процесс
                if process.poll() is not None:
                    stderr = process.stderr.read().decode()
                    print(f"[RTSP] Процесс завершился: {stderr}")
                    break
                continue
            
            audio_queue.put(data)
            
    except Exception as e:
        print(f"[RTSP] Ошибка: {e}")
    finally:
        if 'process' in locals():
            process.kill()


# ------------------- VOSK Main ------------------
if __name__ == "__main__":
    q = queue.Queue()

    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def callback(indata, frames, time, status):
        """Callback для аудио потока"""
        if status:
            print(status, file=sys.stderr)
        if not mic_blocked:
            q.put(bytes(indata))

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    
    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)
        
    parser = argparse.ArgumentParser(
        description="Irene Voice Assistant для KIKO - транскрибация через VOSK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-m', '--model', type=str, metavar='MODEL_PATH',
        help='Path to the VOSK model')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, 
        help='sampling rate')
    parser.add_argument(
        '-k', '--kiko-url', type=str,
        help='KIKO WebSocket URL')
    parser.add_argument(
        '--rtsp', type=str, metavar='RTSP_URL',
        help='RTSP URL для получения аудио с камеры (вместо локального микрофона)')
    parser.add_argument(
        '--source', type=str, choices=['auto', 'rtsp', 'mic'], default='auto',
        help='Источник аудио: auto (авто), rtsp (камера), mic (локальный микрофон)')
    args = parser.parse_args(remaining)

    # Настраиваем логирование
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('runva_kiko')

    # Загружаем настройки из KIKO .env
    kiko_env = load_kiko_env()

    # Применяем аргументы командной строки
    model_path = args.model or options.get("model_path", "model")
    sample_rate = args.samplerate or options.get("sample_rate")
    
    # Определяем устройство микрофона:
    # 1. Из аргумента командной строки
    # 2. Из options/kiko.json
    # 3. Из KIKO .env (MIC_DEVICE / PULSE_SOURCE)
    # 4. Default системы
    device = args.device
    if device is None:
        device = options.get("device")
    if device is None:
        # Пробуем из KIKO .env
        kiko_mic = get_microphone_device(kiko_env)
        if kiko_mic:
            device = find_device_by_name(kiko_mic)
    
    if args.kiko_url:
        options["kiko_ws_url"] = args.kiko_url
    
    # Также можно взять KIKO WS URL из .env
    if not args.kiko_url and kiko_env.get('USE_IRENE_ASR') == '1':
        # Формируем URL из настроек KIKO
        kiko_port = kiko_env.get('PORT', '3000')
        options["kiko_ws_url"] = f"ws://localhost:{kiko_port}/irene-asr"

    try:
        # Проверяем модель
        if not os.path.exists(model_path):
            print(f"[ОШИБКА] Модель VOSK не найдена: {model_path}")
            print("Скачайте модель с https://alphacephei.com/vosk/models")
            print("и распакуйте в папку 'model'")
            sys.exit(1)

        # Определяем источник аудио
        audio_source = args.source or options.get("audio_source", "auto")
        rtsp_url = args.rtsp or options.get("rtsp_url") or kiko_env.get("RTSP_URL")
        
        # Автоопределение: если есть RTSP_URL в KIKO .env — используем его
        if audio_source == "auto":
            if rtsp_url:
                audio_source = "rtsp"
                print(f"[AUDIO] Автоопределение: найден RTSP_URL в KIKO .env")
            else:
                audio_source = "mic"
                print(f"[AUDIO] Автоопределение: используем локальный микрофон")
        
        # Определяем sample rate
        if sample_rate is None:
            if audio_source == "rtsp":
                sample_rate = 16000  # стандартный для VOSK
            else:
                device_info = sd.query_devices(device, 'input')
                sample_rate = int(device_info['default_samplerate'])

        print(f"[VOSK] Загрузка модели: {model_path}")
        model = vosk.Model(model_path)
        print(f"[VOSK] Модель загружена")

        # Запускаем WebSocket клиент в отдельном потоке
        ws_thread = threading.Thread(target=run_websocket_loop, daemon=True)
        ws_thread.start()
        print(f"[KIKO] WebSocket клиент запущен")

        rec = vosk.KaldiRecognizer(model, sample_rate)
        
        # === РЕЖИМ RTSP ===
        if audio_source == "rtsp":
            if not rtsp_url:
                print("[ОШИБКА] RTSP URL не указан!")
                print("Укажите --rtsp URL или добавьте RTSP_URL в KIKO .env")
                sys.exit(1)
            
            print('#' * 60)
            print('Irene VOSK для KIKO (RTSP режим)')
            print(f'KIKO URL: {options.get("kiko_ws_url")}')
            print(f'RTSP: {rtsp_url.split("@")[-1] if "@" in rtsp_url else rtsp_url}')
            print(f'Sample rate: {sample_rate}')
            print('Нажмите Ctrl+C для остановки')
            print('#' * 60)
            
            # Запускаем RTSP поток в отдельном потоке
            rtsp_thread = threading.Thread(
                target=rtsp_audio_stream, 
                args=(rtsp_url, sample_rate, q),
                daemon=True
            )
            rtsp_thread.start()
            
            # Обрабатываем аудио
            while True:
                try:
                    data = q.get(timeout=5.0)
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "")
                        
                        if text:
                            print(f"[VOSK] Распознано: {text}")
                            send_text_sync(text, is_partial=False)
                    else:
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "")
                        
                        if partial_text and options.get("send_partials", False):
                            send_text_sync(partial_text, is_partial=True)
                            
                except queue.Empty:
                    continue
        
        # === РЕЖИМ ЛОКАЛЬНОГО МИКРОФОНА ===
        else:
            # Запускаем аудио поток
            with sd.RawInputStream(
                samplerate=sample_rate, 
                blocksize=8000, 
                device=device, 
                dtype='int16',
                channels=1, 
                callback=callback
            ):
                # Получаем информацию о реальном устройстве
                actual_device_info = sd.query_devices(device, 'input')
                actual_device_name = actual_device_info['name'] if actual_device_info else 'default'
                
                print('#' * 60)
                print('Irene VOSK для KIKO (микрофон)')
                print(f'KIKO URL: {options.get("kiko_ws_url")}')
                print(f'Sample rate: {sample_rate}')
                print(f'Микрофон: {actual_device_name}')
                if device is not None:
                    print(f'Device ID: {device}')
                print('Нажмите Ctrl+C для остановки')
                print('#' * 60)

                while True:
                    data = q.get()
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "")
                        
                        if text:
                            print(f"[VOSK] Распознано: {text}")
                            send_text_sync(text, is_partial=False)
                            mic_blocked = False
                    else:
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "")
                        
                        if partial_text and options.get("send_partials", False):
                            send_text_sync(partial_text, is_partial=True)

    except KeyboardInterrupt:
        print('\n[ВЫХОД] Остановка...')
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
