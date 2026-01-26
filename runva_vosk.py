import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import logging
import json
import subprocess
import threading
import requests

# Настройки KIKO
KIKO_URL = os.environ.get("KIKO_URL", "http://server:3000/ai")
SESSION_ID = "vosk-session-1"

# Wake words - разные вариации на русском и английском
WAKE_WORDS = [
    # Русский
    "оптимус",
    "оптимуса", 
    "оптимусу",
    "оптимусом",
    "оптимусе",
    # Английский
    "optimus",
    "optimous",
    "optimis",
    # Транслит/ошибки распознавания
    "оптимас",
    "оптимос",
    "оптимес",
]

mic_blocked = False
rtsp_process = None

def block_mic():
    global mic_blocked
    print("[DEBUG] Блокировка микрофона")
    mic_blocked = True

def unblock_mic():
    global mic_blocked
    print("[DEBUG] Разблокировка микрофона")
    mic_blocked = False

def clear_queue(q):
    """Очищает очередь от накопившихся данных"""
    cleared = 0
    while not q.empty():
        try:
            q.get_nowait()
            cleared += 1
        except:
            break
    if cleared > 0:
        print(f"[DEBUG] Очищено {cleared} буферов из очереди")


def send_to_kiko(text: str):
    """
    Отправляет текст в KIKO AI через HTTP POST.
    Отправляет полный текст - KIKO сам обработает wake word.
    """
    print(f"[KIKO] Отправка в KIKO: '{text}'")
    
    try:
        payload = {
            "sessionId": SESSION_ID,
            "prompt": text,
            "source": "asr"
        }
        
        response = requests.post(
            KIKO_URL,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                print(f"[KIKO] Ответ: {result['response'][:200]}...")
            elif "error" in result:
                print(f"[KIKO] Ошибка: {result['error']}")
            else:
                print(f"[KIKO] Результат: {result}")
        else:
            print(f"[KIKO] HTTP ошибка {response.status_code}: {response.text[:200]}")
            
    except requests.exceptions.ConnectionError:
        print(f"[KIKO] Не удалось подключиться к {KIKO_URL}")
    except requests.exceptions.Timeout:
        print("[KIKO] Таймаут запроса")
    except Exception as e:
        print(f"[KIKO] Ошибка: {e}")


def check_wake_word(text: str) -> str | None:
    """
    Проверяет наличие любого wake word в тексте.
    Возвращает найденное wake word или None.
    """
    text_lower = text.lower()
    for wake in WAKE_WORDS:
        if wake in text_lower:
            return wake
    return None


def process_voice_input(voice_input_str: str):
    """
    Обрабатывает распознанную речь.
    Если содержит wake word (оптимус/optimus и вариации) - отправляет в KIKO.
    """
    # Проверяем наличие wake word
    found_wake = check_wake_word(voice_input_str)
    
    if found_wake:
        print(f"[WAKE] Обнаружено wake word '{found_wake}'")
        send_to_kiko(voice_input_str)
        return True
    
    # Если нет wake word - игнорируем
    print(f"[SKIP] Нет wake word в: '{voice_input_str}'")
    return False


def rtsp_audio_stream(rtsp_url: str, sample_rate: int, audio_queue: queue.Queue):
    """
    Получает аудио из RTSP потока через ffmpeg и кладёт в очередь.
    """
    global rtsp_process
    print(f"[RTSP] Подключение к {rtsp_url.split('@')[-1] if '@' in rtsp_url else rtsp_url}")
    
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', '1',
        '-f', 's16le',
        '-loglevel', 'error',
        '-'
    ]
    
    try:
        rtsp_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=8000 * 2
        )
        
        print(f"[RTSP] Подключено, получаю аудио...")
        
        block_size = 8000 * 2
        
        while True:
            # ВАЖНО: всегда читаем данные, иначе буфер ffmpeg переполнится!
            data = rtsp_process.stdout.read(block_size)
            if not data:
                if rtsp_process.poll() is not None:
                    stderr = rtsp_process.stderr.read().decode()
                    print(f"[RTSP] Процесс завершился: {stderr}")
                    break
                continue
            
            # Если микрофон заблокирован - просто выбрасываем данные
            if mic_blocked:
                continue
            
            try:
                audio_queue.put(data, timeout=0.5)
            except queue.Full:
                # Очередь переполнена, пропускаем буфер
                pass
            
    except Exception as e:
        print(f"[RTSP] Ошибка: {e}")
    finally:
        if rtsp_process:
            rtsp_process.kill()

# ------------------- vosk ------------------
if __name__ == "__main__":
    # Ограничиваем размер очереди чтобы избежать накопления
    q = queue.Queue(maxsize=50)



    def int_or_str(text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
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
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-f', '--filename', type=str, metavar='FILENAME',
        help='audio file to store recording to')
    parser.add_argument(
        '-m', '--model', type=str, metavar='MODEL_PATH',
        help='Path to the model')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-r', '--samplerate', type=int, help='sampling rate')
    parser.add_argument(
        '--rtsp', type=str, metavar='RTSP_URL',
        help='RTSP URL для получения аудио с IP камеры')
    parser.add_argument(
        '--kiko-url', type=str, metavar='KIKO_URL',
        default=os.environ.get("KIKO_URL", "http://server:3000/ai"),
        help='URL KIKO AI сервера (по умолчанию http://server:3000/ai)')
    args = parser.parse_args(remaining)
    
    # Обновляем глобальные настройки из аргументов
    global KIKO_URL
    KIKO_URL = args.kiko_url

    # настраиваем логирование
    logger = logging.getLogger('runva_vosk')  # задаём конкретное имя, иначе здесь будет  __main__

    try:
        if args.model is None:
            args.model = "model"
        if not os.path.exists(args.model):
            print ("Please download a model for your language from https://alphacephei.com/vosk/models")
            print ("and unpack as 'model' in the current folder.")
            parser.exit(0)
        
        # Определяем режим: RTSP или локальный микрофон
        use_rtsp = args.rtsp is not None
        
        if use_rtsp:
            # Для RTSP используем 16000 по умолчанию
            if args.samplerate is None:
                args.samplerate = 16000
        else:
            if args.samplerate is None:
                device_info = sd.query_devices(args.device, 'input')
                args.samplerate = int(device_info['default_samplerate'])

        model = vosk.Model(args.model)

        if args.filename:
            dump_fn = open(args.filename, "wb")
        else:
            dump_fn = None

        rec = vosk.KaldiRecognizer(model, args.samplerate)

        print('#' * 80)
        print('KIKO Voice Assistant via Irene STT')
        print(f'Wake words: {", ".join(WAKE_WORDS[:5])}...')
        print(f'KIKO URL: {KIKO_URL}')
        if use_rtsp:
            print(f'RTSP режим: {args.rtsp.split("@")[-1] if "@" in args.rtsp else args.rtsp}')
        print(f'Sample rate: {args.samplerate}')
        print('Press Ctrl+C to stop the recording')
        print('#' * 80)

        # === РЕЖИМ RTSP ===
        if use_rtsp:
            rtsp_thread = threading.Thread(
                target=rtsp_audio_stream,
                args=(args.rtsp, args.samplerate, q),
                daemon=True
            )
            rtsp_thread.start()
            
            while True:
                try:
                    data = q.get(timeout=5.0)
                    
                    if rec.AcceptWaveform(data):
                        recognized_data = json.loads(rec.Result())
                        voice_input_str = recognized_data["text"]
                        
                        if voice_input_str != "":
                            print(f"[РАСПОЗНАНО] {voice_input_str}")
                            # Обработка через KIKO (если есть wake word)
                            block_mic()
                            process_voice_input(voice_input_str)
                            # Очищаем очередь и разблокируем
                            clear_queue(q)
                            unblock_mic()
                    
                    if dump_fn is not None:
                        dump_fn.write(data)
                        
                except queue.Empty:
                    print("[DEBUG] Таймаут очереди - проверка соединения")
                    continue
                except Exception as e:
                    print(f"[ERROR] Ошибка в главном цикле: {e}")
                    unblock_mic()
                    continue
        
        # === РЕЖИМ ЛОКАЛЬНОГО МИКРОФОНА ===
        else:
            with sd.RawInputStream(samplerate=args.samplerate, blocksize=8000, device=args.device, dtype='int16',
                                   channels=1, callback=callback):

                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):

                        recognized_data = rec.Result()
                        recognized_data = json.loads(recognized_data)
                        voice_input_str = recognized_data["text"]

                        if voice_input_str != "":
                            print(f"[РАСПОЗНАНО] {voice_input_str}")
                            # Обработка через KIKO (если есть wake word)
                            block_mic()
                            process_voice_input(voice_input_str)
                            unblock_mic()
                    else:
                        pass

                    if dump_fn is not None:
                        dump_fn.write(data)

    except KeyboardInterrupt:
        print('\nDone')
        if rtsp_process:
            rtsp_process.kill()
        parser.exit(0)
    except Exception as e:
        logger.exception(e)
        parser.exit(type(e).__name__ + ': ' + str(e))


