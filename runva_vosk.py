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
# KIKO запущен через docker-compose с network_mode: host, PORT: 3001
KIKO_URL = os.environ.get("KIKO_URL", "http://127.0.0.1:3001/ai")
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

# Smart Turn - настройки умного склеивания фраз
SMART_TURN_TIMEOUT = 7.0  # секунд ждать продолжения после wake word
SMART_TURN_MAX_PHRASES = 10  # максимум фраз для склеивания

import time

# Состояние Smart Turn
class SmartTurnState:
    def __init__(self):
        self.wake_detected = False
        self.wake_time = 0
        self.phrases = []
    
    def reset(self):
        self.wake_detected = False
        self.wake_time = 0
        self.phrases = []
    
    def is_active(self):
        """Проверяет, активен ли режим ожидания продолжения"""
        if not self.wake_detected:
            return False
        elapsed = time.time() - self.wake_time
        return elapsed < SMART_TURN_TIMEOUT
    
    def extend_timeout(self):
        """Продлевает таймаут - вызывать когда есть активность (partial результаты)"""
        if self.wake_detected:
            self.wake_time = time.time()
    
    def add_phrase(self, text: str):
        """Добавляет фразу в буфер"""
        self.phrases.append(text)
        if not self.wake_detected:
            self.wake_detected = True
        self.wake_time = time.time()  # Всегда обновляем время
    
    def get_full_text(self):
        """Возвращает склеенный текст"""
        return " ".join(self.phrases)
    
    def should_send(self, has_wake_word: bool, text: str):
        """
        Определяет, нужно ли отправлять в KIKO.
        Возвращает (should_send, full_text)
        """
        # Если это новая фраза с wake word
        if has_wake_word:
            if self.is_active():
                # Уже был wake word - отправляем накопленное + новое
                self.add_phrase(text)
                full_text = self.get_full_text()
                self.reset()
                return True, full_text
            else:
                # Новый wake word - начинаем накапливать
                self.reset()
                self.add_phrase(text)
                # Ждём ещё - вдруг будет продолжение
                return False, None
        
        # Фраза без wake word
        if self.is_active():
            # Продолжение после wake word
            self.add_phrase(text)
            
            # Если набрали максимум фраз - отправляем
            if len(self.phrases) >= SMART_TURN_MAX_PHRASES:
                full_text = self.get_full_text()
                self.reset()
                return True, full_text
            
            # Иначе ждём ещё
            return False, None
        
        # Нет wake word и не в режиме ожидания
        return False, None
    
    def flush_if_timeout(self):
        """
        Проверяет таймаут и возвращает накопленный текст если пора.
        Вызывать периодически.
        """
        if self.wake_detected and len(self.phrases) > 0:
            elapsed = time.time() - self.wake_time
            if elapsed >= SMART_TURN_TIMEOUT:
                full_text = self.get_full_text()
                self.reset()
                return full_text
        return None

smart_turn = SmartTurnState()

mic_blocked = False
rtsp_process = None

def block_mic():
    global mic_blocked
    mic_blocked = True

def unblock_mic():
    global mic_blocked
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


def send_to_kiko(text: str, kiko_url: str):
    """
    Отправляет текст в KIKO AI через HTTP POST.
    Отправляет полный текст - KIKO сам обработает wake word.
    """
    print(f"[KIKO] → Отправка: '{text}'")
    
    try:
        payload = {
            "sessionId": SESSION_ID,
            "prompt": text,
            "source": "asr"
        }
        
        response = requests.post(
            kiko_url,
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200 or response.status_code == 201:
            try:
                result = response.json()
                if "response" in result:
                    answer = result['response']
                    # Показываем первые 150 символов ответа
                    preview = answer[:150] + '...' if len(answer) > 150 else answer
                    print(f"[KIKO] ✓ Ответ: {preview}")
                    return True
                elif "error" in result:
                    if result['error'] == 'busy':
                        print(f"[KIKO] ⚠ KIKO занят - повторите через пару секунд")
                    else:
                        print(f"[KIKO] ✗ Ошибка: {result['error']}")
                else:
                    print(f"[KIKO] ✓ OK")
                    return True
            except:
                print(f"[KIKO] ✓ Получен ответ")
                return True
        else:
            print(f"[KIKO] ✗ HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"[KIKO] ✗ Нет связи с {kiko_url}")
    except requests.exceptions.Timeout:
        print("[KIKO] ✗ Таймаут (60 сек)")
    except Exception as e:
        print(f"[KIKO] ✗ Ошибка: {e}")
    
    return False


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


def process_voice_input(voice_input_str: str, kiko_url: str):
    """
    Обрабатывает распознанную речь с Smart Turn.
    Склеивает фразы если wake word был сказан с паузой.
    """
    global smart_turn
    
    # Проверяем наличие wake word
    found_wake = check_wake_word(voice_input_str)
    has_wake = found_wake is not None
    
    if has_wake:
        print(f"[WAKE] Обнаружено wake word '{found_wake}'")
    
    # Smart Turn логика
    should_send, full_text = smart_turn.should_send(has_wake, voice_input_str)
    
    if should_send and full_text:
        print(f"[SMART] Отправка склеенной фразы: '{full_text}'")
        send_to_kiko(full_text, kiko_url)
        return True
    
    if smart_turn.is_active():
        print(f"[SMART] Ожидание продолжения... ({len(smart_turn.phrases)} фраз)")
        return False
    
    if not has_wake:
        print(f"[SKIP] Нет wake word в: '{voice_input_str}'")
    
    return False


def check_smart_turn_timeout(kiko_url: str):
    """Проверяет таймаут Smart Turn и отправляет если нужно"""
    global smart_turn
    
    full_text = smart_turn.flush_if_timeout()
    if full_text:
        print(f"[SMART] Таймаут - отправка: '{full_text}'")
        send_to_kiko(full_text, kiko_url)
        return True
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
        default=os.environ.get("KIKO_URL", "http://127.0.0.1:3001/ai"),
        help='URL KIKO AI сервера (по умолчанию http://127.0.0.1:3001/ai)')
    args = parser.parse_args(remaining)
    
    # Используем URL из аргументов
    kiko_url = args.kiko_url

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
        print(f'Smart Turn: {SMART_TURN_TIMEOUT}s таймаут, до {SMART_TURN_MAX_PHRASES} фраз')
        print(f'KIKO URL: {kiko_url}')
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
                    data = q.get(timeout=1.0)  # Короткий таймаут для проверки Smart Turn
                    
                    if rec.AcceptWaveform(data):
                        recognized_data = json.loads(rec.Result())
                        voice_input_str = recognized_data["text"]
                        
                        if voice_input_str != "":
                            print(f"[РАСПОЗНАНО] {voice_input_str}")
                            # Обработка через KIKO (если есть wake word)
                            block_mic()
                            process_voice_input(voice_input_str, kiko_url)
                            # Очищаем очередь и разблокируем
                            clear_queue(q)
                            unblock_mic()
                    else:
                        # Partial результат - человек ещё говорит, продлеваем таймаут
                        partial = json.loads(rec.PartialResult())
                        if partial.get("partial", "").strip():
                            smart_turn.extend_timeout()
                    
                    # Проверяем таймаут Smart Turn
                    check_smart_turn_timeout(kiko_url)
                    
                    if dump_fn is not None:
                        dump_fn.write(data)
                        
                except queue.Empty:
                    # Проверяем таймаут Smart Turn даже если нет данных
                    check_smart_turn_timeout(kiko_url)
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
                    try:
                        data = q.get(timeout=1.0)  # Короткий таймаут для Smart Turn
                    except queue.Empty:
                        # Проверяем таймаут Smart Turn
                        check_smart_turn_timeout(kiko_url)
                        continue
                    
                    if rec.AcceptWaveform(data):
                        recognized_data = rec.Result()
                        recognized_data = json.loads(recognized_data)
                        voice_input_str = recognized_data["text"]

                        if voice_input_str != "":
                            print(f"[РАСПОЗНАНО] {voice_input_str}")
                            # Обработка через KIKO (если есть wake word)
                            block_mic()
                            process_voice_input(voice_input_str, kiko_url)
                            unblock_mic()
                    
                    # Проверяем таймаут Smart Turn
                    check_smart_turn_timeout(kiko_url)

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


