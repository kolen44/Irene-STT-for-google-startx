#!/usr/bin/env python3
"""
KIKO Voice Assistant - Vosk STT клиент
Минимальный клиент для распознавания речи и отправки в KIKO.
"""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time

import requests
import vosk

# Опциональный sounddevice для локального микрофона
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# === КОНФИГУРАЦИЯ ===
KIKO_URL = os.environ.get("KIKO_URL", "http://127.0.0.1:3001/ai")
SESSION_ID = "vosk-session-1"
SMART_TURN_TIMEOUT = 3.0  # секунд тишины для отправки

# Wake words
WAKE_WORDS = frozenset([
    "оптимус", "оптимуса", "оптимусу", "оптимусом", "оптимусе",
    "optimus", "optimous", "optimis",
    "оптимас", "оптимос", "оптимес",
])

# === ГЛОБАЛЬНОЕ СОСТОЯНИЕ ===
mic_blocked = False
rtsp_process = None


class SmartTurn:
    """Накопление фраз после wake word, отправка по таймауту."""
    
    __slots__ = ('active', 'last_time', 'phrases')
    
    def __init__(self):
        self.active = False
        self.last_time = 0.0
        self.phrases = []
    
    def reset(self):
        self.active = False
        self.last_time = 0.0
        self.phrases.clear()
    
    def add(self, text: str):
        self.phrases.append(text)
        self.active = True
        self.last_time = time.time()
    
    def extend(self):
        """Продлить таймаут (при partial результатах)."""
        if self.active:
            self.last_time = time.time()
    
    def check_timeout(self) -> str | None:
        """Возвращает накопленный текст если истёк таймаут."""
        if self.active and self.phrases:
            if time.time() - self.last_time >= SMART_TURN_TIMEOUT:
                result = " ".join(self.phrases)
                self.reset()
                return result
        return None
    
    def status(self) -> str:
        if not self.active:
            return ""
        remaining = SMART_TURN_TIMEOUT - (time.time() - self.last_time)
        return f"[{len(self.phrases)} фраз, {remaining:.1f}с]"


smart_turn = SmartTurn()


def find_wake_word(text: str) -> str | None:
    """Ищет wake word в тексте."""
    words = text.lower().split()
    for word in words:
        if word in WAKE_WORDS:
            return word
    # Проверка на подстроку (если wake word склеился)
    text_lower = text.lower()
    for wake in WAKE_WORDS:
        if wake in text_lower:
            return wake
    return None


def send_to_kiko(text: str, url: str) -> bool:
    """Отправляет текст в KIKO."""
    print(f"[KIKO] → {text}")
    
    try:
        resp = requests.post(
            url,
            json={"sessionId": SESSION_ID, "prompt": text, "source": "asr"},
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if resp.status_code in (200, 201):
            try:
                data = resp.json()
                if "response" in data:
                    preview = data['response'][:100]
                    print(f"[KIKO] ✓ {preview}{'...' if len(data['response']) > 100 else ''}")
                elif data.get("error") == "busy":
                    print("[KIKO] ⚠ Занят")
                else:
                    print("[KIKO] ✓")
            except:
                print("[KIKO] ✓")
            return True
        else:
            print(f"[KIKO] ✗ HTTP {resp.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"[KIKO] ✗ Нет связи")
    except requests.exceptions.Timeout:
        print("[KIKO] ✗ Таймаут")
    except Exception as e:
        print(f"[KIKO] ✗ {e}")
    
    return False


def process_text(text: str, kiko_url: str):
    """Обрабатывает распознанный текст."""
    wake = find_wake_word(text)
    
    if wake:
        print(f"[WAKE] '{wake}' → накапливаю")
        smart_turn.add(text)
    elif smart_turn.active:
        smart_turn.add(text)
        print(f"[+] {smart_turn.status()}")
    else:
        print(f"[SKIP] {text}")


def check_and_send(kiko_url: str) -> bool:
    """Проверяет таймаут и отправляет если нужно."""
    text = smart_turn.check_timeout()
    if text:
        print(f"[SEND] Таймаут → отправка")
        send_to_kiko(text, kiko_url)
        return True
    return False


def rtsp_stream(url: str, sample_rate: int, audio_queue: queue.Queue):
    """Читает аудио из RTSP через ffmpeg."""
    global rtsp_process
    
    display_url = url.split('@')[-1] if '@' in url else url
    print(f"[RTSP] Подключение: {display_url}")
    
    cmd = [
        'ffmpeg', '-rtsp_transport', 'tcp', '-i', url,
        '-vn', '-acodec', 'pcm_s16le', '-ar', str(sample_rate),
        '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
    ]
    
    try:
        rtsp_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=16000
        )
        print("[RTSP] OK")
        
        block_size = 8000 * 2  # 0.5 сек при 16kHz
        
        while True:
            data = rtsp_process.stdout.read(block_size)
            if not data:
                if rtsp_process.poll() is not None:
                    print("[RTSP] Отключено")
                    break
                continue
            
            if not mic_blocked:
                try:
                    audio_queue.put_nowait(data)
                except queue.Full:
                    pass  # Пропускаем если очередь полная
                    
    except Exception as e:
        print(f"[RTSP] Ошибка: {e}")
    finally:
        if rtsp_process:
            rtsp_process.kill()


def main():
    global mic_blocked, rtsp_process
    
    parser = argparse.ArgumentParser(description='KIKO Voice Assistant - Vosk STT')
    parser.add_argument('-m', '--model', default='model', help='Путь к модели Vosk')
    parser.add_argument('-r', '--samplerate', type=int, help='Sample rate')
    parser.add_argument('--rtsp', metavar='URL', help='RTSP URL камеры')
    parser.add_argument('--kiko-url', default=KIKO_URL, help='URL KIKO сервера')
    parser.add_argument('-d', '--device', help='Аудио устройство (для микрофона)')
    parser.add_argument('-l', '--list-devices', action='store_true', help='Показать устройства')
    args = parser.parse_args()
    
    if args.list_devices:
        if HAS_SOUNDDEVICE:
            print(sd.query_devices())
        else:
            print("sounddevice не установлен")
        return
    
    # Проверка модели
    if not os.path.exists(args.model):
        print(f"Модель не найдена: {args.model}")
        print("Скачайте с https://alphacephei.com/vosk/models")
        return 1
    
    use_rtsp = args.rtsp is not None
    
    if not use_rtsp and not HAS_SOUNDDEVICE:
        print("Для режима микрофона нужен sounddevice: pip install sounddevice")
        return 1
    
    # Sample rate
    if args.samplerate:
        sample_rate = args.samplerate
    elif use_rtsp:
        sample_rate = 16000
    else:
        device_info = sd.query_devices(args.device, 'input')
        sample_rate = int(device_info['default_samplerate'])
    
    # Загрузка модели
    print(f"[VOSK] Загрузка модели: {args.model}")
    model = vosk.Model(args.model)
    rec = vosk.KaldiRecognizer(model, sample_rate)
    print("[VOSK] OK")
    
    # Инфо
    print("=" * 60)
    print(f"KIKO: {args.kiko_url}")
    print(f"Wake: {', '.join(list(WAKE_WORDS)[:3])}...")
    print(f"Таймаут: {SMART_TURN_TIMEOUT}с")
    if use_rtsp:
        display = args.rtsp.split('@')[-1] if '@' in args.rtsp else args.rtsp
        print(f"RTSP: {display}")
    print(f"Rate: {sample_rate}")
    print("Ctrl+C для выхода")
    print("=" * 60)
    
    audio_q = queue.Queue(maxsize=50)
    
    try:
        if use_rtsp:
            # RTSP режим
            thread = threading.Thread(
                target=rtsp_stream,
                args=(args.rtsp, sample_rate, audio_q),
                daemon=True
            )
            thread.start()
            
            while True:
                try:
                    data = audio_q.get(timeout=0.5)
                except queue.Empty:
                    check_and_send(args.kiko_url)
                    continue
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"[→] {text}")
                        mic_blocked = True
                        process_text(text, args.kiko_url)
                        # Очистка очереди
                        while not audio_q.empty():
                            try:
                                audio_q.get_nowait()
                            except:
                                break
                        mic_blocked = False
                else:
                    partial = json.loads(rec.PartialResult())
                    if partial.get("partial", "").strip():
                        smart_turn.extend()
                
                check_and_send(args.kiko_url)
        
        else:
            # Микрофон
            def callback(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                if not mic_blocked:
                    audio_q.put(bytes(indata))
            
            with sd.RawInputStream(
                samplerate=sample_rate, blocksize=8000,
                device=args.device, dtype='int16', channels=1,
                callback=callback
            ):
                while True:
                    try:
                        data = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        check_and_send(args.kiko_url)
                        continue
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            print(f"[→] {text}")
                            mic_blocked = True
                            process_text(text, args.kiko_url)
                            mic_blocked = False
                    else:
                        partial = json.loads(rec.PartialResult())
                        if partial.get("partial", "").strip():
                            smart_turn.extend()
                    
                    check_and_send(args.kiko_url)
    
    except KeyboardInterrupt:
        print("\nВыход")
    finally:
        if rtsp_process:
            rtsp_process.kill()
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main() or 0)


