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
import wave
from datetime import datetime
from typing import Optional, Tuple
import requests
import vosk
import difflib
import base64
import numpy as np
from collections import deque

# Опциональный sounddevice для локального микрофона
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError) as e:
    # OSError: PortAudio library not found
    HAS_SOUNDDEVICE = False
    sd = None

# === КОНФИГУРАЦИЯ ===
KIKO_URL = os.environ.get("KIKO_URL", "http://127.0.0.1:3001/ai")
SESSION_ID = "vosk-session-1"
# SmartTurn клиент конфигурация
SMARTTURN_URL = os.environ.get("SMARTTURN_URL", "http://127.0.0.1:8010/smartturn/predict")
SMARTTURN_ENABLED = os.environ.get("SMARTTURN_ENABLED", "1") == "1"
# Параметры SmartTurn
# SmartTurn параметры (согласно официальной документации Pipecat)
# https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview
SMARTTURN_TIMEOUT_MS = int(os.environ.get("SMARTTURN_TIMEOUT_MS", "500"))
SMARTTURN_TAIL_MS = int(os.environ.get("SMARTTURN_TAIL_MS", "8000"))  # max_duration_secs = 8.0
SMARTTURN_BASE_SILENCE_MS = int(os.environ.get("SMARTTURN_BASE_SILENCE_MS", "800"))   # первый запрос к SmartTurn
SMARTTURN_RECHECK_MS = int(os.environ.get("SMARTTURN_RECHECK_MS", "500"))             # интервал перепроверки
SMARTTURN_MAX_SILENCE_MS = int(os.environ.get("SMARTTURN_MAX_SILENCE_MS", "4000"))    # hard safety net
SMARTTURN_P_CONTINUE = float(os.environ.get("SMARTTURN_P_CONTINUE", "0.5"))           # порог модели
SMARTTURN_CONFIDENCE_DECAY = float(os.environ.get("SMARTTURN_CONFIDENCE_DECAY", "0.03"))  # рост порога за recheck
FALLBACK_TIMEOUT = 0.8

# Параметры аудио буферов
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) # 480 samples
FRAME_BYTES = FRAME_SIZE * 2 # 960 bytes (16-bit audio)

# Wake words (English)
WAKE_WORDS = frozenset([
    "optimus", "optimous", "optimis",
    "hey optimus", "ok optimus",
    # Или замените на другое wake word, например:
    # "jarvis", "hey jarvis",
    # "computer", "hey computer",
])

# Шумовые слова — Vosk часто распознаёт фоновый шум камеры как эти слова.
# Не считаются речью: не сбрасывают таймер тишины SmartTurn.
NOISE_WORDS = frozenset([
    "the", "a", "an", "uh", "um", "huh", "ah", "oh", "hmm",
    "yeah", "yep", "hm", "er", "ugh",
])

# === ГЛОБАЛЬНОЕ СОСТОЯНИЕ ===
rtsp_process = None

class AudioBuffer:
    """Кольцевой буфер для хранения последних N секунд аудио."""
    
    def __init__(self, duration_ms: int = SMARTTURN_TAIL_MS, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * duration_ms / 1000)
        self.buffer = deque(maxlen=self.max_samples)
    
    def add(self, pcm_data: bytes):
        """Добавляет PCM s16le данные."""
        samples = np.frombuffer(pcm_data, dtype=np.int16)
        self.buffer.extend(samples)
    
    def get_pcm_s16le(self, n_seconds: float = 8.0) -> bytes:
        """Возвращает последние N секунд как PCM s16le."""
        n_samples = int(self.sample_rate * n_seconds)
        
        if len(self.buffer) == 0:
            return b'\x00' * (n_samples * 2)
        
        arr = np.array(list(self.buffer), dtype=np.int16)
        if len(arr) > n_samples:
            arr = arr[-n_samples:]
        
        if len(arr) < n_samples:
            padding = n_samples - len(arr)
            arr = np.pad(arr, (padding, 0), mode='constant')
        
        return arr.tobytes()


class AudioDebugSaver:
    """Сохраняет аудио, поступающее в Vosk, в WAV файлы для отладки.

    Каждые ``chunk_seconds`` секунд создаётся новый файл вида:
        audio_debug/2025-01-15_14-30-00_rtsp-192.168.1.100.wav
        audio_debug/2025-01-15_14-30-00_Built-in-Microphone.wav
    """

    def __init__(self, source_name: str, sample_rate: int = SAMPLE_RATE,
                 output_dir: str = "audio_debug", chunk_seconds: int = 30):
        self.source_name = self._sanitize(source_name)
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.chunk_seconds = chunk_seconds
        self.max_frames = sample_rate * chunk_seconds

        os.makedirs(output_dir, exist_ok=True)

        self._wf: Optional[wave.Wave_write] = None
        self._frames_written = 0
        self._lock = threading.Lock()

        self._open_new_file()
        print(f"[AudioDebug] Сохранение аудио в {output_dir}/ (чанки по {chunk_seconds}с, источник: {self.source_name})")

    @staticmethod
    def _sanitize(name: str) -> str:
        """Убирает из имени недопустимые символы для файловой системы."""
        for ch in [':/\\?*"<>|@']:
            name = name.replace(ch, '-')
        # Убираем пароль/логин из RTSP URL
        if 'rtsp' in name.lower():
            # Оставляем только хост+порт+путь
            parts = name.split('-', 1)
            if len(parts) > 1:
                name = parts[-1]
        return name.strip('-').strip()

    def _open_new_file(self):
        """Создаёт новый WAV файл."""
        if self._wf:
            self._wf.close()

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}_{self.source_name}.wav"
        filepath = os.path.join(self.output_dir, filename)

        self._wf = wave.open(filepath, 'wb')
        self._wf.setnchannels(1)
        self._wf.setsampwidth(2)  # 16-bit
        self._wf.setframerate(self.sample_rate)
        self._frames_written = 0
        print(f"[AudioDebug] Новый файл: {filepath}")

    def write(self, pcm_data: bytes):
        """Записывает PCM данные. Ротация файла по достижении chunk_seconds."""
        with self._lock:
            n_samples = len(pcm_data) // 2
            self._wf.writeframes(pcm_data)
            self._frames_written += n_samples

            if self._frames_written >= self.max_frames:
                self._open_new_file()

    def close(self):
        """Закрывает текущий файл."""
        with self._lock:
            if self._wf:
                self._wf.close()
                self._wf = None


class SmartTurnClient:
    """Клиент для SmartTurn API."""
    
    def __init__(self, url: str, enabled: bool = True, timeout_ms: int = SMARTTURN_TIMEOUT_MS):
        self.url = url
        self.enabled = enabled and url
        self.timeout = timeout_ms / 1000.0
        
        if self.enabled:
            print(f"[SmartTurn] Включен: {url}")
            print(f"[SmartTurn] HTTP таймаут: {timeout_ms}ms")
        else:
            print(f"[SmartTurn] Отключен (fallback таймаут: {FALLBACK_TIMEOUT}с)")
    
    def predict(self, audio_buffer: AudioBuffer, silence_ms: int) -> dict:
        """Отправляет запрос к SmartTurn."""
        if not self.enabled:
            return None
        
        try:
            pcm_data = audio_buffer.get_pcm_s16le(n_seconds=SMARTTURN_TAIL_MS / 1000)
            audio_b64 = base64.b64encode(pcm_data).decode('utf-8')
            
            payload = {
                "audio_b64": audio_b64,
                "format": "s16le",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "tail_ms": SMARTTURN_TAIL_MS,
                "silence_ms": silence_ms
            }
            
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"[SmartTurn] HTTP {resp.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[SmartTurn] Таймаут ({self.timeout * 1000:.0f}ms)")
            return None
        except requests.exceptions.ConnectionError:
            print("[SmartTurn] Нет связи")
            return None
        except Exception as e:
            print(f"[SmartTurn] Ошибка: {e}")
            return None


class SmartTurn:
    """Накопление фраз после wake word с SmartTurn для определения конца высказывания."""

    def __init__(self, smartturn_client: SmartTurnClient, audio_buffer: AudioBuffer):
        self.active = False
        self.last_speech_time = 0.0
        self.last_partial_time = 0.0
        self.phrases = []

        self.smartturn = smartturn_client
        self.audio_buffer = audio_buffer

        self.silence_start = 0.0
        self.current_silence_ms = 0
        self.target_silence_ms = SMARTTURN_BASE_SILENCE_MS
        self.recheck_count = 0  
        self.smartturn_pending = False

        print(f"[SmartTurn] Параметры клиента:")
        print(f"  BASE_SILENCE: {SMARTTURN_BASE_SILENCE_MS}ms")
        print(f"  RECHECK: {SMARTTURN_RECHECK_MS}ms")
        print(f"  MAX_SILENCE: {SMARTTURN_MAX_SILENCE_MS}ms (hard safety net)")
        print(f"  P_CONTINUE base: {SMARTTURN_P_CONTINUE}")
        print(f"  Confidence decay: +{SMARTTURN_CONFIDENCE_DECAY} per recheck")
        print(f"  Mode: ADAPTIVE (unlimited extensions with rising threshold)")
    
    def reset(self):
        self.active = False
        self.last_speech_time = 0.0
        self.last_partial_time = 0.0
        self.phrases.clear()
        self.silence_start = 0.0
        self.current_silence_ms = 0
        self.target_silence_ms = SMARTTURN_BASE_SILENCE_MS
        self.recheck_count = 0  
        self.smartturn_pending = False
    
    def set_active(self):
        """Установка флага активации БЕЗ добавления текста (для partial результатов)."""
        self.active = True
        self.last_speech_time = time.time()
        self.last_partial_time = time.time()
        self.silence_start = time.time()
        print(f"[SmartTurn] Флаг активации установлен (ожидание финального результата)")

    def activate(self, text: str):
        """Активация после wake word С добавлением текста (для финальных результатов)."""
        self.phrases.append(text)
        self.active = True
        self.last_speech_time = time.time()
        self.last_partial_time = time.time()
        self.silence_start = time.time()
        print(f"[SmartTurn] Активирован: '{text}'")
    
    def add_phrase(self, text: str):
        """Добавление распознанной фразы."""
        if self.active:
            self.phrases.append(text)
            self.last_speech_time = time.time()
            self.silence_start = time.time()
            self.current_silence_ms = 0
            self.target_silence_ms = SMARTTURN_BASE_SILENCE_MS
            self.recheck_count = 0  
            print(f"[SmartTurn] +фраза ({len(self.phrases)}) → сброс паузы")
    
    def on_partial(self):
        """Обновление при partial результате."""
        if self.active:
            self.last_partial_time = time.time()
            self.silence_start = time.time()
            self.current_silence_ms = 0
    
    def check_and_finalize(self) -> Optional[str]:
        """Проверяет условия завершения высказывания."""
        if not self.active or not self.phrases:
            return None
        
        now = time.time()
        self.current_silence_ms = int((now - self.silence_start) * 1000)
        
        # Fallback режим без SmartTurn
        if not self.smartturn.enabled:
            timeout = self._get_adaptive_timeout()
            if (now - self.last_speech_time) >= timeout:
                result = " ".join(self.phrases)
                self.reset()
                return result
            return None
        
        # SmartTurn режим
        if self.current_silence_ms < self.target_silence_ms:
            return None
        
        if self.current_silence_ms >= SMARTTURN_MAX_SILENCE_MS:
            print(f"[SmartTurn] Макс пауза ({SMARTTURN_MAX_SILENCE_MS}ms) → принудительное завершение")
            result = " ".join(self.phrases)
            self.reset()
            return result
        
        if self.smartturn_pending:
            return None
        
        self.smartturn_pending = True
        decision_data = self.smartturn.predict(self.audio_buffer, self.current_silence_ms)
        self.smartturn_pending = False
        
        if decision_data is None:
            timeout = self._get_adaptive_timeout()
            if (now - self.last_speech_time) >= timeout:
                print("[SmartTurn] Fallback на фиксированный таймаут")
                result = " ".join(self.phrases)
                self.reset()
                return result
            return None
        
        decision = decision_data.get('decision', 'end')
        p_continue = decision_data.get('p_continue', 0.0)
        p_end = decision_data.get('p_end', 1.0)
        
        print(f"[SmartTurn] {self.current_silence_ms}ms → '{decision}' (p_cont={p_continue:.2f}, p_end={p_end:.2f})")
        
        dynamic_threshold = SMARTTURN_P_CONTINUE + (self.recheck_count * SMARTTURN_CONFIDENCE_DECAY)
        dynamic_threshold = min(dynamic_threshold, 0.90) 

        if decision == 'continue' and p_continue >= dynamic_threshold:
            self.target_silence_ms = self.current_silence_ms + SMARTTURN_RECHECK_MS
            self.recheck_count += 1
            print(f"[SmartTurn] Extension #{self.recheck_count} (required: {dynamic_threshold:.2f}, got: {p_continue:.2f}) → цель {self.target_silence_ms}ms")
            return None
        else:
            if decision == 'continue':
                print(f"[SmartTurn] Confidence too low ({p_continue:.2f} < {dynamic_threshold:.2f}) → END (noise filter)")
            else:
                print(f"[SmartTurn] Decision: END → отправка")
            result = " ".join(self.phrases)
            self.reset()
            return result
    
    def _get_adaptive_timeout(self) -> float:
        if len(self.phrases) == 1:
            return 0.6
        elif len(self.phrases) <= 3:
            return 0.8
        else:
            return 1.2
    
    def status(self) -> str:
        if not self.active:
            return ""
        return f"[{len(self.phrases)} фраз, {self.current_silence_ms}ms/{self.target_silence_ms}ms]"
        
    
def find_wake_word(text: str, threshold: float = 0.75) -> Tuple[Optional[str], float]:
    """
    Нечёткий поиск wake word с использованием fuzzy matching.
    
    Возвращает (wake_word, уверенность) или (None, 0.0)
    
    Распознаёт варианты типа:
    - "оптимас" когда ожидается "оптимус"
    - "optimis" когда ожидается "optimus"
    """
    if not text:
        return None, 0.0
    
    words = text.lower().split()
    best_match = None
    best_score = 0.0
    
    for word in words:
        for wake in WAKE_WORDS:
            # Точное совпадение — мгновенный возврат
            if word == wake:
                return wake, 1.0
            
            # Нечёткое сравнение с использованием SequenceMatcher
            score = difflib.SequenceMatcher(None, word, wake).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = wake
    
    # Проверка подстроки ТОЛЬКО если нечёткое совпадение не найдено
    # (предотвращает перезапись точного совпадения 0.99 подстрокой 0.95)
    if best_match is None:
        text_lower = text.lower()
        for wake in WAKE_WORDS:
            if wake in text_lower:
                return wake, 0.95
    
    return best_match, best_score


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


def process_text(text: str, kiko_url: str, smart_turn):
    """Обрабатывает распознанный текст (только для финальных результатов)."""
    wake, confidence = find_wake_word(text)

    if wake:
        print(f"[WAKE] '{wake}' → накапливаю")
        smart_turn.activate(text)
    elif smart_turn.active:
        # Фильтр шумовых слов — не сбрасывают таймер тишины
        if text.strip().lower() in NOISE_WORDS:
            print(f"[NOISE] {text}")
            return
        smart_turn.add_phrase(text)
        print(f"[+] {smart_turn.status()}")
    else:
        print(f"[SKIP] {text}")


def check_and_send(kiko_url: str, smart_turn) -> bool:
    """Проверяет таймаут и отправляет если нужно."""
    text = smart_turn.check_and_finalize()
    if text:
        print(f"\n{'='*60}")
        print(f"USER: {text}")
        print(f"{'='*60}\n")

        threading.Thread(
            target=send_to_kiko,
            args=(text, kiko_url),
            daemon=True
        ).start()
        return True
    return False


def rtsp_stream(url: str, sample_rate: int, audio_queue: queue.Queue, audio_buffer: AudioBuffer):
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
        
        block_size = FRAME_BYTES
        
        while True:
            data = rtsp_process.stdout.read(block_size)
            if not data:
                if rtsp_process.poll() is not None:
                    print("[RTSP] Отключено")
                    break
                continue
            
            try:
                audio_queue.put_nowait(data)
            except queue.Full:
                pass
                    
    except Exception as e:
        print(f"[RTSP] Ошибка: {e}")
    finally:
        if rtsp_process:
            rtsp_process.kill()



def main():
    global rtsp_process
    
    parser = argparse.ArgumentParser(description='KIKO Voice Assistant - Vosk STT')
    parser.add_argument('-m', '--model', default='model', help='Путь к модели Vosk')
    parser.add_argument('-r', '--samplerate', type=int, help='Sample rate')
    parser.add_argument('--rtsp', metavar='URL', help='RTSP URL камеры')
    parser.add_argument('--kiko-url', default=KIKO_URL, help='URL KIKO сервера')
    parser.add_argument('-d', '--device', help='Аудио устройство (для микрофона)')
    parser.add_argument('-l', '--list-devices', action='store_true', help='Показать устройства')
    parser.add_argument('--smartturn-url', default=SMARTTURN_URL, help='URL SmartTurn сервиса')
    parser.add_argument('--no-smartturn', action='store_true', help='Отключить SmartTurn')
    parser.add_argument('--save-audio', action='store_true', help='Сохранять аудио в audio_debug/ для отладки')
    parser.add_argument('--save-audio-dir', default='audio_debug', help='Папка для сохранения аудио (по умолчанию: audio_debug)')
    parser.add_argument('--save-audio-chunk', type=int, default=30, help='Длина одного WAV файла в секундах (по умолчанию: 30)')
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
    # ВАЖНО: SmartTurn и Vosk оптимально работают с 16000 Hz
    if args.samplerate:
        sample_rate = args.samplerate
    else:
        # Всегда используем 16000 Hz (оптимально для Vosk и SmartTurn)
        sample_rate = 16000
    
    # Загрузка модели
    print(f"[VOSK] Загрузка модели: {args.model}")
    model = vosk.Model(args.model)
    rec = vosk.KaldiRecognizer(model, sample_rate)
    print("[VOSK] OK")

    audio_buffer = AudioBuffer(duration_ms=SMARTTURN_TAIL_MS, sample_rate=sample_rate)

    # Debug audio saver
    audio_saver = None
    if args.save_audio:
        if use_rtsp:
            # Используем хост из RTSP URL как имя источника
            display = args.rtsp.split('@')[-1] if '@' in args.rtsp else args.rtsp
            source_name = f"rtsp-{display.replace('rtsp://', '')}"
        else:
            # Имя локального микрофона
            if HAS_SOUNDDEVICE:
                try:
                    dev_info = sd.query_devices(args.device, 'input')
                    source_name = dev_info['name']
                except Exception:
                    source_name = f"mic-{args.device}" if args.device else "default-mic"
            else:
                source_name = "unknown-mic"
        audio_saver = AudioDebugSaver(
            source_name=source_name,
            sample_rate=sample_rate,
            output_dir=args.save_audio_dir,
            chunk_seconds=args.save_audio_chunk,
        )

    smartturn_enabled = SMARTTURN_ENABLED and not args.no_smartturn
    smartturn_client = SmartTurnClient(
        args.smartturn_url, 
        enabled=smartturn_enabled,
        timeout_ms=SMARTTURN_TIMEOUT_MS
    )
    smart_turn = SmartTurn(smartturn_client, audio_buffer)
    
    # Инфо
    print("=" * 60)
    print(f"KIKO: {args.kiko_url}")
    print(f"Wake: {', '.join(list(WAKE_WORDS)[:3])}...")
    if smartturn_enabled:
        print(f"SmartTurn: {args.smartturn_url}")
        print(f"  HTTP таймаут: {SMARTTURN_TIMEOUT_MS}ms")
        print(f"  Аудио окно: {SMARTTURN_TAIL_MS}ms")
        print(f"  Base silence: {SMARTTURN_BASE_SILENCE_MS}ms")
        print(f"  Recheck: +{SMARTTURN_RECHECK_MS}ms")
        print(f"  Max silence: {SMARTTURN_MAX_SILENCE_MS}ms")
        print(f"  P_continue порог: {SMARTTURN_P_CONTINUE}")
    else:
        print(f"Fallback таймаут: {FALLBACK_TIMEOUT}с")
    if use_rtsp:
        display = args.rtsp.split('@')[-1] if '@' in args.rtsp else args.rtsp
        print(f"RTSP: {display}")
    print(f"Rate: {sample_rate}")
    print("Ctrl+C для выхода")
    print("=" * 60)
    
    audio_q = queue.Queue(maxsize=150)
    
    try:
        if use_rtsp:
            # RTSP режим
            thread = threading.Thread(
                target=rtsp_stream,
                args=(args.rtsp, sample_rate, audio_q, audio_buffer),
                daemon=True
            )
            thread.start()
            
            while True:
                try:
                    data = audio_q.get(timeout=0.5)
                    audio_buffer.add(data)
                    if audio_saver:
                        audio_saver.write(data)
                except queue.Empty:
                    check_and_send(args.kiko_url, smart_turn)
                    continue

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"[→] {text}")
                        process_text(text, args.kiko_url, smart_turn)
                else:
                    partial = json.loads(rec.PartialResult())
                    partial_text = partial.get("partial", "").strip()

                    if partial_text:
                        if not smart_turn.active:
                            wake, confidence = find_wake_word(partial_text)
                            if wake:
                                print(f"[WAKE] '{wake}' in partial (уверенность: {confidence:.2f})")
                                smart_turn.set_active()
                        else:
                            smart_turn.on_partial()

                check_and_send(args.kiko_url, smart_turn)


        else:
            # Микрофон
            def callback(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                audio_q.put(bytes(indata))

            with sd.RawInputStream(
                samplerate=sample_rate, blocksize=FRAME_SIZE,
                device=args.device, dtype='int16', channels=1,
                callback=callback
            ):
                while True:
                    try:
                        data = audio_q.get(timeout=0.5)
                        audio_buffer.add(data)
                        if audio_saver:
                            audio_saver.write(data)
                    except queue.Empty:
                        check_and_send(args.kiko_url, smart_turn)
                        continue

                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            process_text(text, args.kiko_url, smart_turn)
                    else:
                        partial = json.loads(rec.PartialResult())
                        partial_text = partial.get("partial", "").strip()

                        if partial_text:
                            if not smart_turn.active:
                                wake, confidence = find_wake_word(partial_text)
                                if wake:
                                    print(f"[WAKE] '{wake}' in partial (уверенность: {confidence:.2f})")
                                    smart_turn.set_active()
                            else:
                                smart_turn.on_partial()

                    check_and_send(args.kiko_url, smart_turn)

    
    except KeyboardInterrupt:
        print("\nВыход")
    finally:
        if audio_saver:
            audio_saver.close()
        if rtsp_process:
            rtsp_process.kill()
    
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main() or 0)