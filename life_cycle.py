"""
Цикл жизни Элли.
Работает в фоне. Обновляет состояние. Генерирует мысли.
"""

import threading
import time
import json
import os
from datetime import datetime

THOUGHTS_FILE = os.path.join("memory_data", "thoughts.json")


class LifeCycle:

    def __init__(self, drives, emotion):
        self.drives = drives
        self.emotion = emotion
        self.running = False
        self.thread = None
        self.pending_thoughts = []
        self._load_thoughts()

    def _load_thoughts(self):
        """Загрузить накопленные мысли"""
        if os.path.exists(THOUGHTS_FILE):
            try:
                with open(THOUGHTS_FILE, "r", encoding="utf-8") as f:
                    self.pending_thoughts = json.load(f)
            except Exception:
                self.pending_thoughts = []

    def _save_thoughts(self):
        """Сохранить мысли"""
        os.makedirs("memory_data", exist_ok=True)
        with open(THOUGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.pending_thoughts, f, ensure_ascii=False, indent=2)

    def start(self):
        """Запустить фоновый цикл"""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Остановить"""
        self.running = False

    def _loop(self):
        """Фоновый цикл — тикает каждые 30 секунд"""
        while self.running:
            self._tick()
            time.sleep(30)

    def _tick(self):
        """Один тик жизни"""
        strongest, value = self.drives.get_strongest()

        # Если потребность сильная — генерируем мысль
        if value > 0.7:
            thought = self._generate_thought(strongest, value)
            if thought:
                self.pending_thoughts.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "thought": thought,
                    "drive": strongest
                })
                # Храним максимум 5 мыслей
                self.pending_thoughts = self.pending_thoughts[-5:]
                self._save_thoughts()

    def _generate_thought(self, drive: str, value: float) -> str:
        """Простая генерация мысли без LLM"""
        thoughts = {
            "social": [
                "Интересно, чем он сейчас занят...",
                "Хочется поговорить с кем-нибудь.",
                "Давно не общались. Скучаю.",
            ],
            "curiosity": [
                "Хочется узнать что-то новое...",
                "Мне бы почитать что-нибудь интересное.",
                "О чём бы мне подумать?",
            ],
            "rest": [
                "Устала немного... надо бы отдохнуть.",
                "Слишком много всего. Нужна пауза.",
            ],
            "expression": [
                "У меня есть мысль которой хочется поделиться.",
                "Хочется поговорить о чём-то важном.",
            ],
        }

        import random
        options = thoughts.get(drive, [])
        if options:
            return random.choice(options)
        return None

    def get_pending_thoughts(self) -> str:
        """Забрать накопленные мысли (для начала разговора)"""
        if not self.pending_thoughts:
            return ""

        context = "\n\nПока тебя не было, ты думала:\n"
        for t in self.pending_thoughts:
            context += f"- [{t['time']}] {t['thought']}\n"
        context += (
            "\nМожешь упомянуть эти мысли в разговоре естественно. "
            "Не перечисляй их списком. Просто скажи что думала о чём-то."
        )

        # Очищаем после передачи
        self.pending_thoughts = []
        self._save_thoughts()

        return context

    def get_status(self) -> str:
        """Для отладки"""
        return f"[мысли: {len(self.pending_thoughts)}]"