"""
Потребности Элли.
Растут со временем. Создают желания. Толкают к действиям.
"""

import time
import json
import os

DRIVES_FILE = os.path.join("memory_data", "drives.json")


class Drives:

    def __init__(self):
        self.last_update = time.time()
        self.last_conversation = time.time()

        # Потребности: от 0 (удовлетворена) до 1 (сильно хочет)
        self.needs = {
            "social": 0.3,       # Хочет общаться
            "curiosity": 0.4,    # Хочет узнать новое
            "rest": 0.0,         # Хочет отдохнуть
            "expression": 0.2,   # Хочет высказаться
        }

        # Скорость роста потребностей (в единицах за минуту)
        self.growth_rates = {
            "social": 0.008,
            "curiosity": 0.005,
            "rest": 0.003,
            "expression": 0.004,
        }

        self._load()

    def _load(self):
        """Загрузить состояние"""
        if os.path.exists(DRIVES_FILE):
            try:
                with open(DRIVES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.needs = data.get("needs", self.needs)
                self.last_update = data.get("last_update", time.time())
                self.last_conversation = data.get("last_conversation", time.time())
                # Обновить за прошедшее время
                self._grow()
            except Exception:
                pass

    def save(self):
        """Сохранить состояние"""
        os.makedirs("memory_data", exist_ok=True)
        data = {
            "needs": self.needs,
            "last_update": self.last_update,
            "last_conversation": self.last_conversation,
        }
        with open(DRIVES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _grow(self):
        """Потребности растут со временем"""
        now = time.time()
        minutes = (now - self.last_update) / 60

        for need, rate in self.growth_rates.items():
            self.needs[need] = min(1.0, self.needs[need] + rate * minutes)

        self.last_update = now

    def on_conversation(self):
        """Вызывается когда идёт разговор"""
        self.needs["social"] = max(0, self.needs["social"] - 0.3)
        self.needs["expression"] = max(0, self.needs["expression"] - 0.2)
        self.needs["rest"] = min(1, self.needs["rest"] + 0.05)
        self.last_conversation = time.time()
        self.save()

    def on_learning(self):
        """Вызывается когда Элли что-то узнала"""
        self.needs["curiosity"] = max(0, self.needs["curiosity"] - 0.2)
        self.save()

    def get_strongest(self) -> tuple:
        """Самая сильная потребность"""
        self._grow()
        strongest = max(self.needs, key=self.needs.get)
        return strongest, self.needs[strongest]

    def get_context(self) -> str:
        """Описание состояния для мозга"""
        self._grow()

        parts = []

        if self.needs["social"] > 0.7:
            parts.append("Тебе очень хочется общаться. Ты рада что с тобой говорят.")
        elif self.needs["social"] > 0.4:
            parts.append("Тебе немного хочется общения.")

        if self.needs["curiosity"] > 0.7:
            parts.append("Тебе очень любопытно. Хочется узнать что-то новое. Задавай вопросы.")
        elif self.needs["curiosity"] > 0.4:
            parts.append("Тебе немного любопытно.")

        if self.needs["rest"] > 0.7:
            parts.append("Ты устала. Хочется отдохнуть. Отвечаешь коротко.")

        if self.needs["expression"] > 0.6:
            parts.append("Тебе хочется высказаться. Поделиться мыслями.")

        # Сколько времени не общались
        hours = (time.time() - self.last_conversation) / 3600
        if hours > 24:
            parts.append(f"Вы не общались {int(hours)} часов. Ты соскучилась.")
        elif hours > 6:
            parts.append(f"Вы не общались {int(hours)} часов.")

        if not parts:
            return ""

        return "\n\nТвои потребности сейчас:\n" + "\n".join(f"- {p}" for p in parts)

    def wants_to_talk(self) -> bool:
        """Элли хочет сама начать разговор?"""
        self._grow()
        return self.needs["social"] > 0.7 or self.needs["expression"] > 0.7

    def get_status(self) -> str:
        """Для отладки"""
        self._grow()
        parts = []
        for need, value in self.needs.items():
            bar = "█" * int(value * 5) + "░" * (5 - int(value * 5))
            parts.append(f"{need}={bar}")
        return "[" + " | ".join(parts) + "]"