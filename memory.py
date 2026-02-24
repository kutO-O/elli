"""
Память Элли.
Сохраняет разговоры. Вспоминает к месту. Забывает неважное.
"""

import json
import os
import time
from datetime import datetime


MEMORY_DIR = "memory_data"
EPISODES_FILE = os.path.join(MEMORY_DIR, "episodes.json")
FACTS_FILE = os.path.join(MEMORY_DIR, "facts.json")


class Memory:

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.episodes = self._load(EPISODES_FILE, [])
        self.facts = self._load(FACTS_FILE, [])

    def _load(self, path, default):
        """Загрузить данные из файла"""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save(self):
        """Сохранить всё"""
        with open(EPISODES_FILE, "w", encoding="utf-8") as f:
            json.dump(self.episodes, f, ensure_ascii=False, indent=2)
        with open(FACTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)

    def save_episode(self, messages: list, emotion_valence: float):
        """Сохранить разговор как эпизод"""
        if not messages:
            return

        # Важность зависит от эмоций и длины
        importance = min(1.0, abs(emotion_valence) * 0.5 + len(messages) * 0.05)

        episode = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": messages[-20:],  # Последние 20 сообщений
            "importance": round(importance, 2),
            "emotion": round(emotion_valence, 2),
            "strength": 1.0  # Сила воспоминания (слабеет со временем)
        }

        self.episodes.append(episode)
        self._decay()
        self._save()

    def _decay(self):
        """Забывание: старые неважные воспоминания бледнеют"""
        now = time.time()
        alive = []

        for ep in self.episodes:
            # Сила падает со временем, но важные держатся
            ep["strength"] = max(0.0, ep["strength"] - 0.02 + ep["importance"] * 0.015)

            # Удаляем только совсем забытые
            if ep["strength"] > 0.05:
                alive.append(ep)

        self.episodes = alive

    def add_fact(self, fact: str, source: str = "разговор"):
        """Запомнить конкретный факт"""
        # Не дублировать
        for f in self.facts:
            if f["text"].lower() == fact.lower():
                f["strength"] = 1.0
                f["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                self._save()
                return

        self.facts.append({
            "text": fact,
            "source": source,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "strength": 1.0
        })
        self._save()

    def recall(self, query: str, max_results: int = 3) -> str:
        """Вспомнить что-то по теме"""
        query_words = set(query.lower().split())
        results = []

        # Ищем в фактах
        for fact in self.facts:
            fact_words = set(fact["text"].lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                score = overlap * fact["strength"]
                results.append(("fact", fact["text"], score))

        # Ищем в эпизодах
        for ep in self.episodes:
            ep_text = " ".join(
                m["content"] for m in ep["messages"]
            ).lower()
            ep_words = set(ep_text.split())
            overlap = len(query_words & ep_words)
            if overlap > 0:
                # Краткое содержание: первое и последнее сообщение
                summary = ep["messages"][0]["content"]
                if len(ep["messages"]) > 1:
                    summary += " ... " + ep["messages"][-1]["content"]
                score = overlap * ep["strength"] * ep["importance"]
                results.append(("episode", f"[{ep['date']}] {summary}", score))

        # Сортируем по релевантности
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:max_results]

        if not results:
            return ""

        memory_text = "\n\nТы вспоминаешь:\n"
        for type_, text, _ in results:
            if type_ == "fact":
                memory_text += f"- Факт: {text}\n"
            else:
                memory_text += f"- Из прошлого разговора: {text}\n"

        memory_text += (
            "\nИспользуй эти воспоминания естественно. "
            "Не говори 'я помню что...'. Просто используй как контекст."
        )

        return memory_text

    def extract_facts(self, user_msg: str, elli_response: str) -> None:
        """Извлечь факты из разговора (простой вариант)"""
        msg = user_msg.lower()

        # Шаблоны для извлечения фактов
        patterns = [
            ("меня зовут ", "Имя собеседника: "),
            ("я люблю ", "Собеседник любит "),
            ("мне нравится ", "Собеседнику нравится "),
            ("я работаю ", "Собеседник работает "),
            ("я живу ", "Собеседник живёт "),
            ("мне ", " лет", "Возраст собеседника: "),
            ("я не люблю ", "Собеседник не любит "),
            ("я ненавижу ", "Собеседник ненавидит "),
            ("мой любимый ", "Любимый "),
            ("моя любимая ", "Любимая "),
        ]

        for pattern in patterns:
            if len(pattern) == 2:
                trigger, prefix = pattern
                if trigger in msg:
                    idx = msg.index(trigger) + len(trigger)
                    value = user_msg[idx:].split(".")[0].split(",")[0].strip()
                    if value and len(value) < 100:
                        self.add_fact(prefix + value)
            elif len(pattern) == 3:
                start, end, prefix = pattern
                if start in msg and end in msg:
                    idx_s = msg.index(start) + len(start)
                    idx_e = msg.index(end)
                    value = user_msg[idx_s:idx_e].strip()
                    if value:
                        self.add_fact(prefix + value + end)

    def get_status(self) -> str:
        """Для отладки"""
        return f"[память: {len(self.episodes)} эпизодов, {len(self.facts)} фактов]"