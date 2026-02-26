"""
Эпизодическая память — "что было".

Каждый разговор записывается как эпизод:
- Когда было
- О чём говорили (сжато)
- Какие эмоции были
- Насколько важно

Важные эпизоды помнятся дольше.
Неважные — забываются.
"""

import json
import os
import time
from datetime import datetime

import numpy as np


class Episode:
    """Один эпизод (воспоминание)"""
    
    def __init__(self, summary, messages, emotion_valence, emotion_arousal):
        self.timestamp = time.time()
        self.date = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.summary = summary          # Краткое содержание
        self.messages = messages[-10:]   # Последние 10 сообщений
        self.valence = emotion_valence   # Эмоциональная окраска
        self.arousal = emotion_arousal   # Возбуждение
        
        # Важность: сильные эмоции → важнее
        self.importance = min(1.0, abs(emotion_valence) * 0.6 + emotion_arousal * 0.3 + 0.1)
        
        # Сила воспоминания (слабеет со временем)
        self.strength = 1.0
        
        # Сколько раз вспоминали (replay усиливает)
        self.recall_count = 0
    
    def decay(self, hours_passed):
        """
        Забывание со временем.
        Важные воспоминания забываются медленнее.
        """
        # Кривая забывания Эббингауза (упрощённая)
        # strength = importance^(time / rate)
        decay_rate = 24.0 / (self.importance + 0.1)  # Важные = медленнее
        
        # Каждое вспоминание усиливает
        recall_bonus = self.recall_count * 0.1
        
        self.strength = min(1.0, 
            (self.importance + recall_bonus) * np.exp(-hours_passed / (decay_rate * 24))
        )
        
        return self.strength > 0.05  # True если ещё помнит
    
    def recall(self):
        """Вспомнить эпизод (усиливает память)"""
        self.recall_count += 1
        self.strength = min(1.0, self.strength + 0.2)
    
    def to_dict(self):
        """Сериализация"""
        return {
            "timestamp": self.timestamp,
            "date": self.date,
            "summary": self.summary,
            "messages": self.messages,
            "valence": self.valence,
            "arousal": self.arousal,
            "importance": round(self.importance, 3),
            "strength": round(self.strength, 3),
            "recall_count": self.recall_count,
        }
    
    @staticmethod
    def from_dict(data):
        """Десериализация"""
        ep = Episode(
            summary=data["summary"],
            messages=data.get("messages", []),
            emotion_valence=data["valence"],
            emotion_arousal=data.get("arousal", 0.5),
        )
        ep.timestamp = data["timestamp"]
        ep.date = data["date"]
        ep.importance = data["importance"]
        ep.strength = data["strength"]
        ep.recall_count = data.get("recall_count", 0)
        return ep


class EpisodicMemory:
    """
    Эпизодическая память.
    Хранит воспоминания о событиях.
    """
    
    SAVE_PATH = os.path.join("data", "memory", "episodes.json")
    
    def __init__(self, max_episodes=500):
        self.episodes = []
        self.max_episodes = max_episodes
        self._load()
    
    def store(self, summary, messages, valence, arousal):
        """
        Сохранить новый эпизод.
        
        Args:
            summary: Краткое описание
            messages: Список сообщений [{role, content}, ...]
            valence: Эмоциональная валентность (-1 до +1)
            arousal: Возбуждение (0 до 1)
        """
        episode = Episode(summary, messages, valence, arousal)
        self.episodes.append(episode)
        
        # Удаляем самые слабые если переполнение
        if len(self.episodes) > self.max_episodes:
            self.episodes.sort(key=lambda e: e.strength, reverse=True)
            self.episodes = self.episodes[:self.max_episodes]
        
        self._save()
    
    def recall(self, query, top_k=3):
        """
        Вспомнить релевантные эпизоды.
        
        Args:
            query: Текст запроса
            top_k: Сколько эпизодов вернуть
            
        Returns:
            list: Список эпизодов (словари)
        """
        if not self.episodes:
            return []
        
        query_words = set(query.lower().split())
        
        scored = []
        for ep in self.episodes:
            # Совпадение слов в summary и messages
            ep_text = ep.summary.lower()
            for msg in ep.messages:
                ep_text += " " + msg.get("content", "").lower()
            
            ep_words = set(ep_text.split())
            overlap = len(query_words & ep_words)
            
            if overlap > 0:
                # Оценка = совпадение * сила * важность
                score = overlap * ep.strength * ep.importance
                scored.append((ep, score))
        
        # Сортируем по оценке
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Отмечаем что вспомнили (усиливает память)
        results = []
        for ep, score in scored[:top_k]:
            ep.recall()
            results.append(ep.to_dict())
        
        self._save()
        return results
    
    def get_recent(self, n=5):
        """Последние n эпизодов"""
        sorted_eps = sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)
        return [ep.to_dict() for ep in sorted_eps[:n]]
    
    def decay_all(self, hours=1):
        """
        Забывание: ослабить все воспоминания.
        Вызывается периодически.
        """
        alive = []
        for ep in self.episodes:
            hours_since = (time.time() - ep.timestamp) / 3600
            if ep.decay(hours_since):
                alive.append(ep)
        
        forgotten = len(self.episodes) - len(alive)
        self.episodes = alive
        
        if forgotten > 0:
            self._save()
        
        return forgotten
    
    def get_stats(self):
        """Статистика"""
        if not self.episodes:
            return {"count": 0, "avg_strength": 0, "avg_importance": 0}
        
        return {
            "count": len(self.episodes),
            "avg_strength": round(np.mean([e.strength for e in self.episodes]), 3),
            "avg_importance": round(np.mean([e.importance for e in self.episodes]), 3),
        }
    
    def _save(self):
        """Сохранить на диск"""
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        data = [ep.to_dict() for ep in self.episodes]
        try:
            with open(self.SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _load(self):
        """Загрузить с диска"""
        if not os.path.exists(self.SAVE_PATH):
            return
        try:
            with open(self.SAVE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.episodes = [Episode.from_dict(d) for d in data]
        except Exception:
            self.episodes = []