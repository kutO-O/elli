"""
Семантическая память — факты и знания.

Хранит:
- Факты о собеседнике ("Иван любит джаз")
- Факты о мире ("Земля круглая")
- Выученные ассоциации
"""

import json
import os
import time
from datetime import datetime

import numpy as np


class SemanticMemory:
    """
    Память фактов и знаний.
    """
    
    SAVE_PATH = os.path.join("data", "memory", "facts.json")
    
    def __init__(self):
        self.facts = {}  # ключ → {text, source, strength, created, updated}
        self._load()
    
    def store(self, fact, source="разговор", importance=0.5):
        """
        Запомнить факт.
        
        Args:
            fact: Текст факта
            source: Откуда узнала
            importance: Важность (0-1)
        """
        key = fact.lower().strip()
        
        if key in self.facts:
            # Уже знаем — усиливаем
            self.facts[key]["strength"] = min(1.0, self.facts[key]["strength"] + 0.2)
            self.facts[key]["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            # Новый факт
            self.facts[key] = {
                "text": fact,
                "source": source,
                "strength": min(1.0, 0.5 + importance * 0.5),
                "importance": importance,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        
        self._save()
    
    def recall(self, query, top_k=5):
        """
        Вспомнить факты по теме.
        
        Args:
            query: Текст запроса
            top_k: Сколько фактов вернуть
            
        Returns:
            list: Список фактов (строки)
        """
        if not self.facts:
            return []
        
        query_words = set(query.lower().split())
        
        scored = []
        for key, fact in self.facts.items():
            fact_words = set(key.split())
            overlap = len(query_words & fact_words)
            
            if overlap > 0:
                score = overlap * fact["strength"]
                scored.append((fact["text"], score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, score in scored[:top_k]]
    
    def extract_facts(self, user_text):
        """
        Автоматически извлечь факты из текста пользователя.
        
        Args:
            user_text: Что сказал пользователь
        """
        text = user_text.lower()
        
        # Шаблоны для извлечения
        patterns = [
            ("меня зовут ", "Имя собеседника: "),
            ("я люблю ", "Собеседник любит "),
            ("мне нравится ", "Собеседнику нравится "),
            ("я работаю ", "Собеседник работает "),
            ("я живу в ", "Собеседник живёт в "),
            ("я не люблю ", "Собеседник не любит "),
            ("я ненавижу ", "Собеседник ненавидит "),
            ("мой любимый ", "Любимый у собеседника — "),
            ("моя любимая ", "Любимая у собеседника — "),
            ("я хочу ", "Собеседник хочет "),
            ("я мечтаю ", "Собеседник мечтает "),
            ("я боюсь ", "Собеседник боится "),
            ("я умею ", "Собеседник умеет "),
        ]
        
        for trigger, prefix in patterns:
            if trigger in text:
                idx = text.index(trigger) + len(trigger)
                value = user_text[idx:].split(".")[0].split(",")[0].split("!")[0].strip()
                if value and len(value) < 100:
                    self.store(prefix + value, source="разговор", importance=0.7)
    
    def get_all(self):
        """Все факты"""
        return [f["text"] for f in self.facts.values() if f["strength"] > 0.1]
    
    def get_stats(self):
        """Статистика"""
        return {
            "count": len(self.facts),
            "strong": sum(1 for f in self.facts.values() if f["strength"] > 0.5),
        }
    
    def decay_all(self):
        """Ослабить старые факты"""
        to_delete = []
        for key, fact in self.facts.items():
            fact["strength"] *= 0.999  # Очень медленное забывание
            if fact["strength"] < 0.05:
                to_delete.append(key)
        
        for key in to_delete:
            del self.facts[key]
        
        if to_delete:
            self._save()
        
        return len(to_delete)
    
    def _save(self):
        """Сохранить"""
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)
        try:
            with open(self.SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(self.facts, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _load(self):
        """Загрузить"""
        if not os.path.exists(self.SAVE_PATH):
            return
        try:
            with open(self.SAVE_PATH, "r", encoding="utf-8") as f:
                self.facts = json.load(f)
        except Exception:
            self.facts = {}