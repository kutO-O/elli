"""
Амигдала — эмоциональная оценка.
Оценивает текст как позитивный/негативный.
Обучается через опыт.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.encoding import TextEncoder


class Amygdala:
    """
    Амигдала — центр эмоциональной оценки.
    
    Каждое слово получает эмоциональный вес (от -1 до +1).
    Текст оценивается как среднее весов известных слов.
    Обучается: новые слова получают веса через обратную связь.
    """
    
    def __init__(self, input_size=100):
        self.input_size = input_size
        self.encoder = TextEncoder(n_neurons=input_size)
        
        # Эмоциональный словарь: слово → вес (-1 до +1)
        self.word_valence = {}
        
        # Текущее состояние
        self.valence = 0.0      # -1 (плохо) до +1 (хорошо)
        self.arousal = 0.0      # 0 (спокойно) до 1 (возбуждённо)
        
        # Инерция
        self.inertia = 0.85
        
        # Затухание к нейтральному
        self.decay_rate = 0.005
        
        # История
        self.valence_history = []
        self.arousal_history = []
        
        # Врождённые слова
        self._pretrain()
    
    def _pretrain(self):
        """Врождённые эмоциональные ассоциации"""
        
        innate = {
            # Позитивные (русский)
            "привет": 0.6, "хорошо": 0.7, "отлично": 0.8,
            "люблю": 0.9, "нравится": 0.7, "спасибо": 0.8,
            "красиво": 0.6, "прекрасно": 0.8, "рад": 0.7,
            "рада": 0.7, "счастье": 0.9, "весело": 0.7,
            "смешно": 0.6, "супер": 0.7, "молодец": 0.8,
            "друг": 0.7, "дружба": 0.7, "тепло": 0.5,
            "солнце": 0.4, "радость": 0.8, "ценю": 0.8,
            "обнимаю": 0.8, "скучаю": 0.5, "интересно": 0.5,
            
            # Позитивные (английский)
            "hello": 0.5, "good": 0.6, "great": 0.7,
            "love": 0.9, "thanks": 0.7, "happy": 0.8,
            "nice": 0.6, "cool": 0.6, "awesome": 0.8,
            
            # Негативные (русский)
            "плохо": -0.7, "ужас": -0.8, "ненавижу": -0.9,
            "злюсь": -0.7, "грустно": -0.6, "тупой": -0.8,
            "тупая": -0.8, "дура": -0.9, "идиот": -0.9,
            "заткнись": -0.9, "скучно": -0.4, "надоело": -0.5,
            "бесит": -0.7, "раздражает": -0.6, "больно": -0.7,
            "страшно": -0.6, "обидно": -0.7, "противно": -0.7,
            "мерзко": -0.8, "ужасно": -0.8, "уйди": -0.7,
            "ненависть": -0.9, "злость": -0.7,
            
            # Негативные (английский)
            "bad": -0.6, "hate": -0.9, "stupid": -0.8,
            "shut": -0.7, "angry": -0.7, "sad": -0.6,
            "terrible": -0.8, "awful": -0.8, "ugly": -0.6,
        }
        
        self.word_valence = innate.copy()
    
    def process(self, text, n_steps=50):
        """
        Обработать текст и получить эмоциональную оценку.
        
        Args:
            text: Входной текст
            n_steps: Не используется (для совместимости)
        
        Returns:
            dict: {valence, arousal, emotion}
        """
        words = text.lower().split()
        
        if not words:
            return self._make_result()
        
        # Собираем оценки известных слов
        known_valences = []
        for word in words:
            # Убираем знаки препинания
            clean = word.strip(".,!?;:\"'()[]{}…")
            if clean in self.word_valence:
                known_valences.append(self.word_valence[clean])
        
        if known_valences:
            # Среднее эмоциональных слов
            raw_valence = float(np.mean(known_valences))
            
            # Возбуждение: чем сильнее эмоция, тем выше
            raw_arousal = float(np.mean([abs(v) for v in known_valences]))
            
            # Усиливаем если много эмоциональных слов
            emotion_ratio = len(known_valences) / len(words)
            raw_valence *= (0.5 + emotion_ratio)
            raw_arousal *= (0.5 + emotion_ratio)
        else:
            # Неизвестные слова → слабый сдвиг к нейтральному
            raw_valence = 0.0
            raw_arousal = 0.1
        
        # Ограничиваем
        raw_valence = float(np.clip(raw_valence, -1, 1))
        raw_arousal = float(np.clip(raw_arousal, 0, 1))
        
        # Затухание к нейтральному
        self.valence *= (1 - self.decay_rate)
        self.arousal *= (1 - self.decay_rate)
        
        # Инерция
        self.valence = self.inertia * self.valence + (1 - self.inertia) * raw_valence
        self.arousal = self.inertia * self.arousal + (1 - self.inertia) * raw_arousal
        
        # Ограничиваем
        self.valence = float(np.clip(self.valence, -1, 1))
        self.arousal = float(np.clip(self.arousal, 0, 1))
        
        # История
        self.valence_history.append(self.valence)
        self.arousal_history.append(self.arousal)
        
        return self._make_result()
    
    def _make_result(self):
        """Сформировать результат"""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "emotion": self._get_emotion_name()
        }
    
    def _get_emotion_name(self):
        """Название эмоции"""
        v, a = self.valence, self.arousal
        
        if v > 0.3:
            if a > 0.5:
                return "радость"
            else:
                return "спокойствие"
        elif v < -0.3:
            if a > 0.5:
                return "тревога"
            else:
                return "грусть"
        else:
            if a > 0.5:
                return "возбуждение"
            else:
                return "нейтральность"
    
    def learn(self, text, target_valence, n_iterations=10):
        """
        Обучить: связать слова из текста с эмоцией.
        
        Args:
            text: Текст
            target_valence: Целевая валентность (-1 до +1)
            n_iterations: Сила обучения (больше = сильнее)
        """
        words = text.lower().split()
        lr = 0.05 * n_iterations
        
        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}…")
            if not clean:
                continue
            
            if clean in self.word_valence:
                # Сдвигаем существующий вес к цели
                current = self.word_valence[clean]
                self.word_valence[clean] = current + lr * (target_valence - current)
            else:
                # Новое слово — записываем
                self.word_valence[clean] = target_valence * lr
            
            # Ограничиваем
            self.word_valence[clean] = float(np.clip(
                self.word_valence[clean], -1, 1
            ))
    
    def feedback(self, text, is_positive):
        """
        Обратная связь от человека.
        
        Args:
            text: Что было сказано
            is_positive: True если хорошо, False если плохо
        """
        target = 0.7 if is_positive else -0.7
        self.learn(text, target, n_iterations=5)
    
    def get_known_words_count(self):
        """Сколько слов знает"""
        return len(self.word_valence)
    
    def reset(self):
        """Сброс состояния (не словаря!)"""
        self.valence = 0.0
        self.arousal = 0.0
        self.valence_history = []
        self.arousal_history = []
    
    def get_status(self):
        """Статус для отладки"""
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "emotion": self._get_emotion_name(),
            "known_words": self.get_known_words_count(),
        }