"""
Кодирование информации в спайки и обратно.

Три способа:
1. Rate coding: значение → частота спайков
2. Temporal coding: значение → время первого спайка
3. Population coding: значение → какие нейроны активны
"""

import numpy as np
from config import DT


class RateEncoder:
    """
    Rate coding: число → частота спайков.
    Больше значение → чаще спайки.
    """
    
    def __init__(self, max_rate=100.0):
        """
        Args:
            max_rate: Максимальная частота (Гц) при значении 1.0
        """
        self.max_rate = max_rate
    
    def encode(self, value, duration_ms=100.0):
        """
        Превратить число в последовательность спайков.
        
        Args:
            value: Число от 0 до 1
            duration_ms: Длительность кодирования (мс)
        
        Returns:
            numpy array: Массив спайков (True/False) для каждого шага
        """
        value = max(0.0, min(1.0, value))
        
        n_steps = int(duration_ms / DT)
        spikes = np.zeros(n_steps, dtype=bool)
        
        # Вероятность спайка на каждом шаге
        rate_hz = value * self.max_rate
        prob_per_step = rate_hz * DT / 1000.0
        
        # Генерируем спайки с нужной вероятностью
        spikes = np.random.random(n_steps) < prob_per_step
        
        return spikes
    
    def decode(self, spikes, duration_ms=None):
        """
        Превратить спайки обратно в число.
        
        Args:
            spikes: Массив спайков (bool)
            duration_ms: Длительность (если None — считаем из длины)
        
        Returns:
            float: Значение от 0 до 1
        """
        if duration_ms is None:
            duration_ms = len(spikes) * DT
        
        if duration_ms == 0:
            return 0.0
        
        # Считаем частоту
        spike_count = np.sum(spikes)
        rate_hz = spike_count / (duration_ms / 1000.0)
        
        # Нормализуем
        value = rate_hz / self.max_rate
        return max(0.0, min(1.0, value))


class PopulationEncoder:
    """
    Population coding: число → паттерн активности популяции.
    
    Каждый нейрон "настроен" на определённое значение.
    Нейрон спайкает сильнее когда входное значение 
    близко к его предпочтительному значению.
    """
    
    def __init__(self, n_neurons=20, value_range=(0.0, 1.0)):
        """
        Args:
            n_neurons: Количество нейронов в популяции
            value_range: Диапазон кодируемых значений
        """
        self.n = n_neurons
        self.v_min, self.v_max = value_range
        
        # Предпочтительные значения нейронов (равномерно)
        self.preferred = np.linspace(self.v_min, self.v_max, n_neurons)
        
        # Ширина "кривой настройки" (tuning curve)
        self.sigma = (self.v_max - self.v_min) / (n_neurons * 0.5)
    
    def encode(self, value):
        """
        Превратить число в уровни активации популяции.
        
        Args:
            value: Число из value_range
        
        Returns:
            numpy array: Активации каждого нейрона (0-1)
        """
        # Гауссова кривая настройки
        activations = np.exp(-0.5 * ((value - self.preferred) / self.sigma) ** 2)
        return activations
    
    def decode(self, activations):
        """
        Превратить активации обратно в число.
        Средневзвешенное по предпочтительным значениям.
        
        Args:
            activations: Массив активаций нейронов
        
        Returns:
            float: Декодированное значение
        """
        total = np.sum(activations)
        if total == 0:
            return (self.v_min + self.v_max) / 2
        
        return np.sum(activations * self.preferred) / total


class TextEncoder:
    """
    Кодирование текста в спайковые паттерны.
    Простой вариант: каждое слово → хэш → паттерн активации.
    """
    
    def __init__(self, n_neurons=100):
        """
        Args:
            n_neurons: Размер паттерна (количество нейронов)
        """
        self.n = n_neurons
    
    def encode_word(self, word):
        """
        Одно слово → паттерн активации.
        
        Использует хэш для создания уникального 
        но воспроизводимого паттерна.
        
        Args:
            word: Строка
        
        Returns:
            numpy array: Бинарный паттерн (0/1), ~10% единиц
        """
        # Используем хэш слова как seed для генератора
        seed = hash(word.lower().strip()) % (2**31)
        rng = np.random.RandomState(seed)
        
        # Разреженный паттерн (~10% активных нейронов)
        pattern = np.zeros(self.n, dtype=float)
        active_indices = rng.choice(self.n, size=max(1, self.n // 10), replace=False)
        pattern[active_indices] = 1.0
        
        return pattern
    
    def encode_text(self, text):
        """
        Текст → комбинированный паттерн.
        Паттерны слов складываются и нормализуются.
        
        Args:
            text: Строка текста
        
        Returns:
            numpy array: Паттерн активации (0-1)
        """
        words = text.lower().split()
        if not words:
            return np.zeros(self.n)
        
        # Суммируем паттерны всех слов
        combined = np.zeros(self.n)
        for word in words:
            combined += self.encode_word(word)
        
        # Нормализуем
        max_val = np.max(combined)
        if max_val > 0:
            combined /= max_val
        
        return combined
    
    def similarity(self, text1, text2):
        """
        Похожесть двух текстов (0-1).
        
        Args:
            text1, text2: Строки
        
        Returns:
            float: Косинусное сходство (0-1)
        """
        p1 = self.encode_text(text1)
        p2 = self.encode_text(text2)
        
        dot = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))