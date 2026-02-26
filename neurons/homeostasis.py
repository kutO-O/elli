"""
Гомеостаз.
Стабилизирует активность нейронов.

Если нейрон слишком активен → порог растёт → активность падает.
Если нейрон молчит → порог снижается → активность растёт.

Это как термостат: поддерживает нужную "температуру" активности.
"""

import numpy as np
from config import DT


class HomeostaticRegulator:
    """
    Регулятор гомеостаза для популяции нейронов.
    
    Следит за средней активностью и корректирует возбудимость.
    """
    
    def __init__(
        self,
        target_rate=5.0,    # Целевая частота спайков (Гц)
        tau=10000.0,        # Скорость адаптации (мс). Больше = медленнее.
        strength=0.1,       # Сила коррекции
    ):
        """
        Args:
            target_rate: Желаемая частота спайков (Гц)
            tau: Постоянная времени адаптации (мс)
            strength: Насколько сильно корректировать
        """
        self.target_rate = target_rate
        self.tau = tau
        self.strength = strength
        
        # Скользящая средняя активности
        self.activity_avg = target_rate
        
        # Множитель возбудимости (1.0 = норма)
        self.excitability = 1.0
        
        # Накопитель спайков для подсчёта частоты
        self.spike_count = 0
        self.time_window = 0
    
    def update(self, spike_count, n_neurons, dt_ms=None):
        """
        Обновить гомеостаз.
        
        Args:
            spike_count: Количество спайков в этом шаге
            n_neurons: Общее количество нейронов
            dt_ms: Длительность шага (мс). По умолчанию DT.
        """
        if dt_ms is None:
            dt_ms = DT
        
        # Считаем текущую частоту (Гц)
        # spike_count / n_neurons = доля активных
        # Переводим в Гц: доля * (1000 / dt_ms)
        current_rate = (spike_count / n_neurons) * (1000.0 / dt_ms)
        
        # Обновляем скользящую среднюю
        alpha = dt_ms / self.tau
        self.activity_avg += alpha * (current_rate - self.activity_avg)
        
        # Корректируем возбудимость
        error = self.target_rate - self.activity_avg
        self.excitability += self.strength * error * alpha
        
        # Ограничиваем (не может быть отрицательной или слишком большой)
        self.excitability = max(0.1, min(5.0, self.excitability))
    
    def scale_input(self, input_current):
        """
        Масштабировать входной ток с учётом гомеостаза.
        
        Args:
            input_current: Оригинальный ток (число или массив)
        Returns:
            Масштабированный ток
        """
        return input_current * self.excitability
    
    def get_status(self):
        """Статус для отладки"""
        return {
            "target_rate": self.target_rate,
            "current_avg": round(self.activity_avg, 2),
            "excitability": round(self.excitability, 3),
        }