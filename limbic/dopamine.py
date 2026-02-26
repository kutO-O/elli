"""
Дофаминовая система — награда и обучение.

Ключевой принцип: Reward Prediction Error (RPE).
Дофамин = полученная награда - ожидаемая награда.

Если получил больше чем ожидал → дофамин (запоминай!)
Если получил меньше чем ожидал → анти-дофамин (разочарование)
Если получил сколько ожидал → ничего (привычка)
"""

import numpy as np


class DopamineSystem:
    """
    Дофаминовая система.
    Отслеживает ожидания и генерирует сигнал reward prediction error.
    """
    
    def __init__(self):
        # Ожидаемая награда (учится со временем)
        self.expected_reward = 0.0
        
        # Текущий уровень дофамина (-1 до +1)
        # +1 = сильный приятный сюрприз
        # -1 = сильное разочарование  
        # 0 = ожидаемо
        self.level = 0.0
        
        # Скорость обучения ожиданий
        self.learning_rate = 0.1
        
        # Затухание дофамина (возвращается к нулю)
        self.decay = 0.3
        
        # Базовый уровень (настроение влияет)
        self.baseline = 0.0
        
        # История
        self.level_history = []
        self.rpe_history = []  # Reward Prediction Error
    
    def process(self, actual_reward):
        """
        Обработать полученную награду.
        
        Args:
            actual_reward: Фактическая награда (-1 до +1)
                          Обычно = valence из амигдалы
        
        Returns:
            dict: {level, rpe, expected}
        """
        # Reward Prediction Error
        rpe = actual_reward - self.expected_reward
        
        # Обновляем ожидания (учимся)
        self.expected_reward += self.learning_rate * rpe
        
        # Ограничиваем ожидания
        self.expected_reward = float(np.clip(self.expected_reward, -1, 1))
        
        # Затухание текущего уровня
        self.level *= (1 - self.decay)
        
        # Добавляем новый RPE
        self.level += rpe * 0.5
        
        # Ограничиваем
        self.level = float(np.clip(self.level, -1, 1))
        
        # История
        self.level_history.append(self.level)
        self.rpe_history.append(rpe)
        
        return {
            "level": round(self.level, 3),
            "rpe": round(rpe, 3),
            "expected": round(self.expected_reward, 3),
        }
    
    def get_learning_boost(self):
        """
        Множитель обучения.
        Высокий дофамин → лучше запоминаем.
        
        Returns:
            float: Множитель (0.5 до 3.0)
        """
        # Положительный дофамин → усиливаем обучение
        # Отрицательный → тоже усиливаем (запоминаем плохое!)
        # Нейтральный → обычное обучение
        boost = 1.0 + abs(self.level) * 2.0
        return float(np.clip(boost, 0.5, 3.0))
    
    def reset(self):
        """Сброс (не ожиданий!)"""
        self.level = 0.0
        self.level_history = []
        self.rpe_history = []
    
    def get_status(self):
        """Статус"""
        return {
            "level": round(self.level, 3),
            "expected_reward": round(self.expected_reward, 3),
            "learning_boost": round(self.get_learning_boost(), 2),
        }