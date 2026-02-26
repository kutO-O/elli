"""
LIF (Leaky Integrate-and-Fire) нейрон.
Самый простой спайковый нейрон. Основа всего.

Как работает:
1. Получает входной ток (input current)
2. Мембранный потенциал растёт
3. Достигает порога → СПАЙК
4. Сбрасывается
5. "Утекает" к состоянию покоя (leak)
"""

import numpy as np
from config import DT


class LIFNeuron:
    """
    Leaky Integrate-and-Fire нейрон.
    
    Параметры:
        tau_m: Постоянная времени мембраны (мс). Больше = медленнее утечка.
        v_rest: Потенциал покоя (мВ)
        v_threshold: Порог спайка (мВ)
        v_reset: Потенциал сброса после спайка (мВ)
    """
    
    def __init__(
        self,
        tau_m=20.0,         # Типичное значение: 10-30 мс
        v_rest=-65.0,       # Покой: -65 мВ
        v_threshold=-50.0,  # Порог: -50 мВ
        v_reset=-65.0,      # Сброс: -65 мВ
    ):
        # Параметры
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        # Состояние
        self.v = v_rest  # Текущий потенциал
        self.spike = False  # Спайк на этом шаге?
        
        # Счётчик времени
        self.time_step = 0
        
        # История спайков (время каждого спайка)
        self.spike_times = []
    
    def step(self, input_current):
        """
        Один шаг симуляции (DT миллисекунд).
        
        Формула: dV/dt = (-(V - V_rest) + I) / tau_m
        
        I — входной ток в мВ (уже масштабированный).
        Для простоты: I = 1.0 значит "слабый вход",
                      I = 20.0 значит "сильный вход".
        
        Args:
            input_current: Входной ток (мВ)
            
        Returns:
            bool: True если был спайк
        """
        self.time_step += 1
        
        # Утечка к покою + входной ток
        dv = (-(self.v - self.v_rest) + input_current) / self.tau_m
        self.v += dv * DT
        
        # Проверка порога
        if self.v >= self.v_threshold:
            self.spike = True
            self.v = self.v_reset
            self.spike_times.append(self.time_step)
            return True
        else:
            self.spike = False
            return False
    
    def reset(self):
        """Сброс нейрона в начальное состояние"""
        self.v = self.v_rest
        self.spike = False
        self.time_step = 0
        self.spike_times = []
    
    def get_firing_rate(self, window_ms=1000.0):
        """
        Частота спайков (Гц) за последние window_ms миллисекунд.
        """
        if not self.spike_times:
            return 0.0
        
        window_steps = int(window_ms / DT)
        recent = [t for t in self.spike_times if t >= self.time_step - window_steps]
        
        return len(recent) / (window_ms / 1000.0)  # В Гц


class LIFPopulation:
    """
    Популяция LIF нейронов.
    Используется для представления концепций/паттернов.
    """
    
    def __init__(self, n_neurons, **neuron_params):
        """
        Args:
            n_neurons: Количество нейронов в популяции
            **neuron_params: Параметры для LIF нейронов
        """
        self.neurons = [LIFNeuron(**neuron_params) for _ in range(n_neurons)]
        self.n = n_neurons
    
    def step(self, input_currents):
        """
        Шаг для всей популяции.
        
        Args:
            input_currents: Массив токов для каждого нейрона (или одно число)
        """
        # Если один ток для всех
        if np.isscalar(input_currents):
            input_currents = [input_currents] * self.n
        
        spikes = []
        for neuron, current in zip(self.neurons, input_currents):
            spikes.append(neuron.step(current))
        
        return np.array(spikes)
    
    def get_activity(self):
        """Доля активных нейронов (0.0 - 1.0)"""
        return sum(n.spike for n in self.neurons) / self.n
    
    def get_mean_potential(self):
        """Средний мембранный потенциал"""
        return np.mean([n.v for n in self.neurons])
    
    def reset(self):
        """Сброс всей популяции"""
        for neuron in self.neurons:
            neuron.reset()