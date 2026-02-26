"""
Izhikevich нейрон.
Простая модель, но воспроизводит 20+ типов поведения реальных нейронов.

Два уравнения:
    dv/dt = 0.04*v² + 5*v + 140 - u + I
    du/dt = a*(b*v - u)

Четыре параметра (a, b, c, d) определяют тип нейрона.

После спайка:
    v = c
    u = u + d
"""

import numpy as np
from config import DT


# Пресеты типов нейронов
# Формат: (a, b, c, d, описание)
NEURON_TYPES = {
    # Возбуждающие (excitatory)
    "regular_spiking":     (0.02, 0.2, -65.0, 8.0,   "Обычный нейрон коры"),
    "intrinsic_bursting":  (0.02, 0.2, -55.0, 4.0,   "Вспышки спайков"),
    "chattering":          (0.02, 0.2, -50.0, 2.0,   "Быстрые вспышки"),
    "tonic_spiking":       (0.02, 0.2, -65.0, 6.0,   "Постоянные спайки"),
    
    # Тормозные (inhibitory)
    "fast_spiking":        (0.1,  0.2, -65.0, 2.0,   "Быстрый тормозной"),
    "low_threshold":       (0.02, 0.25, -65.0, 2.0,  "Низкий порог"),
    
    # Специальные
    "thalamic":            (0.02, 0.25, -65.0, 0.05,  "Таламус (сознание)"),
    "resonator":           (0.1,  0.26, -65.0, 2.0,   "Резонатор"),
}


class IzhikevichNeuron:
    """
    Izhikevich нейрон.
    
    Параметры:
        neuron_type: Тип из NEURON_TYPES или "custom"
        a: Скорость восстановления (меньше = медленнее)
        b: Чувствительность восстановления
        c: Потенциал сброса после спайка (мВ)
        d: Прибавка восстановления после спайка
    """
    
    def __init__(self, neuron_type="regular_spiking", a=None, b=None, c=None, d=None):
        # Загружаем пресет
        if neuron_type in NEURON_TYPES:
            preset_a, preset_b, preset_c, preset_d, self.description = NEURON_TYPES[neuron_type]
        else:
            preset_a, preset_b, preset_c, preset_d = 0.02, 0.2, -65.0, 8.0
            self.description = "Custom"
        
        # Параметры (можно переопределить)
        self.a = a if a is not None else preset_a
        self.b = b if b is not None else preset_b
        self.c = c if c is not None else preset_c
        self.d = d if d is not None else preset_d
        
        self.neuron_type = neuron_type
        
        # Состояние
        self.v = self.c       # Мембранный потенциал
        self.u = self.b * self.v  # Переменная восстановления
        self.spike = False
        
        # Счётчик и история
        self.time_step = 0
        self.spike_times = []
    
    def step(self, input_current):
        """
        Один шаг симуляции.
        
        Args:
            input_current: Входной ток (пА). 
                          Типичные значения: 5-15 для обычных спайков.
        Returns:
            bool: True если был спайк
        """
        self.time_step += 1
        dt = DT
        
        # Уравнения Izhikevich (используем два полушага для стабильности)
        half_dt = dt * 0.5
        
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input_current)
        self.v += half_dt * (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input_current)
        self.u += dt * self.a * (self.b * self.v - self.u)
        
        # Проверка спайка
        if self.v >= 30.0:
            self.spike = True
            self.v = self.c
            self.u += self.d
            self.spike_times.append(self.time_step)
            return True
        else:
            self.spike = False
            return False
    
    def reset(self):
        """Сброс в начальное состояние"""
        self.v = self.c
        self.u = self.b * self.v
        self.spike = False
        self.time_step = 0
        self.spike_times = []
    
    def get_firing_rate(self, window_ms=1000.0):
        """Частота спайков (Гц)"""
        if not self.spike_times:
            return 0.0
        
        window_steps = int(window_ms / DT)
        recent = [t for t in self.spike_times if t >= self.time_step - window_steps]
        
        return len(recent) / (window_ms / 1000.0)


class IzhikevichPopulation:
    """
    Популяция Izhikevich нейронов одного типа.
    """
    
    def __init__(self, n_neurons, neuron_type="regular_spiking", noise=0.0):
        """
        Args:
            n_neurons: Количество нейронов
            neuron_type: Тип из NEURON_TYPES
            noise: Уровень шума (0 = нет, 1 = сильный)
        """
        self.n = n_neurons
        self.noise = noise
        self.neurons = [IzhikevichNeuron(neuron_type=neuron_type) for _ in range(n_neurons)]
        
        # Немного разнообразия в параметрах (как в реальном мозге)
        if noise > 0:
            for neuron in self.neurons:
                neuron.a *= (1 + np.random.uniform(-noise * 0.1, noise * 0.1))
                neuron.b *= (1 + np.random.uniform(-noise * 0.1, noise * 0.1))
                neuron.c += np.random.uniform(-noise * 2, noise * 2)
                neuron.d *= (1 + np.random.uniform(-noise * 0.1, noise * 0.1))
    
    def step(self, input_currents):
        """
        Шаг для всей популяции.
        
        Args:
            input_currents: Число (для всех) или массив
        """
        if np.isscalar(input_currents):
            input_currents = [input_currents] * self.n
        
        spikes = []
        for neuron, current in zip(self.neurons, input_currents):
            # Добавляем шум
            noisy_current = current + np.random.normal(0, self.noise) if self.noise > 0 else current
            spikes.append(neuron.step(noisy_current))
        
        return np.array(spikes)
    
    def get_activity(self):
        """Доля активных нейронов (0.0 - 1.0)"""
        return sum(n.spike for n in self.neurons) / self.n
    
    def get_mean_potential(self):
        """Средний потенциал"""
        return np.mean([n.v for n in self.neurons])
    
    def get_spike_count(self):
        """Общее количество спайков"""
        return sum(len(n.spike_times) for n in self.neurons)
    
    def reset(self):
        """Сброс"""
        for neuron in self.neurons:
            neuron.reset()