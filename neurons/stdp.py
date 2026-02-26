"""
STDP (Spike-Timing Dependent Plasticity).
Обучение через время спайков. Основа обучения мозга.

Правило:
  pre спайкает ДО post  → усиление связи (LTP)
  pre спайкает ПОСЛЕ post → ослабление связи (LTD)
  
Формула:
  Δw = A_plus * exp(-Δt / tau_plus)   если Δt > 0 (pre до post)
  Δw = -A_minus * exp(Δt / tau_minus)  если Δt < 0 (pre после post)
"""

import numpy as np
from config import DT


class Synapse:
    """
    Один синапс между двумя нейронами.
    Имеет вес (силу связи) и обучается через STDP.
    """
    
    def __init__(self, weight=0.5, w_min=0.0, w_max=1.0, delay=1.0):
        """
        Args:
            weight: Начальный вес (0-1)
            w_min: Минимальный вес
            w_max: Максимальный вес
            delay: Задержка передачи (мс)
        """
        self.weight = weight
        self.w_min = w_min
        self.w_max = w_max
        self.delay = delay
    
    def clip(self):
        """Ограничить вес в пределах [w_min, w_max]"""
        self.weight = max(self.w_min, min(self.w_max, self.weight))


class STDPRule:
    """
    Правило обучения STDP.
    Управляет изменением весов синапсов.
    """
    
    def __init__(
        self,
        a_plus=0.01,    # Сила усиления (LTP)
        a_minus=0.012,  # Сила ослабления (LTD) — чуть сильнее
        tau_plus=20.0,  # Окно усиления (мс)
        tau_minus=20.0, # Окно ослабления (мс)
    ):
        """
        a_minus > a_plus — это важно!
        Без этого все связи станут максимальными.
        Баланс LTP/LTD — ключ к стабильности.
        """
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
    
    def compute_dw(self, dt):
        """
        Вычислить изменение веса.
        
        Args:
            dt: Разница во времени (t_post - t_pre) в мс.
                dt > 0 → pre до post → усиление
                dt < 0 → pre после post → ослабление
        
        Returns:
            float: Изменение веса
        """
        if dt > 0:
            # Pre до post → усиление (LTP)
            return self.a_plus * np.exp(-dt / self.tau_plus)
        elif dt < 0:
            # Pre после post → ослабление (LTD)
            return -self.a_minus * np.exp(dt / self.tau_minus)
        else:
            return 0.0


class SynapticNetwork:
    """
    Сеть синапсов с STDP обучением.
    Соединяет два слоя нейронов.
    """
    
    def __init__(self, n_pre, n_post, connectivity=1.0, initial_weight=0.5):
        """
        Args:
            n_pre: Количество пресинаптических нейронов
            n_post: Количество постсинаптических нейронов
            connectivity: Доля связей (1.0 = все со всеми)
            initial_weight: Начальный вес
        """
        self.n_pre = n_pre
        self.n_post = n_post
        
        # Матрица весов
        self.weights = np.full((n_pre, n_post), initial_weight)
        
        # Маска связей (какие существуют)
        self.mask = np.random.random((n_pre, n_post)) < connectivity
        self.weights *= self.mask
        
        # STDP правило
        self.stdp = STDPRule()
        
        # Следы активности (traces) для эффективного STDP
        # Trace растёт при спайке, затухает экспоненциально
        self.pre_trace = np.zeros(n_pre)
        self.post_trace = np.zeros(n_post)
        
        # Скорость затухания следов
        self.trace_decay_pre = np.exp(-DT / self.stdp.tau_plus)
        self.trace_decay_post = np.exp(-DT / self.stdp.tau_minus)
    
    def step(self, pre_spikes, post_spikes):
        """
        Один шаг: передать сигнал и обучить.
        
        Args:
            pre_spikes: Массив спайков пресинаптических нейронов (bool)
            post_spikes: Массив спайков постсинаптических нейронов (bool)
            
        Returns:
            numpy array: Входные токи для постсинаптических нейронов
        """
        pre_spikes = np.asarray(pre_spikes, dtype=float)
        post_spikes = np.asarray(post_spikes, dtype=float)
        
        # 1. Затухание следов
        self.pre_trace *= self.trace_decay_pre
        self.post_trace *= self.trace_decay_post
        
        # 2. Обновление следов при спайках
        self.pre_trace += pre_spikes
        self.post_trace += post_spikes
        
        # 3. STDP обучение
        # Если post спайкает → усиление связей от недавно активных pre
        if np.any(post_spikes > 0):
            for j in range(self.n_post):
                if post_spikes[j]:
                    # Усиление пропорционально pre_trace
                    self.weights[:, j] += self.stdp.a_plus * self.pre_trace * self.mask[:, j]
        
        # Если pre спайкает → ослабление связей к недавно активным post
        if np.any(pre_spikes > 0):
            for i in range(self.n_pre):
                if pre_spikes[i]:
                    # Ослабление пропорционально post_trace
                    self.weights[i, :] -= self.stdp.a_minus * self.post_trace * self.mask[i, :]
        
        # 4. Ограничение весов
        self.weights = np.clip(self.weights, 0.0, 1.0)
        
        # 5. Вычисление входных токов для post нейронов
        # Ток = сумма (вес * спайк) по всем пресинаптическим
        currents = pre_spikes @ self.weights
        
        return currents
    
    def get_mean_weight(self):
        """Средний вес активных связей"""
        active = self.weights[self.mask]
        if len(active) == 0:
            return 0.0
        return np.mean(active)
    
    def get_weight_stats(self):
        """Статистика весов"""
        active = self.weights[self.mask]
        if len(active) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(active)),
            "std": float(np.std(active)),
            "min": float(np.min(active)),
            "max": float(np.max(active)),
        }
    
    def reset_traces(self):
        """Сброс следов (не весов!)"""
        self.pre_trace = np.zeros(self.n_pre)
        self.post_trace = np.zeros(self.n_post)