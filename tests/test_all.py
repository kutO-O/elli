"""
Автотесты для всех модулей Элли.
Запуск: python tests/test_all.py
"""

import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurons.lif import LIFNeuron, LIFPopulation
import numpy as np


def test_lif_neuron():
    """Тест одиночного LIF нейрона"""
    print("Testing LIF Neuron...")
    
    neuron = LIFNeuron()
    
    # Тест 1: Без входа не должно быть спайков
    neuron.reset()
    spiked = False
    for _ in range(1000):
        if neuron.step(0.0):
            spiked = True
    assert not spiked, "Нейрон не должен спайкать без входного тока"
    print("  ✓ Без входа → нет спайков")
    
    # Тест 2: С постоянным током должен спайкать регулярно
    neuron.reset()
    spike_count = 0
    for _ in range(1000):
        if neuron.step(20.0):  # Ток 20 мВ
            spike_count += 1
    assert spike_count > 0, "Нейрон должен спайкать с постоянным током"
    print(f"  ✓ С током 20 мВ → {spike_count} спайков за 1000 шагов")
    
    # Тест 3: Больше тока → больше спайков
    neuron.reset()
    spike_count_high = 0
    for _ in range(1000):
        if neuron.step(30.0):
            spike_count_high += 1
    assert spike_count_high > spike_count, "Больше тока → больше спайков"
    print(f"  ✓ С током 30 мВ → {spike_count_high} спайков (больше)")
    
    # Тест 4: Частота спайков
    rate = neuron.get_firing_rate()
    assert rate > 0, "Частота спайков должна быть > 0"
    print(f"  ✓ Частота спайков: {rate:.1f} Гц")
    
    # Тест 5: Потенциал покоя
    neuron.reset()
    assert neuron.v == neuron.v_rest, "После сброса потенциал = покой"
    print(f"  ✓ Сброс работает: V = {neuron.v} мВ")
    
    print("LIF Neuron: OK\n")


def test_lif_population():
    """Тест популяции нейронов"""
    print("Testing LIF Population...")
    
    pop = LIFPopulation(n_neurons=100)
    
    # Тест 1: Без входа активность нулевая
    pop.reset()
    for _ in range(100):
        pop.step(0.0)
    activity = pop.get_activity()
    assert activity == 0.0, "Популяция не должна быть активна без входа"
    print("  ✓ Без входа → активность = 0")
    
    # Тест 2: С входом часть нейронов активна
    pop.reset()
    activities = []
    for _ in range(1000):
        pop.step(20.0)
        activities.append(pop.get_activity())
    
    mean_activity = np.mean(activities)
    assert mean_activity > 0, "Нейроны должны быть активны"
    print(f"  ✓ С входом 20 мВ → средняя активность: {mean_activity:.3f}")
    
    # Тест 3: Разные токи → разная активность
    pop.reset()
    activity_low = []
    for _ in range(1000):
        pop.step(16.0)
        activity_low.append(pop.get_activity())
    
    pop.reset()
    activity_high = []
    for _ in range(1000):
        pop.step(30.0)
        activity_high.append(pop.get_activity())
    
    mean_low = np.mean(activity_low)
    mean_high = np.mean(activity_high)
    assert mean_high > mean_low, "Больше тока → больше активность"
    print(f"  ✓ Ток 16 мВ → активность {mean_low:.3f}")
    print(f"  ✓ Ток 30 мВ → активность {mean_high:.3f}")
    
    # Тест 4: Средний потенциал
    pop.reset()
    for _ in range(100):
        pop.step(10.0)
    mean_v = pop.get_mean_potential()
    assert mean_v > pop.neurons[0].v_rest, "Потенциал должен вырасти от входа"
    print(f"  ✓ Средний потенциал: {mean_v:.1f} мВ (выше покоя)")
    
    print("LIF Population: OK\n")

from neurons.izhikevich import IzhikevichNeuron, IzhikevichPopulation, NEURON_TYPES


def test_izhikevich_neuron():
    """Тест Izhikevich нейрона"""
    print("Testing Izhikevich Neuron...")
    
    # Тест 1: Без входа не спайкает
    neuron = IzhikevichNeuron("regular_spiking")
    neuron.reset()
    spiked = False
    for _ in range(1000):
        if neuron.step(0.0):
            spiked = True
    assert not spiked, "Не должен спайкать без входа"
    print("  ✓ Без входа → нет спайков")
    
    # Тест 2: С током спайкает
    neuron.reset()
    spike_count = 0
    for _ in range(10000):
        if neuron.step(10.0):
            spike_count += 1
    assert spike_count > 0, "Должен спайкать с током"
    print(f"  ✓ Regular Spiking: {spike_count} спайков")
    
    # Тест 3: Все типы нейронов работают
    for name in NEURON_TYPES:
        n = IzhikevichNeuron(name)
        n.reset()
        count = 0
        for _ in range(10000):
            if n.step(10.0):
                count += 1
        print(f"  ✓ {name}: {count} спайков — {n.description}")
    
    # Тест 4: Fast spiking быстрее regular
    rs = IzhikevichNeuron("regular_spiking")
    fs = IzhikevichNeuron("fast_spiking")
    rs.reset()
    fs.reset()
    
    rs_count = sum(1 for _ in range(10000) if rs.step(10.0))
    fs_count = sum(1 for _ in range(10000) if fs.step(10.0))
    
    assert fs_count > rs_count, "Fast spiking должен быть быстрее"
    print(f"  ✓ Fast ({fs_count}) > Regular ({rs_count})")
    
    print("Izhikevich Neuron: OK\n")


def test_izhikevich_population():
    """Тест популяции Izhikevich"""
    print("Testing Izhikevich Population...")
    
    # Тест 1: Популяция с шумом
    pop = IzhikevichPopulation(n_neurons=50, neuron_type="regular_spiking", noise=0.5)
    pop.reset()
    
    activities = []
    for _ in range(1000):
        pop.step(10.0)
        activities.append(pop.get_activity())
    
    mean_act = np.mean(activities)
    assert mean_act > 0, "Популяция должна быть активна"
    print(f"  ✓ Популяция (50 нейронов, шум): активность = {mean_act:.3f}")
    
    # Тест 2: Больше тока → больше активность
    pop.reset()
    act_low = []
    for _ in range(1000):
        pop.step(5.0)
        act_low.append(pop.get_activity())
    
    pop.reset()
    act_high = []
    for _ in range(1000):
        pop.step(15.0)
        act_high.append(pop.get_activity())
    
    assert np.mean(act_high) > np.mean(act_low), "Больше тока → больше активность"
    print(f"  ✓ Ток 5: {np.mean(act_low):.3f}, Ток 15: {np.mean(act_high):.3f}")
    
    print("Izhikevich Population: OK\n")

from neurons.stdp import STDPRule, SynapticNetwork


def test_stdp():
    """Тест STDP обучения"""
    print("Testing STDP...")
    
    # Тест 1: Pre до post → усиление
    stdp = STDPRule()
    dw = stdp.compute_dw(dt=10.0)  # Pre на 10 мс раньше
    assert dw > 0, "Pre до post должно усиливать связь"
    print(f"  ✓ Pre до post (dt=10мс): Δw = +{dw:.4f} (усиление)")
    
    # Тест 2: Pre после post → ослабление
    dw = stdp.compute_dw(dt=-10.0)  # Pre на 10 мс позже
    assert dw < 0, "Pre после post должно ослаблять связь"
    print(f"  ✓ Pre после post (dt=-10мс): Δw = {dw:.4f} (ослабление)")
    
    # Тест 3: Далёкие спайки → малое изменение
    dw_close = abs(stdp.compute_dw(dt=5.0))
    dw_far = abs(stdp.compute_dw(dt=50.0))
    assert dw_close > dw_far, "Близкие спайки → большее изменение"
    print(f"  ✓ Близко (5мс): {dw_close:.4f} > Далеко (50мс): {dw_far:.4f}")
    
    print("STDP Rule: OK\n")


def test_synaptic_network():
    """Тест синаптической сети с STDP"""
    print("Testing Synaptic Network...")
    
    net = SynapticNetwork(n_pre=10, n_post=5, initial_weight=0.5)
    
    # Тест 1: Начальный вес
    initial_mean = net.get_mean_weight()
    assert abs(initial_mean - 0.5) < 0.1, "Начальный вес должен быть ~0.5"
    print(f"  ✓ Начальный средний вес: {initial_mean:.3f}")
    
    # Тест 2: Коррелированные спайки → усиление
    net_corr = SynapticNetwork(n_pre=10, n_post=5, initial_weight=0.5)
    
    for _ in range(1000):
        # Pre спайкают
        pre = np.zeros(10)
        pre[0:3] = 1  # Нейроны 0,1,2 активны
        
        # Post спайкают ПОСЛЕ pre (через 1 шаг)
        currents = net_corr.step(pre, np.zeros(5))
        
        post = np.zeros(5)
        post[0] = 1  # Нейрон 0 активен
        net_corr.step(np.zeros(10), post)
    
    # Связи от pre[0:3] к post[0] должны усилиться
    mean_after = net_corr.get_mean_weight()
    stats = net_corr.get_weight_stats()
    print(f"  ✓ После обучения: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
    
    # Тест 3: Токи передаются
    net2 = SynapticNetwork(n_pre=5, n_post=3, initial_weight=0.8)
    pre_spikes = np.array([1, 0, 1, 0, 0], dtype=float)
    post_spikes = np.zeros(3)
    currents = net2.step(pre_spikes, post_spikes)
    
    assert np.any(currents > 0), "Должны быть входные токи"
    print(f"  ✓ Токи передаются: {currents}")
    
    print("Synaptic Network: OK\n")

def main():
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ НЕЙРОННОГО СУБСТРАТА")
    print("=" * 50)
    print()
    
    try:
        test_lif_neuron()
        test_lif_population()
        test_izhikevich_neuron()
        test_izhikevich_population()
        test_stdp()
        test_synaptic_network()
        
        print("=" * 50)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✓")
        print("=" * 50)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ ОШИБКА: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())