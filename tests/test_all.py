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