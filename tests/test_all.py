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
from limbic.amygdala import Amygdala
from limbic.dopamine import DopamineSystem
from limbic.emotion_core import EmotionCore

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

from neurons.homeostasis import HomeostaticRegulator


def test_homeostasis():
    """Тест гомеостаза"""
    print("Testing Homeostasis...")
    
    # Тест 1: Слишком высокая активность → возбудимость падает
    reg = HomeostaticRegulator(target_rate=5.0, tau=100.0, strength=0.5)
    
    # Имитируем высокую активность
    for _ in range(1000):
        reg.update(spike_count=10, n_neurons=100)  # Много спайков
    
    assert reg.excitability < 1.0, "При высокой активности возбудимость должна упасть"
    print(f"  ✓ Высокая активность → возбудимость: {reg.excitability:.3f} (< 1.0)")
    
    # Тест 2: Слишком низкая активность → возбудимость растёт
    reg2 = HomeostaticRegulator(target_rate=5.0, tau=100.0, strength=0.5)
    
    for _ in range(1000):
        reg2.update(spike_count=0, n_neurons=100)  # Нет спайков
    
    assert reg2.excitability > 1.0, "При низкой активности возбудимость должна вырасти"
    print(f"  ✓ Низкая активность → возбудимость: {reg2.excitability:.3f} (> 1.0)")
    
    # Тест 3: Нормальная активность → возбудимость ~1.0
    reg3 = HomeostaticRegulator(target_rate=5.0, tau=100.0, strength=0.5)
    
    for _ in range(1000):
        # spike_count чтобы получить ~5 Гц:
        # 5 Гц = spike_count/n_neurons * 1000/DT
        # spike_count = 5 * 100 * 0.1 / 1000 = 0.05
        reg3.update(spike_count=0.05, n_neurons=100)
    
    assert 0.8 < reg3.excitability < 1.2, "При нормальной активности возбудимость ~1.0"
    print(f"  ✓ Нормальная активность → возбудимость: {reg3.excitability:.3f} (~1.0)")
    
    # Тест 4: scale_input работает
    reg4 = HomeostaticRegulator()
    reg4.excitability = 2.0
    scaled = reg4.scale_input(10.0)
    assert scaled == 20.0, "Масштабирование должно работать"
    print(f"  ✓ scale_input(10.0) с excitability=2.0 → {scaled}")
    
    print("Homeostasis: OK\n")

from neurons.encoding import RateEncoder, PopulationEncoder, TextEncoder


def test_encoding():
    """Тест кодирования"""
    print("Testing Encoding...")
    
    # Тест 1: Rate coding — encode/decode
    rate = RateEncoder(max_rate=100.0)
    
    original = 0.7
    spikes = rate.encode(original, duration_ms=1000.0)
    decoded = rate.decode(spikes, duration_ms=1000.0)
    
    error = abs(original - decoded)
    assert error < 0.2, f"Rate coding ошибка слишком большая: {error}"
    print(f"  ✓ Rate coding: {original} → спайки → {decoded:.2f} (ошибка: {error:.2f})")
    
    # Тест 2: Rate coding — больше значение → больше спайков
    spikes_low = rate.encode(0.2, duration_ms=1000.0)
    spikes_high = rate.encode(0.8, duration_ms=1000.0)
    
    assert np.sum(spikes_high) > np.sum(spikes_low), "Больше значение → больше спайков"
    print(f"  ✓ Значение 0.2 → {np.sum(spikes_low)} спайков, 0.8 → {np.sum(spikes_high)} спайков")
    
    # Тест 3: Population coding — encode/decode
    pop = PopulationEncoder(n_neurons=20, value_range=(0.0, 1.0))
    
    original = 0.6
    activations = pop.encode(original)
    decoded = pop.decode(activations)
    
    error = abs(original - decoded)
    assert error < 0.1, f"Population coding ошибка: {error}"
    print(f"  ✓ Population coding: {original} → паттерн → {decoded:.2f} (ошибка: {error:.2f})")
    
    # Тест 4: Population coding — пик в правильном месте
    activations = pop.encode(0.5)
    peak_idx = np.argmax(activations)
    peak_value = pop.preferred[peak_idx]
    assert abs(peak_value - 0.5) < 0.1, "Пик должен быть около 0.5"
    print(f"  ✓ Пик активации для 0.5 на нейроне с preferred={peak_value:.2f}")
    
    # Тест 5: Text encoding — одинаковые слова → одинаковые паттерны
    text_enc = TextEncoder(n_neurons=100)
    
    p1 = text_enc.encode_word("привет")
    p2 = text_enc.encode_word("привет")
    p3 = text_enc.encode_word("пока")
    
    assert np.array_equal(p1, p2), "Одинаковые слова → одинаковые паттерны"
    assert not np.array_equal(p1, p3), "Разные слова → разные паттерны"
    print(f"  ✓ 'привет' == 'привет': True")
    print(f"  ✓ 'привет' == 'пока': False")
    
    # Тест 6: Text similarity
    sim_same = text_enc.similarity("я люблю кошек", "я люблю кошек")
    sim_similar = text_enc.similarity("я люблю кошек", "я люблю собак")
    
def test_amygdala():
    """Тест амигдалы"""
    print("Testing Amygdala...")
    
    amygdala = Amygdala(input_size=100)
    
    # Тест 1: Позитивный текст → положительная валентность
    amygdala.reset()
    result = amygdala.process("привет друг люблю тебя")
    assert result["valence"] > 0, f"Позитивный текст должен дать +, получено: {result['valence']}"
    print(f"  ✓ 'привет друг люблю' → валентность: {result['valence']:+.2f} ({result['emotion']})")
    
    # Тест 2: Негативный текст → отрицательная валентность
    amygdala.reset()
    result = amygdala.process("ненавижу тупой злюсь")
    assert result["valence"] < 0, f"Негативный текст должен дать -, получено: {result['valence']}"
    print(f"  ✓ 'ненавижу тупой злюсь' → валентность: {result['valence']:+.2f} ({result['emotion']})")
    
    # Тест 3: Нейтральный текст → около нуля
    amygdala.reset()
    result = amygdala.process("стол стоит у окна")
    assert -0.3 < result["valence"] < 0.3, f"Нейтральный текст должен дать ~0, получено: {result['valence']}"
    print(f"  ✓ 'стол стоит у окна' → валентность: {result['valence']:+.2f} ({result['emotion']})")
    
    # Тест 4: Инерция работает
    amygdala.reset()
    amygdala.process("отлично супер прекрасно")
    high_valence = amygdala.valence
    
    amygdala.process("плохо")
    assert amygdala.valence > -0.5, "Инерция должна смягчать резкие изменения"
    print(f"  ✓ Инерция: было {high_valence:+.2f}, после 'плохо' стало {amygdala.valence:+.2f} (плавно)")
    
    # Тест 5: Обучение работает
    amygdala.reset()
    amygdala.learn("кирпич", target_valence=0.8, n_iterations=20)
    result = amygdala.process("кирпич")
    assert result["valence"] > 0, "После обучения 'кирпич' должен быть позитивным"
    print(f"  ✓ После обучения 'кирпич' → валентность: {result['valence']:+.2f}")
    
    # Тест 6: Сколько слов знает
    count = amygdala.get_known_words_count()
    assert count > 40, "Должна знать минимум 40 слов"
    print(f"  ✓ Знает {count} эмоциональных слов")
    
    print("Amygdala: OK\n")

def test_dopamine():
    """Тест дофаминовой системы"""
    print("Testing Dopamine...")
    
    dop = DopamineSystem()
    
    # Тест 1: Неожиданная награда → положительный дофамин
    dop.reset()
    result = dop.process(0.8)  # Получил хорошее, ожидал 0
    assert result["rpe"] > 0, "Неожиданная награда → положительный RPE"
    assert result["level"] > 0, "Дофамин должен быть положительным"
    print(f"  ✓ Неожиданная награда: RPE={result['rpe']:+.2f}, дофамин={result['level']:+.2f}")
    
    # Тест 2: Ожидаемая награда → мало дофамина
    result2 = dop.process(0.8)  # Повторно — уже ожидает
    assert abs(result2["rpe"]) < abs(result["rpe"]), "Повторная награда → меньше RPE"
    print(f"  ✓ Повторная награда: RPE={result2['rpe']:+.2f} (меньше)")
    
    # Тест 3: Разочарование
    dop2 = DopamineSystem()
    dop2.expected_reward = 0.5  # Ожидает хорошее
    result3 = dop2.process(-0.3)  # Получает плохое
    assert result3["rpe"] < 0, "Разочарование → отрицательный RPE"
    print(f"  ✓ Разочарование: RPE={result3['rpe']:+.2f}, дофамин={result3['level']:+.2f}")
    
    # Тест 4: Learning boost
    dop3 = DopamineSystem()
    dop3.level = 0.8  # Высокий дофамин
    boost = dop3.get_learning_boost()
    assert boost > 1.5, "Высокий дофамин → усиленное обучение"
    print(f"  ✓ Learning boost при дофамине 0.8: {boost:.1f}x")
    
    print("Dopamine: OK\n")


def test_emotion_core():
    """Тест эмоционального ядра"""
    print("Testing Emotion Core...")
    
    core = EmotionCore()
    
    # Тест 1: Позитивный текст
    result = core.process("привет друг!")
    assert result["valence"] > 0, "Позитивный текст → положительная валентность"
    print(f"  ✓ 'привет друг!' → валентность={result['valence']:+.2f}, эмоция={result['emotion']}")
    
    # Тест 2: Серия позитива → привязанность растёт
    initial_attachment = core.attachment
    for msg in ["ты супер", "люблю тебя", "спасибо", "ты молодец"]:
        core.process(msg)
    assert core.attachment > initial_attachment, "Позитив → привязанность растёт"
    print(f"  ✓ Привязанность после позитива: {core.attachment:.2f} (было {initial_attachment:.2f})")
    
    # Тест 3: Грубость → доверие падает
    initial_trust = core.trust
    core.process("ты тупая дура")
    assert core.trust < initial_trust, "Грубость → доверие падает"
    print(f"  ✓ Доверие после грубости: {core.trust:.2f} (было {initial_trust:.2f})")
    
    # Тест 4: Контекст для LLM
    context = core.get_context_for_llm()
    assert len(context) > 20, "Контекст должен быть непустым"
    print(f"  ✓ Контекст для LLM: {len(context)} символов")
    print(f"    '{context[:80]}...'")
    
    # Тест 5: Энергия падает
    initial_energy = core.energy
    for _ in range(10):
        core.process("тест")
    assert core.energy < initial_energy, "Энергия должна падать"
    print(f"  ✓ Энергия: {core.energy:.2f} (было {initial_energy:.2f})")
    
    # Тест 6: Отдых восстанавливает энергию
    core.rest(minutes=120)
    assert core.energy > 0.9, "Отдых должен восстановить энергию"
    print(f"  ✓ Энергия после отдыха: {core.energy:.2f}")
    
    print("Emotion Core: OK\n")

def main():
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ ЭЛЛИ")
    print("=" * 50)
    print()
    
    try:
        test_lif_neuron()
        test_lif_population()
        test_izhikevich_neuron()
        test_izhikevich_population()
        test_stdp()
        test_synaptic_network()
        test_homeostasis()
        test_encoding()
        test_amygdala()
        test_dopamine()
        test_emotion_core()
        
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