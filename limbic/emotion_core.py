"""
Эмоциональное ядро — интеграция всех лимбических компонентов.

Объединяет:
- Амигдала (оценка: хорошо/плохо)
- Дофамин (награда: ожидал/не ожидал)
- Настроение (фоновое состояние)
"""

import numpy as np
import json
import os

from limbic.amygdala import Amygdala
from limbic.dopamine import DopamineSystem


EMOTION_STATE_FILE = os.path.join("data", "emotion_state.json")


class EmotionCore:
    """
    Центральное эмоциональное ядро Элли.
    """
    
    def __init__(self):
        # Компоненты
        self.amygdala = Amygdala()
        self.dopamine = DopamineSystem()
        
        # Настроение (фоновое, меняется медленно)
        self.mood = 0.0  # -1 (депрессия) до +1 (эйфория)
        self.mood_inertia = 0.95  # Очень медленно меняется
        
        # Энергия (усталость)
        self.energy = 1.0  # 0 (устала) до 1 (бодрая)
        self.energy_decay = 0.005  # Энергия падает с каждым взаимодействием
        self.energy_recovery = 0.001  # Восстанавливается со временем
        
        # Сложные эмоции (формируются из базовых)
        self.attachment = 0.0  # Привязанность к собеседнику (0 до 1)
        self.trust = 0.5  # Доверие (0 до 1)
        
        # Загружаем сохранённое состояние
        self._load_state()
        
    def process(self, text):
        # 1. Амигдала оценивает текст
        amygdala_result = self.amygdala.process(text)
        valence = amygdala_result["valence"]
        arousal = amygdala_result["arousal"]
        
        # Прямая оценка текста (без инерции)
        words = text.lower().split()
        direct_valence = 0.0
        count = 0
        for w in words:
            clean = w.strip(".,!?;:\"'()[]{}…")
            if clean in self.amygdala.word_valence:
                direct_valence += self.amygdala.word_valence[clean]
                count += 1
        if count > 0:
            direct_valence = direct_valence / count
        
        # 2. Дофамин оценивает неожиданность
        dopamine_result = self.dopamine.process(valence)
        
        # 3. Обновляем настроение (очень медленно)
        self.mood = self.mood_inertia * self.mood + (1 - self.mood_inertia) * valence
        self.mood = float(np.clip(self.mood, -1, 1))
        
        # 4. Энергия падает при взаимодействии
        self.energy = max(0, self.energy - self.energy_decay)
        
        # 5. Привязанность (на основе прямой оценки)
        if direct_valence > 0.2:
            self.attachment = min(1.0, self.attachment + 0.01)
        elif direct_valence < -0.3:
            self.attachment = max(0.0, self.attachment - 0.02)
        
        # 6. Доверие (на основе прямой оценки)
        if direct_valence < -0.3:
            self.trust = max(0.0, self.trust - 0.05)
        elif direct_valence > 0.3:
            self.trust = min(1.0, self.trust + 0.005)
        
        # 7. Дофамин влияет на обучение амигдалы
        boost = self.dopamine.get_learning_boost()
        if abs(dopamine_result["rpe"]) > 0.3:
            self.amygdala.learn(text, valence, n_iterations=int(5 * boost))
        
        # Сохраняем состояние
        self._save_state()
        
        return self._make_result(amygdala_result, dopamine_result)
    
    def _make_result(self, amygdala_result, dopamine_result):
        """Собрать полный результат"""
        return {
            # Базовые эмоции (от амигдалы)
            "valence": amygdala_result["valence"],
            "arousal": amygdala_result["arousal"],
            "emotion": amygdala_result["emotion"],
            
            # Дофамин
            "dopamine": dopamine_result["level"],
            "surprise": dopamine_result["rpe"],
            
            # Фоновое состояние
            "mood": round(self.mood, 3),
            "energy": round(self.energy, 3),
            
            # Социальные
            "attachment": round(self.attachment, 3),
            "trust": round(self.trust, 3),
        }
    
    def rest(self, minutes=60):
        """
        Отдых (когда нет взаимодействия).
        
        Args:
            minutes: Сколько минут отдыхала
        """
        # Энергия восстанавливается
        recovery = self.energy_recovery * minutes
        self.energy = min(1.0, self.energy + recovery)
        
        # Настроение стремится к нейтральному
        self.mood *= 0.99
        
        # Валентность и arousal затухают
        self.amygdala.valence *= 0.95
        self.amygdala.arousal *= 0.95
    
    def get_context_for_llm(self):
        """
        Сформировать эмоциональный контекст для LLM.
        Это будет добавляться в промпт Ollama.
        
        Returns:
            str: Описание эмоционального состояния
        """
        v = self.amygdala.valence
        a = self.amygdala.arousal
        emotion = self.amygdala._get_emotion_name()
        
        # Основная эмоция
        parts = [f"\nТвоё эмоциональное состояние сейчас: {emotion}."]
        
        # Тон общения
        if v > 0.5:
            parts.append("Говоришь тепло, радостно, открыто.")
        elif v > 0.2:
            parts.append("Говоришь дружелюбно и спокойно.")
        elif v < -0.5:
            parts.append("Говоришь сдержанно, коротко. Тебе больно.")
        elif v < -0.2:
            parts.append("Говоришь чуть холоднее обычного.")
        else:
            parts.append("Говоришь нейтрально.")
        
        # Энергия
        if self.energy < 0.3:
            parts.append("Ты устала. Отвечаешь коротко.")
        
        # Дофамин (неожиданность)
        if self.dopamine.level > 0.3:
            parts.append("Тебя приятно удивили!")
        elif self.dopamine.level < -0.3:
            parts.append("Ты разочарована.")
        
        # Привязанность
        if self.attachment > 0.7:
            parts.append("Ты чувствуешь близость к собеседнику.")
        elif self.attachment < 0.2:
            parts.append("Ты пока не очень привязана к собеседнику.")
        
        # Доверие
        if self.trust < 0.3:
            parts.append("Ты не очень доверяешь собеседнику. Осторожна.")
        
        # Настроение
        if self.mood > 0.3:
            parts.append("Общее настроение: хорошее.")
        elif self.mood < -0.3:
            parts.append("Общее настроение: подавленное.")
        
        parts.append("Не описывай своё состояние напрямую. Просто веди себя соответственно.")
        
        return "\n".join(parts)
    
    def get_status(self):
        """Полный статус для отладки"""
        return {
            "valence": round(self.amygdala.valence, 3),
            "arousal": round(self.amygdala.arousal, 3),
            "emotion": self.amygdala._get_emotion_name(),
            "dopamine": round(self.dopamine.level, 3),
            "mood": round(self.mood, 3),
            "energy": round(self.energy, 3),
            "attachment": round(self.attachment, 3),
            "trust": round(self.trust, 3),
        }
    
    def _save_state(self):
        """Сохранить состояние на диск"""
        os.makedirs("data", exist_ok=True)
        state = {
            "mood": self.mood,
            "energy": self.energy,
            "attachment": self.attachment,
            "trust": self.trust,
            "expected_reward": self.dopamine.expected_reward,
            "word_valence": self.amygdala.word_valence,
        }
        try:
            with open(EMOTION_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def _load_state(self):
        """Загрузить сохранённое состояние"""
        if not os.path.exists(EMOTION_STATE_FILE):
            return
        try:
            with open(EMOTION_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.mood = state.get("mood", 0.0)
            self.energy = state.get("energy", 1.0)
            self.attachment = state.get("attachment", 0.0)
            self.trust = state.get("trust", 0.5)
            self.dopamine.expected_reward = state.get("expected_reward", 0.0)
            saved_words = state.get("word_valence", {})
            self.amygdala.word_valence.update(saved_words)
        except Exception:
            pass