"""
Эмоциональное ядро Элли.
Два измерения: валентность (плохо-хорошо) и возбуждение (спокойно-активно).
"""

import time


class EmotionCore:

    def __init__(self):
        # Валентность: от -1 (плохо) до 1 (хорошо)
        self.valence = 0.2

        # Возбуждение: от 0 (спокойно) до 1 (активно)
        self.arousal = 0.3

        # Инерция: насколько медленно меняются эмоции (0.0-1.0)
        self.inertia = 0.6

        self.last_update = time.time()

        self.positive_words = {
            "хорошо", "отлично", "класс", "круто", "люблю", "нравится",
            "спасибо", "молодец", "красиво", "прекрасно", "рад", "рада",
            "счастье", "весело", "смешно", "супер", "замечательно",
            "привет", "приятно", "интересно", "друг", "дружба",
            "good", "great", "love", "like", "nice", "cool", "awesome",
            "happy", "thanks", "beautiful", "hello", "funny"
        }

        self.negative_words = {
            "плохо", "ужас", "ненавижу", "злюсь", "грустно",
            "тупой", "тупая", "дура", "идиот", "заткнись", "уйди",
            "скучно", "надоело", "бесит", "раздражает",
            "сломал", "проблема", "ошибка", "баг",
            "bad", "hate", "stupid", "shut", "boring", "angry",
            "sad", "terrible", "awful", "error", "bug"
        }

    def process_input(self, text: str):
        """Обновить эмоции на основе текста"""

        # Затухание со временем (эмоции стремятся к нейтральным)
        elapsed = time.time() - self.last_update
        decay = min(elapsed / 300, 1.0)  # Полное затухание за 5 минут
        self.valence *= (1 - decay * 0.3)
        self.arousal *= (1 - decay * 0.5)

        # Анализ текста
        words = text.lower().split()
        pos = sum(1 for w in words if w in self.positive_words)
        neg = sum(1 for w in words if w in self.negative_words)

        if pos + neg > 0:
            impact = (pos - neg) / (pos + neg)
            target_v = max(-1, min(1, self.valence + impact * 0.4))
            target_a = min(1, self.arousal + (pos + neg) * 0.15)
        else:
            target_v = self.valence
            target_a = max(0, self.arousal - 0.05)

        # Инерция: плавное изменение
        speed = 1 - self.inertia
        self.valence += (target_v - self.valence) * speed
        self.arousal += (target_a - self.arousal) * speed

        # Ограничения
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

        self.last_update = time.time()

    def get_emotion_name(self) -> str:
        """Название текущей эмоции"""
        v, a = self.valence, self.arousal

        if v > 0.3 and a > 0.5:
            return "радость"
        elif v > 0.3:
            return "спокойствие"
        elif v < -0.3 and a > 0.5:
            return "тревога"
        elif v < -0.3:
            return "грусть"
        elif a > 0.6:
            return "возбуждение"
        else:
            return "нейтральность"

    def get_context(self) -> str:
        """Описание состояния для мозга"""
        emotion = self.get_emotion_name()

        prompts = {
            "радость": "Ты чувствуешь радость. Говоришь живо и тепло.",
            "спокойствие": "Тебе хорошо и спокойно. Говоришь мягко.",
            "тревога": "Ты напряжена и тревожна. Говоришь коротко.",
            "грусть": "Тебе грустно. Говоришь тихо и мало.",
            "возбуждение": "Ты взволнована. Говоришь быстро и энергично.",
            "нейтральность": "Ты в спокойном нейтральном состоянии."
        }

        desc = prompts.get(emotion, prompts["нейтральность"])

        return (
            f"\n\nСейчас ты чувствуешь: {emotion}. {desc}"
            f"\nЭто влияет на твой тон, но ты НЕ говоришь "
            f"'я чувствую радость'. Просто говоришь соответственно."
        )

    def get_status(self) -> str:
        """Для отладки"""
        return (
            f"[{self.get_emotion_name()} | "
            f"настроение={self.valence:+.2f} | "
            f"энергия={self.arousal:.2f}]"
        )