"""
Мозг Элли — речевой центр.
"""

import sys
import ollama
from config import OLLAMA_MODEL, ELLI_PERSONALITY, MAX_HISTORY


class Brain:

    def __init__(self):
        self.model = OLLAMA_MODEL
        self.history = []
        self._check_ollama()

    def _check_ollama(self):
        """Проверяем что Ollama работает"""
        try:
            ollama.list()
        except Exception:
            print("ОШИБКА: Ollama не запущена!")
            print("Запусти Ollama и попробуй снова.")
            sys.exit(1)

    def think(self, message: str) -> str:
        """Элли думает и отвечает"""

        # Запоминаем что сказал человек
        self.history.append({"role": "user", "content": message})

        # Собираем контекст для модели
        messages = [
            {"role": "system", "content": ELLI_PERSONALITY}
        ] + self.history[-MAX_HISTORY:]

        # Думаем
        try:
            response = ollama.chat(model=self.model, messages=messages)
            answer = response["message"]["content"]

            # Запоминаем свой ответ
            self.history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            if "not found" in str(e).lower():
                print(f"\nМодель '{self.model}' не найдена!")
                print(f"Скачай: ollama pull {self.model}")
                sys.exit(1)
            return f"*ошибка: {e}*"