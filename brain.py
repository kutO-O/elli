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

    def think(self, message: str, context: str = "") -> str:
        """Элли думает и отвечает"""

        self.history.append({"role": "user", "content": message})

        messages = [
            {"role": "system", "content": ELLI_PERSONALITY + context}
        ] + self.history[-MAX_HISTORY:]

        try:
            response = ollama.chat(model=self.model, messages=messages)
            answer = response["message"]["content"]
            self.history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            if "not found" in str(e).lower():
                print(f"\nМодель '{self.model}' не найдена!")
                print(f"Скачай: ollama pull {self.model}")
                sys.exit(1)
            return f"*ошибка: {e}*"