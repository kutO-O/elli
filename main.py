"""
Запуск Элли.
"""

import sys
from brain import Brain
from emotion import EmotionCore
from config import ELLI_NAME


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    print(f"\n  {ELLI_NAME} просыпается...")
    print(f"  Пиши что-нибудь. 'выход' — уйти.\n")

    brain = Brain()
    emotion = EmotionCore()

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{ELLI_NAME}: Пока!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("выход", "exit"):
            print(f"\n{ELLI_NAME}: Пока!\n")
            break

        # Эмоции реагируют на текст
        emotion.process_input(user_input)

        # Мозг думает с учётом эмоций
        print(f"\n  ...думает... {emotion.get_status()}\n")
        context = emotion.get_context()
        answer = brain.think(user_input, context)

        # Эмоции реагируют на свой ответ тоже
        emotion.process_input(answer)

        print(f"  {ELLI_NAME}: {answer}")
        print(f"  {emotion.get_status()}\n")


if __name__ == "__main__":
    main()