"""
Запуск Элли.
"""

import sys
from brain import Brain
from emotion import EmotionCore
from memory import Memory
from config import ELLI_NAME


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    print(f"\n  {ELLI_NAME} просыпается...")

    brain = Brain()
    emotion = EmotionCore()
    memory = Memory()

    print(f"  {memory.get_status()}")
    print(f"  Пиши что-нибудь. 'выход' — уйти.\n")

    session_messages = []

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{ELLI_NAME}: Пока!")
            memory.save_episode(session_messages, emotion.valence)
            break

        if not user_input:
            continue

        if user_input.lower() in ("выход", "exit"):
            print(f"\n{ELLI_NAME}: Пока!")
            memory.save_episode(session_messages, emotion.valence)
            break

        # Эмоции реагируют
        emotion.process_input(user_input)

        # Вспоминаем что-то по теме
        memories = memory.recall(user_input)

        # Собираем контекст
        context = emotion.get_context() + memories

        # Думаем
        print(f"\n  ...думает... {emotion.get_status()} {memory.get_status()}\n")
        answer = brain.think(user_input, context)

        # Эмоции реагируют на свой ответ
        emotion.process_input(answer)

        # Извлекаем факты из разговора
        memory.extract_facts(user_input, answer)

        # Записываем в сессию
        session_messages.append({"role": "user", "content": user_input})
        session_messages.append({"role": "assistant", "content": answer})

        print(f"  {ELLI_NAME}: {answer}")
        print(f"  {emotion.get_status()} {memory.get_status()}\n")


if __name__ == "__main__":
    main()