"""
Запуск Элли.
"""

import sys
from brain import Brain
from emotion import EmotionCore
from memory import Memory
from drives import Drives
from life_cycle import LifeCycle
from config import ELLI_NAME
from telegram_bot import TelegramInterface

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    print(f"\n  {ELLI_NAME} просыпается...")

    brain = Brain()
    emotion = EmotionCore()
    memory = Memory()
    drives = Drives()
    life = LifeCycle(drives, emotion)

    # Запускаем фоновую жизнь
    telegram = TelegramInterface(brain, emotion, memory, drives)
    telegram.start_bot()
    life.set_telegram(telegram)

    life.start()

    print(f"  {memory.get_status()} {drives.get_status()}")
    print(f"  Пиши что-нибудь. 'выход' — уйти.\n")

    session_messages = []
    first_message = True

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{ELLI_NAME}: Пока!")
            memory.save_episode(session_messages, emotion.valence)
            drives.save()
            life.stop()
            break

        if not user_input:
            continue

        if user_input.lower() in ("выход", "exit"):
            print(f"\n{ELLI_NAME}: Пока!")
            memory.save_episode(session_messages, emotion.valence)
            drives.save()
            life.stop()
            break

        # Эмоции
        emotion.process_input(user_input)

        # Память
        memories = memory.recall(user_input)

        # Потребности
        drives_context = drives.get_context()
        drives.on_conversation()

        # Накопленные мысли (только в первом сообщении сессии)
        thoughts = ""
        if first_message:
            thoughts = life.get_pending_thoughts()
            first_message = False

        # Собираем контекст
        context = emotion.get_context() + memories + drives_context + thoughts

        # Думаем
        status = f"{emotion.get_status()} {memory.get_status()} {drives.get_status()}"
        print(f"\n  ...думает... {status}\n")
        answer = brain.think(user_input, context)

        # Обновляем
        emotion.process_input(answer)
        memory.extract_facts(user_input, answer)
        session_messages.append({"role": "user", "content": user_input})
        session_messages.append({"role": "assistant", "content": answer})

        print(f"  {ELLI_NAME}: {answer}")
        status = f"{emotion.get_status()} {memory.get_status()} {drives.get_status()}"
        print(f"  {status}\n")


if __name__ == "__main__":
    main()