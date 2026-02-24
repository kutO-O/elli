"""
Запуск Элли.
"""

import sys
from brain import Brain
from config import ELLI_NAME


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")

    print(f"\n  {ELLI_NAME} просыпается...")
    print(f"  Пиши что-нибудь. 'выход' — уйти.\n")

    brain = Brain()

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

        print(f"\n  ...думает...\n")
        answer = brain.think(user_input)
        print(f"  {ELLI_NAME}: {answer}\n")


if __name__ == "__main__":
    main()