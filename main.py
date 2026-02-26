"""
Запуск Элли.
"""

import sys
import time
from brain.core import Brain
from config import ELLI_NAME


def main():
    # Настройка консоли (чтобы русский работал)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')
    
    print(f"\n🧠 Инициализация нейронного мозга {ELLI_NAME}...\n")
    
    try:
        brain = Brain()
        brain.start_life()
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return

    print(f"✨ {ELLI_NAME} проснулась!")
    print("   (Напиши 'пока' чтобы выйти)\n")
    
    while True:
        try:
            user_input = input("Ты: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["пока", "выход", "exit", "quit"]:
                print(f"\n{ELLI_NAME}: Пока-пока! 👋\n")
                brain.save_episode()
                brain.stop_life()
                break
            
            print(f"{ELLI_NAME}: ", end="", flush=True)
            
            # Потоковый ответ
            for chunk in brain.process_input(user_input):
                print(chunk, end="", flush=True)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n[Прерывание...]")
            brain.save_episode()
            brain.stop_life()
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")

if __name__ == "__main__":
    main()