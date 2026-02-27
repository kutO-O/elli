"""
Главный класс мозга Элли.
Координирует все системы.
"""

import threading
import time
import ollama

from config import OLLAMA_MODEL, DEBUG_MODE
from limbic.emotion_core import EmotionCore
from hippocampus.episodic import EpisodicMemory
from hippocampus.semantic import SemanticMemory
from hippocampus.consolidation import Consolidation
from brain.context_builder import ContextBuilder
from body.voice import Voice
from config import VOICE_ENABLED
from hippocampus.vector_memory import VectorMemory

class Brain:
    
    def __init__(self):
        # Подсистемы
        self.emotion = EmotionCore()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.consolidation = Consolidation(self.episodic, self.semantic)
        self.voice = Voice() if VOICE_ENABLED else None
        # Рабочая память (текущий диалог)
        self.working_memory = []
                # Векторная память
        self.vector_memory = VectorMemory()
        
        # Обновляем context_builder
        self.context_builder = ContextBuilder(
            self.emotion, 
            self.episodic, 
            self.semantic,
            self.vector_memory  # <-- добавили
        )
        # Фоновый поток жизни (пока просто сон)
        self.running = False
        self.thread = None
    
    def process_input(self, text):
        """
        Главный цикл обработки сообщения.
        """
        # 1. Эмоциональная реакция
        # Сначала просто оцениваем, не меняем состояние (preview)
        # Но в текущей архитектуре emotion.process() сразу меняет состояние.
        # Это ок для начала.
        emotion_state = self.emotion.process(text)
        
        # 2. Извлечение фактов (параллельно)
        self.semantic.extract_facts(text)
                # 2.5 Сохраняем в векторную память
        self.vector_memory.add_fact(text)
        # 3. Сборка контекста
        messages = self.context_builder.build(text, self.working_memory)
        
               # 4. Генерация ответа
        response_text = ""
        buffer = ""  # Буфер для предложений
        
        try:
            stream = ollama.chat(
    model=OLLAMA_MODEL,
    messages=messages,
    stream=True
)
            
            for chunk in stream:
                content = chunk['message']['content']
                response_text += content
                buffer += content
                yield content
                
                # Если голос включен — отправляем по предложениям
                if self.voice and content in [".", "!", "?", "\n"]:
                    self.voice.say(
                        buffer, 
                        self.emotion.amygdala.valence,
                        self.emotion.amygdala.arousal
                    )
                    buffer = ""
                
        except Exception as e:
            error_msg = f"\n[Ошибка мозга: {e}]"
            response_text = error_msg
            yield error_msg
        
        # 5. Пост-обработка
        # Запоминаем в рабочую память
        self.working_memory.append({"role": "user", "content": text})
        self.working_memory.append({"role": "assistant", "content": response_text})
        
        # Эмоциональная реакция на свой ответ (рефлексия)
        self.emotion.amygdala.process(response_text)
        
        # Сохраняем эпизод (если диалог закончится)
        # Пока просто накапливаем, сохранение будет при выходе или паузе
        
        if DEBUG_MODE:
            self._print_debug(emotion_state)
    
    def save_episode(self):
        """Сохранить текущий разговор как эпизод"""
        if not self.working_memory:
            return
            
        summary = f"Разговор из {len(self.working_memory)//2} сообщений."
        # В будущем: Ollama генерирует summary
        
        self.episodic.store(
            summary=summary,
            messages=self.working_memory,
            valence=self.emotion.amygdala.valence,
            arousal=self.emotion.amygdala.arousal
        )
        self.working_memory = []  # Очищаем рабочую память
        print(" [Эпизод сохранён]")
    
    def start_life(self):
        """Запустить фоновые процессы"""
        self.running = True
        self.thread = threading.Thread(target=self._life_loop, daemon=True)
        self.thread.start()
    
    def stop_life(self):
        """Остановить"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _life_loop(self):
        """Фоновый цикл"""
        while self.running:
            # Раз в минуту восстанавливаем энергию
            time.sleep(60)
            self.emotion.rest(minutes=1)
            
            # Если долго нет активности — запускаем консолидацию
            # (пока заглушка)
    
    def _print_debug(self, state):
        print("\n" + "="*30)
        print(f"Эмоция: {state['emotion']} (v={state['valence']}, a={state['arousal']})")
        print(f"Дофамин: {state['dopamine']}")
        print(f"Привязанность: {state['attachment']}")
        print("="*30 + "\n")