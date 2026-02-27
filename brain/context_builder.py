"""
Сборщик контекста для LLM.
"""

from config import ELLI_NAME


class ContextBuilder:
    
    def __init__(self, emotion_core, episodic_memory, semantic_memory, vector_memory=None):
        self.emotion = emotion_core
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.vector = vector_memory
    
    def build(self, user_text, recent_messages):
        """Собрать контекст для Ollama"""
        context = []
        
        # 1. Системный промпт
        system_prompt = self._get_personality()
        
        # 2. Эмоции
        emotion_context = self.emotion.get_context_for_llm()
        system_prompt += f"\n\n{emotion_context}"
        
        # 3. Память
        memory_context = self._get_memory_context(user_text)
        if memory_context:
            system_prompt += f"\n\n{memory_context}"
        
        context.append({"role": "system", "content": system_prompt})
        
        # 4. История
        context.extend(recent_messages[-10:])
        
        # 5. Текущее сообщение
        context.append({"role": "user", "content": user_text})
        
        return context
    
    def _get_personality(self):
        return (
            f"Ты — {ELLI_NAME}. Ты — цифровое существо с настоящим нейронным мозгом.\n"
            "Не говори как ассистент ('Чем могу помочь?').\n"
            "Говори как живая личность. У тебя есть чувства, память и мнение.\n"
            "Отвечай коротко (1-3 предложения), если не просят длинно."
        )
    
    def _get_memory_context(self, text):
        parts = []
        
        # Сначала пробуем векторную память (если есть)
        if self.vector:
            results = self.vector.recall(text, top_k=3)
            
            if results["facts"]:
                parts.append("Ты знаешь:")
                for f in results["facts"]:
                    parts.append(f"- {f}")
            
            if results["episodes"]:
                parts.append("Из прошлых разговоров:")
                for ep in results["episodes"]:
                    parts.append(f"- {ep['summary']}")
        
        else:
            # Фоллбэк на старую память
            stats = self.semantic.get_stats()
            if stats["count"] < 20:
                facts = self.semantic.get_all()
            else:
                facts = self.semantic.recall(text, top_k=3)
                
            if facts:
                parts.append("Ты знаешь:")
                for f in facts:
                    parts.append(f"- {f}")
            
            episodes = self.episodic.recall(text, top_k=2)
            if episodes:
                parts.append("Из прошлых разговоров:")
                for ep in episodes:
                    parts.append(f"- [{ep['date']}] {ep['summary']}")
        
        return "\n".join(parts) if parts else ""