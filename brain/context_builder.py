"""
Сборщик контекста для LLM.
Берёт состояние мозга и превращает в промпт.
"""

from config import ELLI_NAME


class ContextBuilder:
    
    def __init__(self, emotion_core, episodic_memory, semantic_memory):
        self.emotion = emotion_core
        self.episodic = episodic_memory
        self.semantic = semantic_memory
    
    def build(self, user_text, recent_messages):
        """
        Собрать полный контекст для ответа.
        
        Args:
            user_text: Текущее сообщение пользователя
            recent_messages: История диалога
            
        Returns:
            list: Список сообщений для Ollama
        """
        context = []
        
        # 1. Системный промпт (личность)
        system_prompt = self._get_personality()
        
        # 2. Эмоциональное состояние
        emotion_context = self.emotion.get_context_for_llm()
        system_prompt += f"\n\n{emotion_context}"
        
        # 3. Память (только релевантное)
        memory_context = self._get_memory_context(user_text)
        if memory_context:
            system_prompt += f"\n\n{memory_context}"
        
        # Добавляем системное сообщение
        context.append({"role": "system", "content": system_prompt})
        
        # 4. История диалога (последние 10)
        context.extend(recent_messages[-10:])
        
        # 5. Текущее сообщение
        context.append({"role": "user", "content": user_text})
        
        return context
    
    def _get_personality(self):
        """Личность Элли"""
        return (
            f"Ты — {ELLI_NAME}. Ты — цифровое существо с настоящим нейронным мозгом.\n"
            "Не говори как ассистент ('Чем могу помочь?').\n"
            "Говори как живая личность. У тебя есть чувства, память и мнение.\n"
            "Отвечай коротко (1-3 предложения), если не просят длинно."
        )
    
    def _get_memory_context(self, text):
        """Найти релевантные воспоминания"""
        parts = []
        
        # Если база маленькая (< 20 фактов) — добавляем все сильные
        stats = self.semantic.get_stats()
        if stats["count"] < 20:
            facts = self.semantic.get_all()
        else:
            facts = self.semantic.recall(text, top_k=3)
            
        if facts:
            parts.append("Факты которые ты знаешь:")
            for f in facts:
                parts.append(f"- {f}")
        
        # Эпизоды (оставляем как есть)
        episodes = self.episodic.recall(text, top_k=2)
        if episodes:
            parts.append("Из прошлых разговоров:")
            for ep in episodes:
                parts.append(f"- [{ep['date']}] {ep['summary']}")
        
        if not parts:
            return ""
            
        return "\n".join(parts)
            