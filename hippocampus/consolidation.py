"""
Консолидация памяти — "сон".

Когда Элли "спит" (нет активности):
1. Важные эпизоды "прокручиваются" (replay)
2. Слабые воспоминания забываются
3. Из эпизодов извлекаются обобщения → семантическая память
"""


class Consolidation:
    """
    Процесс консолидации памяти.
    """
    
    def __init__(self, episodic_memory, semantic_memory):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
    
    def run(self, cycles=5):
        """
        Запустить консолидацию.
        
        Args:
            cycles: Количество циклов (больше = глубже обработка)
            
        Returns:
            dict: Результаты консолидации
        """
        results = {
            "replayed": 0,
            "forgotten_episodes": 0,
            "forgotten_facts": 0,
            "extracted_facts": 0,
        }
        
        for _ in range(cycles):
            # 1. Replay важных эпизодов (усиливает их)
            results["replayed"] += self._replay()
            
            # 2. Забывание слабых
            results["forgotten_episodes"] += self.episodic.decay_all()
            results["forgotten_facts"] += self.semantic.decay_all()
        
        # 3. Извлечение обобщений
        results["extracted_facts"] += self._extract_generalizations()
        
        return results
    
    def _replay(self):
        """
        Replay: прокрутить важные эпизоды.
        Усиливает их в памяти.
        """
        count = 0
        for ep in self.episodic.episodes:
            # Прокручиваем только важные
            if ep.importance > 0.5 and ep.strength > 0.3:
                ep.strength = min(1.0, ep.strength + 0.05)
                ep.recall_count += 1
                count += 1
        return count
    
    def _extract_generalizations(self):
        """
        Извлечь обобщения из эпизодов.
        Например: если в 3 эпизодах обсуждали музыку →
        факт "собеседник интересуется музыкой".
        """
        count = 0
        
        # Считаем частоту тем
        topic_counts = {}
        for ep in self.episodic.episodes:
            words = ep.summary.lower().split()
            for word in words:
                if len(word) > 4:  # Только значимые слова
                    topic_counts[word] = topic_counts.get(word, 0) + 1
        
        # Частые темы → факты
        for topic, freq in topic_counts.items():
            if freq >= 3:
                fact = f"Часто обсуждаемая тема: {topic}"
                self.semantic.store(fact, source="консолидация", importance=0.4)
                count += 1
        
        return count