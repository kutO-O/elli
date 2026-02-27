"""
Векторная память — поиск по смыслу.
Использует sentence-transformers для эмбеддингов
и ChromaDB для хранения и поиска.
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorMemory:
    """
    Умная память с поиском по смыслу.
    """
    
    def __init__(self, db_path="data/vector_db"):
        os.makedirs(db_path, exist_ok=True)
        
        # Модель для превращения текста в вектор
        print("🧠 Загрузка модели памяти...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Модель памяти загружена")
        
        # База данных векторов
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Коллекции
        self.facts = self.client.get_or_create_collection(
            name="facts",
            metadata={"description": "Факты о мире и собеседнике"}
        )
        self.episodes = self.client.get_or_create_collection(
            name="episodes",
            metadata={"description": "Эпизоды разговоров"}
        )
    
    def add_fact(self, fact_text, metadata=None):
        """
        Добавить факт в память.
        
        Args:
            fact_text: Текст факта
            metadata: Дополнительная информация (dict)
        """
        if not fact_text.strip():
            return
            
        # Генерируем ID из текста
        fact_id = str(hash(fact_text.lower().strip()))
        
        # Проверяем, есть ли уже
        existing = self.facts.get(ids=[fact_id])
        if existing and existing['ids']:
            return  # Уже есть
        
        # Добавляем
        self.facts.add(
            ids=[fact_id],
            documents=[fact_text],
            metadatas=[metadata or {}]
        )
    
    def add_episode(self, summary, full_text, emotion=0.0, importance=0.5):
        """
        Добавить эпизод (разговор).
        
        Args:
            summary: Краткое описание
            full_text: Полный текст разговора
            emotion: Эмоциональная окраска (-1 до 1)
            importance: Важность (0 до 1)
        """
        import time
        episode_id = str(int(time.time() * 1000))
        
        self.episodes.add(
            ids=[episode_id],
            documents=[full_text],
            metadatas=[{
                "summary": summary,
                "emotion": emotion,
                "importance": importance,
                "timestamp": time.time()
            }]
        )
    
    def recall(self, query, top_k=5):
        """
        Вспомнить релевантную информацию.
        
        Args:
            query: Что ищем
            top_k: Сколько результатов
            
        Returns:
            dict: {"facts": [...], "episodes": [...]}
        """
        results = {"facts": [], "episodes": []}
        
        # Ищем факты
        if self.facts.count() > 0:
            fact_results = self.facts.query(
                query_texts=[query],
                n_results=min(top_k, self.facts.count())
            )
            if fact_results['documents'] and fact_results['documents'][0]:
                results["facts"] = fact_results['documents'][0]
        
        # Ищем эпизоды
        if self.episodes.count() > 0:
            ep_results = self.episodes.query(
                query_texts=[query],
                n_results=min(top_k, self.episodes.count())
            )
            if ep_results['documents'] and ep_results['documents'][0]:
                for i, doc in enumerate(ep_results['documents'][0]):
                    meta = ep_results['metadatas'][0][i] if ep_results['metadatas'] else {}
                    results["episodes"].append({
                        "summary": meta.get("summary", doc[:100]),
                        "emotion": meta.get("emotion", 0),
                        "text": doc
                    })
        
        return results
    
    def get_stats(self):
        """Статистика"""
        return {
            "facts": self.facts.count(),
            "episodes": self.episodes.count()
        }