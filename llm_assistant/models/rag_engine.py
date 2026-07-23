from typing import List, Dict, Optional
from ..utils.text_processor import TextProcessor


class RAGEngine:
    """Retrieval-Augmented Generation engine using TF-IDF + keyword matching."""

    def __init__(self):
        self.text_processor = TextProcessor(max_chunk_size=256, overlap=32)
        self.documents: List[Dict] = []
        self.is_ready = False

    def build_index(self, documents: List[Dict]) -> None:
        self.documents = documents
        self.text_processor.index_documents(documents)
        self.is_ready = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.is_ready:
            return []

        results = self.text_processor.search(query, top_k=top_k)
        retrieved = []
        for content, score, metadata in results:
            retrieved.append({
                "content": content,
                "relevance_score": round(score, 4),
                "title": metadata.get("title", ""),
                "category": metadata.get("category", ""),
                "source": metadata.get("source", ""),
            })
        return retrieved

    def generate_answer(self, query: str, top_k: int = 5) -> Dict:
        if not self.is_ready:
            return {
                "answer": "Knowledge base is not built yet. Please run training first.",
                "sources": [],
                "confidence": 0.0,
            }

        retrieved = self.retrieve(query, top_k=top_k)

        if not retrieved:
            return {
                "answer": "No relevant information found for your query. Try rephrasing your question.",
                "sources": [],
                "confidence": 0.0,
            }

        answer_parts = []
        sources = []
        total_relevance = 0.0

        for i, item in enumerate(retrieved[:3]):
            answer_parts.append(item["content"])
            sources.append({
                "title": item["title"],
                "category": item["category"],
                "relevance_score": item["relevance_score"],
            })
            total_relevance += item["relevance_score"]

        avg_confidence = total_relevance / min(len(retrieved), 3)

        combined_answer = " ".join(answer_parts)

        return {
            "answer": combined_answer,
            "sources": sources,
            "confidence": round(avg_confidence, 4),
            "num_retrieved": len(retrieved),
        }

    def get_stats(self) -> Dict:
        stats = self.text_processor.get_stats()
        stats["engine_ready"] = self.is_ready
        return stats
