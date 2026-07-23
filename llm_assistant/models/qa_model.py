from typing import Dict, List, Optional
from ..utils.text_processor import TextProcessor


class QAModel:
    """Extractive QA model that finds answer spans within retrieved documents."""

    def __init__(self):
        self.processor = TextProcessor(max_chunk_size=256, overlap=32)
        self.documents: List[Dict] = []
        self.is_ready = False

    def load_documents(self, documents: List[Dict]) -> None:
        self.documents = documents
        self.processor.index_documents(documents)
        self.is_ready = True

    def extract_answer(self, question: str, context: Optional[str] = None, top_k: int = 5) -> Dict:
        if not self.is_ready:
            return {
                "answer": "Model not loaded. Run training first.",
                "confidence": 0.0,
                "passages": [],
            }

        if context:
            passages = self._extract_from_context(question, context)
        else:
            search_results = self.processor.search(question, top_k=top_k)
            passages = []
            for content, score, metadata in search_results:
                answer_span = self._find_best_span(question, content)
                passages.append({
                    "text": content,
                    "answer_span": answer_span,
                    "relevance": round(score, 4),
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", ""),
                })

        if not passages:
            return {
                "answer": "Unable to extract answer from available documents.",
                "confidence": 0.0,
                "passages": [],
            }

        best = passages[0]
        answer = best.get("answer_span", best["text"])
        confidence = best.get("relevance", 0.0)

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "passages": passages[:3],
        }

    def _extract_from_context(self, question: str, context: str) -> List[Dict]:
        sentences = [s.strip() for s in context.replace(".", ".\n").split("\n") if s.strip()]
        question_words = set(self.processor.tokenize(self.processor.clean_text(question)))

        scored = []
        for sentence in sentences:
            clean_sent = self.processor.clean_text(sentence)
            sent_words = set(self.processor.tokenize(clean_sent))
            overlap = len(question_words & sent_words)
            score = overlap / max(len(question_words), 1)
            scored.append({"text": sentence, "relevance": round(score, 4)})

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:5]

    def _find_best_span(self, question: str, passage: str) -> str:
        sentences = [s.strip() for s in passage.split(".") if s.strip()]
        if not sentences:
            return passage

        question_words = set(self.processor.tokenize(self.processor.clean_text(question)))

        best_score = -1
        best_span = sentences[0]

        for sentence in sentences:
            sent_words = set(self.processor.tokenize(self.processor.clean_text(sentence)))
            overlap = len(question_words & sent_words)
            if overlap > best_score:
                best_score = overlap
                best_span = sentence.strip()

        return best_span + "."

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "ready": self.is_ready,
        }
