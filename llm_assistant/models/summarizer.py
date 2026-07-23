from typing import Dict, List
from ..utils.text_processor import TextProcessor


class Summarizer:
    """Extractive text summarization using sentence scoring."""

    def __init__(self):
        self.processor = TextProcessor()

    def summarize(self, text: str, num_sentences: int = 3) -> Dict:
        if not text or not text.strip():
            return {"summary": "", "num_sentences": 0, "compression_ratio": 0.0}

        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return {
                "summary": text,
                "num_sentences": len(sentences),
                "compression_ratio": 1.0,
            }

        scores = self._score_sentences(sentences)
        ranked = sorted(
            range(len(sentences)), key=lambda i: scores[i], reverse=True
        )[:num_sentences]
        ranked.sort()

        summary_sentences = [sentences[i] for i in ranked]
        summary = ". ".join(s.strip() for s in summary_sentences)
        if not summary.endswith("."):
            summary += "."

        original_len = len(text.split())
        summary_len = len(summary.split())
        compression = summary_len / original_len if original_len > 0 else 0.0

        return {
            "summary": summary,
            "num_sentences": num_sentences,
            "compression_ratio": round(compression, 4),
            "total_sentences": len(sentences),
        }

    def summarize_documents(self, documents: List[Dict], max_docs: int = 3) -> List[Dict]:
        results = []
        for doc in documents[:max_docs]:
            content = doc.get("content", "")
            if content:
                result = self.summarize(content)
                result["title"] = doc.get("title", "")
                result["category"] = doc.get("category", "")
                results.append(result)
        return results

    def _split_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(self, sentences: List[str]) -> List[float]:
        word_freq = {}
        all_words = []
        for sentence in sentences:
            words = self.processor.tokenize(self.processor.clean_text(sentence))
            all_words.extend(words)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        if not word_freq:
            return [0.0] * len(sentences)

        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] /= max_freq

        scores = []
        for sentence in sentences:
            words = self.processor.tokenize(self.processor.clean_text(sentence))
            if not words:
                scores.append(0.0)
                continue
            score = sum(word_freq.get(w, 0) for w in words) / len(words)
            if words == all_words[: len(words)]:
                score *= 1.5
            scores.append(score)
        return scores
