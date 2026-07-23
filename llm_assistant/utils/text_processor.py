import re
import math
from collections import Counter
from typing import List, Tuple, Dict, Optional
import numpy as np


class TextProcessor:
    """Text chunking, TF-IDF vectorization, and cosine similarity search."""

    def __init__(self, max_chunk_size: int = 512, overlap: int = 64):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.vocabulary: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.document_vectors: Optional[np.ndarray] = None
        self.documents: List[Dict] = []
        self.document_chunks: List[str] = []
        self.document_metadata: List[Dict] = []

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\;\:\-\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.split()

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.max_chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.max_chunk_size - self.overlap
        return chunks if chunks else [text]

    def build_vocabulary(self, corpus: List[str]) -> Dict[str, int]:
        vocab = {}
        for doc in corpus:
            for token in self.tokenize(doc):
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocabulary = vocab
        return vocab

    def compute_tfidf(self, corpus: List[str]) -> np.ndarray:
        n_docs = len(corpus)
        self.build_vocabulary(corpus)
        vocab_size = len(self.vocabulary)

        tf_matrix = np.zeros((n_docs, vocab_size))
        for i, doc in enumerate(corpus):
            tokens = self.tokenize(doc)
            counts = Counter(tokens)
            max_count = max(counts.values()) if counts else 1
            for token, count in counts.items():
                if token in self.vocabulary:
                    tf_matrix[i, self.vocabulary[token]] = count / max_count

        df = np.sum(tf_matrix > 0, axis=0)
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1

        tfidf_matrix = tf_matrix * self.idf
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf_matrix = tfidf_matrix / norms
        return tfidf_matrix

    def index_documents(self, documents: List[Dict]) -> None:
        self.documents = documents
        self.document_chunks = []
        self.document_metadata = []

        for doc in documents:
            content = doc.get("content", "")
            chunks = self.chunk_text(content)
            for chunk in chunks:
                self.document_chunks.append(chunk)
                self.document_metadata.append({
                    "title": doc.get("title", ""),
                    "category": doc.get("category", ""),
                    "source": doc.get("source", ""),
                })

        cleaned = [self.clean_text(c) for c in self.document_chunks]
        self.document_vectors = self.compute_tfidf(cleaned)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if self.document_vectors is None or len(self.document_chunks) == 0:
            return []

        query_clean = self.clean_text(query)
        query_tokens = self.tokenize(query_clean)
        query_vec = np.zeros(len(self.vocabulary))
        counts = Counter(query_tokens)
        max_count = max(counts.values()) if counts else 1
        for token, count in counts.items():
            if token in self.vocabulary:
                query_vec[self.vocabulary[token]] = count / max_count
        query_vec = query_vec * self.idf
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        similarities = self.document_vectors @ query_vec
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:
                results.append((
                    self.document_chunks[idx],
                    score,
                    self.document_metadata[idx],
                ))
        return results

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.document_chunks),
            "vocabulary_size": len(self.vocabulary),
            "indexed": self.document_vectors is not None,
        }
