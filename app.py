"""FastAPI application for Oil & Gas Retrieval-Augmented Information System."""

import os
import sys
import json
import re
import time
from typing import Any, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(
    title="Oil & Gas Retrieval-Augmented Information System",
    description="RAG-based retrieval system for Oil & Gas domain (retrieval-only, no LLM generation)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
FAISS_INDEX_DIR = os.path.join(OUTPUT_DIR, "faiss_index")

models = {}


def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_DIR):
        models["vectorstore"] = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        models["ready"] = True
        print(f"  Loaded FAISS index from {FAISS_INDEX_DIR}")
    else:
        models["vectorstore"] = None
        models["ready"] = False
        print("  Warning: FAISS index not found. Run train.py first.")
    models["embeddings"] = embeddings


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    context: str
    key_findings: List[str]
    sources: List[Any]
    num_retrieved: int
    processing_time_ms: float
    note: str = "Retrieval-based system. Responses are extracted from retrieved documents, not LLM-generated."


class SummarizeRequest(BaseModel):
    text: str
    num_sentences: int = 3


class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    num_sentences: int
    processing_time_ms: float


@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "rag_ready": models.get("ready", False),
        "models_loaded": ["FAISS Vector Store (HuggingFace embeddings)"] if models.get("ready") else [],
        "version": "2.0.0",
    }


@app.get("/api/models")
async def models_info():
    info = {
        "vectorstore": {
            "type": "FAISS + HuggingFace all-MiniLM-L6-v2",
            "ready": models.get("ready", False),
        },
    }
    if models.get("vectorstore"):
        info["total_vectors"] = models["vectorstore"].index.ntotal
    return info


def _extract_key_findings(query: str, context_text: str, top_n: int = 5) -> List[str]:
    """Extract the most relevant sentences from context using keyword overlap with the query."""
    if not context_text:
        return []
    query_words = set(re.findall(r'\w+', query.lower()))
    query_words = {w for w in query_words if len(w) > 2}
    if not query_words:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', context_text)
    scored = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped or len(s_stripped) < 20:
            continue
        sent_words = set(re.findall(r'\w+', s_stripped.lower()))
        overlap = len(query_words & sent_words)
        scored.append((overlap, s_stripped))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:top_n] if _ > 0]


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not models.get("vectorstore"):
        raise HTTPException(status_code=503, detail="Vector store not loaded")

    t0 = time.time()
    docs = models["vectorstore"].similarity_search(query, k=request.top_k)
    context = "\n".join([d.page_content for d in docs])
    key_findings = _extract_key_findings(query, context)
    answer = context if context else "No relevant documents found."
    sources = [{"title": d.metadata.get("title", ""), "category": d.metadata.get("category", ""), "source": d.metadata.get("source", "")} for d in docs]
    elapsed = time.time() - t0

    return AskResponse(
        query=query,
        answer=answer,
        context=context,
        key_findings=key_findings,
        sources=sources,
        num_retrieved=len(docs),
        processing_time_ms=round(elapsed * 1000, 2),
    )


def _tfidf_extractive_summary(text: str, num_sentences: int) -> str:
    """Extractive summarization using TF-IDF sentence scoring."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return ". ".join(sentences) + "."

    doc_freq = {}
    tokenized = []
    for s in sentences:
        words = set(re.findall(r'\w+', s.lower()))
        tokenized.append(words)
        for w in words:
            doc_freq[w] = doc_freq.get(w, 0) + 1

    n_docs = len(sentences)
    scores = []
    for words in tokenized:
        score = 0.0
        for w in words:
            df = doc_freq.get(w, 1)
            idf = np.log((n_docs + 1) / (df + 1)) + 1
            score += idf
        scores.append(score)

    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:num_sentences]
    top_indices.sort()
    return ". ".join(sentences[i] for i in top_indices) + "."


@app.post("/api/summarize", response_model=SummarizeResponse)
async def api_summarize(request: SummarizeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    t0 = time.time()
    summary = _tfidf_extractive_summary(text, request.num_sentences)
    elapsed = time.time() - t0

    return SummarizeResponse(
        summary=summary,
        original_length=len(text),
        summary_length=len(summary),
        num_sentences=request.num_sentences,
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "Oil & Gas Retrieval-Augmented Information System", "version": "2.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/ask": {"post": {"summary": "Ask a question using FAISS retrieval (retrieval-only, no LLM generation)"}},
            "/api/summarize": {"post": {"summary": "Extractive summarization using TF-IDF scoring"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    load_models()
    print("\n  Oil & Gas Retrieval-Augmented Information System - Starting server on port 5015")
    print("  Elaborado por Ing. Kelvin Cabrera\n")
    uvicorn.run(app, host="0.0.0.0", port=5015)
