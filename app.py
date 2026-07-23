"""FastAPI application for LLM Oil & Gas Assistant."""

import os
import sys
import json
import time
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(
    title="LLM Oil & Gas Assistant",
    description="RAG + QA LLM Assistant for Oil & Gas domain",
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
    sources: List[Any]
    num_retrieved: int
    processing_time_ms: float


class SummarizeRequest(BaseModel):
    text: str
    num_sentences: int = 3


class SummarizeResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    num_sentences: int
    processing_time_ms: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    query: str
    results: Any
    total_results: int
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
    answer = context[:500] if context else "No relevant documents found."
    sources = [{"title": d.metadata.get("title", ""), "category": d.metadata.get("category", ""), "source": d.metadata.get("source", "")} for d in docs]
    elapsed = time.time() - t0

    return AskResponse(
        query=query,
        answer=answer,
        sources=sources,
        num_retrieved=len(docs),
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
async def api_summarize(request: SummarizeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    t0 = time.time()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    num = min(request.num_sentences, len(sentences))
    summary = ". ".join(sentences[:num]) + "." if num > 0 else ""
    elapsed = time.time() - t0

    return SummarizeResponse(
        summary=summary,
        original_length=len(text),
        summary_length=len(summary),
        num_sentences=num,
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not models.get("vectorstore"):
        raise HTTPException(status_code=503, detail="Vector store not loaded")

    t0 = time.time()
    results_with_scores = models["vectorstore"].similarity_search_with_score(query, k=request.top_k)
    results = []
    for doc, score in results_with_scores:
        results.append({
            "title": doc.metadata.get("title", ""),
            "category": doc.metadata.get("category", ""),
            "source": doc.metadata.get("source", ""),
            "score": float(score),
            "content": doc.page_content[:200],
        })
    elapsed = time.time() - t0

    return SearchResponse(
        query=query,
        results=results,
        total_results=len(results),
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "LLM Oil & Gas Assistant", "version": "2.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/ask": {"post": {"summary": "Ask a question using FAISS RAG"}},
            "/api/summarize": {"post": {"summary": "Summarize text"}},
            "/api/search": {"post": {"summary": "Search knowledge base"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    load_models()
    print("\n  LLM Oil & Gas Assistant - Starting server on port 5015")
    print("  Elaborado por Ing. Kelvin Cabrera\n")
    uvicorn.run(app, host="0.0.0.0", port=5015)
