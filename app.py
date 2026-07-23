"""FastAPI application for LLM Oil & Gas Assistant."""

import os
import sys
import json
import time
import joblib
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_assistant.models.rag_engine import RAGEngine
from llm_assistant.models.qa_model import QAModel
from llm_assistant.models.summarizer import Summarizer

app = FastAPI(
    title="LLM Oil & Gas Assistant",
    description="RAG + QA LLM Assistant for Oil & Gas domain",
    version="1.0.0",
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
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_assistant", "knowledge_base", "documents.json")

rag_engine = None
qa_model = None
summarizer = None
loaded_models = []


def load_models():
    global rag_engine, qa_model, summarizer, loaded_models
    rag_engine = RAGEngine()
    qa_model = QAModel()
    summarizer = Summarizer()

    rag_path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    qa_path = os.path.join(OUTPUT_DIR, "qa_model.joblib")

    if os.path.exists(rag_path):
        rag_engine = joblib.load(rag_path)
        loaded_models.append("RAG Engine (TF-IDF)")

    if os.path.exists(qa_path):
        qa_model = joblib.load(qa_path)
        loaded_models.append("QA Model (Extractive)")

    loaded_models.append("Summarizer (Extractive)")

    if rag_engine and rag_engine.is_ready:
        docs = rag_engine.documents
        print(f"Loaded {len(docs)} documents into knowledge base")


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: Any
    num_retrieved: int
    qa_answer: str
    qa_confidence: float
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
    load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "rag_ready": rag_engine.is_ready if rag_engine else False,
        "models_loaded": loaded_models,
        "version": "1.0.0",
    }


@app.get("/api/models")
async def models_info():
    stats = {}
    if rag_engine:
        stats.update(rag_engine.get_stats())
    if qa_model:
        stats["qa_model"] = qa_model.get_stats()
    return {
        "loaded_models": loaded_models,
        "stats": stats,
    }


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(request: AskRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    t0 = time.time()
    answer_result = rag_engine.generate_answer(query, top_k=request.top_k)
    qa_result = qa_model.extract_answer(query, top_k=request.top_k)
    elapsed = time.time() - t0

    return AskResponse(
        query=query,
        answer=answer_result["answer"],
        confidence=answer_result["confidence"],
        sources=answer_result["sources"],
        num_retrieved=answer_result["num_retrieved"],
        qa_answer=qa_result["answer"],
        qa_confidence=qa_result["confidence"],
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
async def api_summarize(request: SummarizeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    t0 = time.time()
    result = summarizer.summarize(text, num_sentences=request.num_sentences)
    elapsed = time.time() - t0

    return SummarizeResponse(
        summary=result.get("summary", ""),
        original_length=result.get("original_length", 0),
        summary_length=result.get("summary_length", 0),
        num_sentences=result.get("num_sentences", request.num_sentences),
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/search", response_model=SearchResponse)
async def api_search(request: SearchRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    t0 = time.time()
    results = rag_engine.retrieve(query, top_k=request.top_k)
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
        "info": {"title": "LLM Oil & Gas Assistant", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/ask": {"post": {"summary": "Ask a question using RAG + QA models"}},
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

