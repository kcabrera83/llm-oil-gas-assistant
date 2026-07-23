"""Flask API application for LLM Oil & Gas Assistant."""

import os
import sys
import json
import time
import joblib
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_assistant.models.rag_engine import RAGEngine
from llm_assistant.models.qa_model import QAModel
from llm_assistant.models.summarizer import Summarizer

app = Flask(__name__, template_folder="templates")

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    t0 = time.time()
    answer_result = rag_engine.generate_answer(query, top_k=top_k)
    qa_result = qa_model.extract_answer(query, top_k=top_k)
    elapsed = time.time() - t0

    return jsonify({
        "query": query,
        "answer": answer_result["answer"],
        "confidence": answer_result["confidence"],
        "sources": answer_result["sources"],
        "num_retrieved": answer_result["num_retrieved"],
        "qa_answer": qa_result["answer"],
        "qa_confidence": qa_result["confidence"],
        "processing_time_ms": round(elapsed * 1000, 2),
    })


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    num_sentences = data.get("num_sentences", 3)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    t0 = time.time()
    result = summarizer.summarize(text, num_sentences=num_sentences)
    elapsed = time.time() - t0

    result["processing_time_ms"] = round(elapsed * 1000, 2)
    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 10)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    t0 = time.time()
    results = rag_engine.retrieve(query, top_k=top_k)
    elapsed = time.time() - t0

    return jsonify({
        "query": query,
        "results": results,
        "total_results": len(results),
        "processing_time_ms": round(elapsed * 1000, 2),
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    stats = {}
    if rag_engine:
        stats.update(rag_engine.get_stats())
    if qa_model:
        stats["qa_model"] = qa_model.get_stats()
    return jsonify({
        "loaded_models": loaded_models,
        "stats": stats,
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "healthy",
        "rag_ready": rag_engine.is_ready if rag_engine else False,
        "models_loaded": loaded_models,
        "version": "1.0.0",
    })


if __name__ == "__main__":
    load_models()
    print("\n  LLM Oil & Gas Assistant - Starting server on port 5015")
    print("  Elaborado por Ing. Kelvin Cabrera\n")
    app.run(host="0.0.0.0", port=5015, debug=True)
