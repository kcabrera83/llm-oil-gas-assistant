"""Training script - builds knowledge base index and saves to outputs/models/."""

import os
import sys
import json
import time
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_assistant.data_generator import generate_knowledge_base
from llm_assistant.utils.text_processor import TextProcessor
from llm_assistant.models.rag_engine import RAGEngine
from llm_assistant.models.qa_model import QAModel
from llm_assistant.models.summarizer import Summarizer


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_assistant", "knowledge_base", "documents.json")


def main():
    print("=" * 60)
    print("  LLM Oil & Gas Assistant - Training Pipeline")
    print("  Elaborado por Ing. Kelvin Cabrera")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate synthetic knowledge base
    print("\n[1/5] Generating synthetic oil & gas knowledge base...")
    t0 = time.time()
    synthetic_docs = generate_knowledge_base()
    print(f"  Generated {len(synthetic_docs)} synthetic documents in {time.time() - t0:.2f}s")

    # Step 2: Load pre-built knowledge base
    print("\n[2/5] Loading pre-built knowledge base...")
    t0 = time.time()
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_docs = json.load(f)
        print(f"  Loaded {len(kb_docs)} pre-built documents in {time.time() - t0:.2f}s")
    else:
        print("  Warning: documents.json not found, using synthetic docs only")
        kb_docs = []

    # Merge documents
    all_docs = []
    for doc in synthetic_docs:
        all_doc = {"title": doc["title"], "category": doc["category"], "content": doc["content"], "source": doc["source"]}
        all_doc["id"] = len(all_docs) + 1
        all_docs.append(all_doc)
    for doc in kb_docs:
        doc_entry = {"title": doc["title"], "category": doc["category"], "content": doc["content"], "source": doc["source"]}
        if "id" not in doc:
            doc_entry["id"] = len(all_docs) + 1
        else:
            doc_entry["id"] = doc["id"]
        all_docs.append(doc_entry)

    print(f"  Total documents in merged knowledge base: {len(all_docs)}")

    # Step 3: Build RAG index
    print("\n[3/5] Building RAG index (TF-IDF vectors)...")
    t0 = time.time()
    rag_engine = RAGEngine()
    rag_engine.build_index(all_docs)
    rag_stats = rag_engine.get_stats()
    print(f"  Index built in {time.time() - t0:.2f}s")
    print(f"  - Documents indexed: {rag_stats['total_documents']}")
    print(f"  - Text chunks created: {rag_stats['total_chunks']}")
    print(f"  - Vocabulary size: {rag_stats['vocabulary_size']}")

    # Step 4: Build QA model
    print("\n[4/5] Building QA model...")
    t0 = time.time()
    qa_model = QAModel()
    qa_model.load_documents(all_docs)
    print(f"  QA model loaded in {time.time() - t0:.2f}s")

    # Step 5: Save models
    print("\n[5/5] Saving models to disk...")
    t0 = time.time()
    joblib.dump(rag_engine, os.path.join(OUTPUT_DIR, "rag_engine.joblib"))
    joblib.dump(qa_model, os.path.join(OUTPUT_DIR, "qa_model.joblib"))
    print(f"  Models saved in {time.time() - t0:.2f}s")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Test retrieval
    print("\n" + "=" * 60)
    print("  Verification Tests")
    print("=" * 60)

    test_queries = [
        "What is hydraulic fracturing?",
        "How to prevent blowouts?",
        "Casing design standards",
        "artificial lift methods",
        "H2S safety requirements",
    ]

    for q in test_queries:
        result = rag_engine.generate_answer(q)
        print(f"\n  Query: '{q}'")
        print(f"  Sources found: {result['num_retrieved']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Answer preview: {result['answer'][:120]}...")

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total documents: {len(all_docs)}")
    print(f"  Categories: {set(d['category'] for d in all_docs)}")
    print(f"  Models saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
