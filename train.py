"""Training script - builds FAISS vector store index and saves to outputs/models/."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from llm_assistant.data_generator import generate_knowledge_base


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_assistant", "knowledge_base", "documents.json")
FAISS_INDEX_DIR = os.path.join(OUTPUT_DIR, "faiss_index")


def build_rag_pipeline(all_docs):
    """Build RAG pipeline with LangChain + FAISS"""
    documents = []
    for doc in all_docs:
        page_content = doc.get("content", "")
        metadata = {
            "title": doc.get("title", ""),
            "category": doc.get("category", ""),
            "source": doc.get("source", ""),
            "id": doc.get("id", 0),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, embeddings, len(docs)


def search_similar(query, vectorstore, k=5):
    """Search for similar documents using FAISS"""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def main():
    print("=" * 60)
    print("  LLM Oil & Gas Assistant - Training Pipeline")
    print("  Elaborado por Ing. Kelvin Cabrera")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate synthetic knowledge base
    print("\n[1/4] Generating synthetic oil & gas knowledge base...")
    t0 = time.time()
    synthetic_docs = generate_knowledge_base()
    print(f"  Generated {len(synthetic_docs)} synthetic documents in {time.time() - t0:.2f}s")

    # Step 2: Load pre-built knowledge base
    print("\n[2/4] Loading pre-built knowledge base...")
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
        entry = {"title": doc["title"], "category": doc["category"], "content": doc["content"], "source": doc["source"]}
        entry["id"] = len(all_docs) + 1
        all_docs.append(entry)
    for doc in kb_docs:
        entry = {"title": doc["title"], "category": doc["category"], "content": doc["content"], "source": doc["source"]}
        if "id" not in doc:
            entry["id"] = len(all_docs) + 1
        else:
            entry["id"] = doc["id"]
        all_docs.append(entry)

    print(f"  Total documents in merged knowledge base: {len(all_docs)}")

    # Step 3: Build FAISS vector store
    print("\n[3/4] Building FAISS vector store (HuggingFace embeddings)...")
    t0 = time.time()
    vectorstore, embeddings, num_chunks = build_rag_pipeline(all_docs)
    print(f"  Vector store built in {time.time() - t0:.2f}s")
    print(f"  - Documents indexed: {len(all_docs)}")
    print(f"  - Text chunks created: {num_chunks}")

    # Step 4: Save FAISS index
    print("\n[4/4] Saving FAISS index to disk...")
    t0 = time.time()
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    print(f"  FAISS index saved in {time.time() - t0:.2f}s")
    print(f"  Output directory: {FAISS_INDEX_DIR}")

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
        results = search_similar(q, vectorstore, k=3)
        print(f"\n  Query: '{q}'")
        print(f"  Results found: {len(results)}")
        for doc, score in results:
            print(f"    - [{score:.4f}] {doc.metadata.get('title', 'N/A')}")

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Total documents: {len(all_docs)}")
    print(f"  Categories: {set(d['category'] for d in all_docs)}")
    print(f"  FAISS index saved to: {FAISS_INDEX_DIR}")
    print()


if __name__ == "__main__":
    main()
