"""API tests for LLM Oil & Gas Assistant."""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app, load_models


def get_test_client():
    load_models()
    return TestClient(app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    print("[PASS] /api/health")
    return data


def test_models(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "loaded_models" in data
    assert "stats" in data
    print(f"[PASS] /api/models - Models: {data['loaded_models']}")
    return data


def test_ask_valid(client):
    payload = {"query": "What is rotary drilling?"}
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "confidence" in data
    assert "sources" in data
    assert data["confidence"] > 0
    print(f"[PASS] /api/ask (valid) - Confidence: {data['confidence']:.4f}")
    return data


def test_ask_no_query(client):
    resp = client.post("/api/ask", json={"query": ""})
    assert resp.status_code == 400
    print("[PASS] /api/ask (empty query)")
    return resp.json()


def test_ask_safety(client):
    payload = {"query": "What are the OSHA confined space requirements?"}
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence"] > 0
    print(f"[PASS] /api/ask (safety) - Confidence: {data['confidence']:.4f}")
    return data


def test_search(client):
    payload = {"query": "blowout preventer BOP well control", "top_k": 5}
    resp = client.post("/api/search", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) > 0
    print(f"[PASS] /api/search - Found {data['total_results']} results")
    return data


def test_search_no_query(client):
    resp = client.post("/api/search", json={"query": ""})
    assert resp.status_code == 400
    print("[PASS] /api/search (empty query)")
    return resp.json()


def test_summarize(client):
    text = (
        "Rotary drilling is the most common method used to drill oil and gas wells. "
        "The process involves rotating a drill bit at the bottom of the hole while drilling "
        "fluid (mud) is pumped down through the drill string to cool the bit and carry "
        "cuttings to the surface. The rotary table or top drive provides the rotational "
        "force. Typical penetration rates range from 10 to 50 feet per hour depending "
        "on formation hardness. The weight on bit (WOB) is controlled by the number of "
        "drill collars in the bottom hole assembly. A standard BHA includes drill collars, "
        "stabilizers, and the drill bit."
    )
    payload = {"text": text, "num_sentences": 2}
    resp = client.post("/api/summarize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    print(f"[PASS] /api_summarize - Summary length: {len(data['summary'])} chars")
    return data


def test_summarize_no_text(client):
    resp = client.post("/api/summarize", json={"text": ""})
    assert resp.status_code == 400
    print("[PASS] /api/summarize (empty text)")
    return resp.json()


def test_ask_equipment(client):
    payload = {"query": "How does an electric submersible pump work?"}
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence"] > 0
    print(f"[PASS] /api/ask (equipment) - Confidence: {data['confidence']:.4f}")
    return data


def test_ask_production(client):
    payload = {"query": "What are the methods for enhanced oil recovery?"}
    resp = client.post("/api/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence"] > 0
    print(f"[PASS] /api/ask (production) - Confidence: {data['confidence']:.4f}")
    return data


def main():
    print("=" * 60)
    print("  LLM Oil & Gas Assistant - API Tests")
    print("  Elaborado por Ing. Kelvin Cabrera")
    print("=" * 60)

    client = get_test_client()

    tests = [
        test_health,
        test_models,
        test_ask_valid,
        test_ask_no_query,
        test_ask_safety,
        test_search,
        test_search_no_query,
        test_summarize,
        test_summarize_no_text,
        test_ask_equipment,
        test_ask_production,
    ]

    passed = 0
    failed = 0
    t0 = time.time()

    for test_fn in tests:
        try:
            test_fn(client)
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            failed += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"  Time: {elapsed:.2f}s")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
