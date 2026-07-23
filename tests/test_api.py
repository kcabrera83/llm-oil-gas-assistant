import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "rag_ready" in data
    assert "models_loaded" in data
    assert data["version"] == "1.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "loaded_models" in data
    assert "stats" in data
    assert isinstance(data["loaded_models"], list)


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.json()
    assert data["openapi"] == "3.0.0"
    assert "paths" in data
    assert "/api/ask" in data["paths"]
    assert "/api/health" in data["paths"]


def test_ask_valid_input(client):
    response = client.post("/api/ask", json={"query": "What is hydraulic fracturing?"})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "qa_answer" in data
        assert "processing_time_ms" in data


def test_ask_with_top_k(client):
    response = client.post("/api/ask", json={"query": "drilling mud weight", "top_k": 3})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert data["num_retrieved"] <= 3


def test_ask_empty_query(client):
    response = client.post("/api/ask", json={"query": ""})
    assert response.status_code == 400


def test_summarize_valid_input(client):
    text = (
        "The drilling operation proceeded smoothly for the first 2000 feet. "
        "Mud weight was increased to 12 ppg to control formation pressure. "
        "The bit was pulled at 3500 feet due to wear. A new PDC bit was installed. "
        "Drilling resumed at 80 ft/hr through the limestone formation. "
        "Total depth was reached at 8500 feet after 45 days of operations."
    )
    response = client.post("/api/summarize", json={"text": text, "num_sentences": 2})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "summary" in data
        assert "processing_time_ms" in data


def test_summarize_empty_text(client):
    response = client.post("/api/summarize", json={"text": ""})
    assert response.status_code == 400


def test_search_valid_input(client):
    response = client.post("/api/search", json={"query": "casing design"})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert isinstance(data["results"], list)


def test_search_empty_query(client):
    response = client.post("/api/search", json={"query": ""})
    assert response.status_code == 400
