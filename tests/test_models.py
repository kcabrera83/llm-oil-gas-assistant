import pytest
import joblib
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_rag_engine_loads():
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found - run train.py first")
    rag = joblib.load(path)
    assert rag is not None


def test_qa_model_loads():
    path = os.path.join(OUTPUT_DIR, "qa_model.joblib")
    if not os.path.exists(path):
        pytest.skip("qa_model.joblib not found - run train.py first")
    qa = joblib.load(path)
    assert qa is not None


def test_rag_engine_has_documents():
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found")
    rag = joblib.load(path)
    assert hasattr(rag, "is_ready")
    assert rag.is_ready is True


def test_rag_retrieval():
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found")
    rag = joblib.load(path)
    results = rag.retrieve("drilling mud weight", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0


def test_rag_generate_answer():
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found")
    rag = joblib.load(path)
    result = rag.generate_answer("What is blowout prevention?")
    assert "answer" in result
    assert "confidence" in result
    assert "num_retrieved" in result


def test_qa_extract_answer():
    path = os.path.join(OUTPUT_DIR, "qa_model.joblib")
    if not os.path.exists(path):
        pytest.skip("qa_model.joblib not found")
    qa = joblib.load(path)
    result = qa.extract_answer("What are casing design standards?", top_k=3)
    assert "answer" in result
    assert "confidence" in result
