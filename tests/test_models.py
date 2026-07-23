import pytest
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(OUTPUT_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_rag_engine_loads():
    import joblib
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found - run train.py first")
    rag = joblib.load(path)
    assert rag is not None


def test_qa_model_loads():
    import joblib
    path = os.path.join(OUTPUT_DIR, "qa_model.joblib")
    if not os.path.exists(path):
        pytest.skip("qa_model.joblib not found - run train.py first")
    qa = joblib.load(path)
    assert qa is not None


def test_rag_engine_has_documents():
    import joblib
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found")
    rag = joblib.load(path)
    assert hasattr(rag, "is_ready")
    assert rag.is_ready is True


def test_rag_retrieval():
    import joblib
    path = os.path.join(OUTPUT_DIR, "rag_engine.joblib")
    if not os.path.exists(path):
        pytest.skip("rag_engine.joblib not found")
    rag = joblib.load(path)
    results = rag.retrieve("drilling mud weight", top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
