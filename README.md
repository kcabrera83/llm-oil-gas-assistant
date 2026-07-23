# LLM Oil & Gas Assistant

An LLM-powered Q&A assistant for oil & gas documentation using retrieval-augmented generation (RAG).

## Overview

This project provides an intelligent question-answering system for oil and gas industry documentation. It uses TF-IDF vectorization and cosine similarity for document retrieval, extractive QA for answer generation, and extractive summarization for content summarization.

## Architecture

The system follows a Retrieval-Augmented Generation (RAG) pattern:

1. **Knowledge Base**: 60+ pre-built documents covering Drilling, Safety, Equipment, Production, and Reservoir engineering topics.
2. **Text Processor**: TF-IDF vectorization with cosine similarity search for document retrieval.
3. **RAG Engine**: Retrieves relevant documents and generates answers from the knowledge base.
4. **QA Model**: Extractive question-answering that finds answer spans within retrieved documents.
5. **Summarizer**: Extractive text summarization using sentence scoring.

## Project Structure

```
llm-oil-gas-assistant/
  llm_assistant/
    __init__.py
    data_generator.py          # Synthetic knowledge base generator
    knowledge_base/
      documents.json           # Pre-built knowledge base (60+ topics)
    models/
      __init__.py
      rag_engine.py            # RAG engine (TF-IDF + retrieval)
      qa_model.py              # Extractive QA model
      summarizer.py            # Text summarizer
    utils/
      __init__.py
      text_processor.py        # Text chunking, TF-IDF, similarity
  outputs/models/              # Trained model artifacts
  templates/
    index.html                 # Web interface
  .github/workflows/
    ci.yml                     # CI/CD pipeline
  train.py                     # Training script
  app.py                       # Flask API server
  test_api.py                  # API tests
  requirements.txt
  setup.py
  README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Knowledge Base

```bash
python train.py
```

This builds the TF-IDF index from the knowledge base and saves model artifacts to `outputs/models/`.

### 2. Start the API Server

```bash
python app.py
```

The server starts on port 5015.

### 3. Open the Web Interface

Navigate to `http://localhost:5015` in your browser.

### 4. Run Tests

```bash
python test_api.py
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/ask` | POST | Ask a question (body: `{"query": "...", "top_k": 5}`) |
| `/api/summarize` | POST | Summarize text (body: `{"text": "...", "num_sentences": 3}`) |
| `/api/search` | POST | Search documents (body: `{"query": "...", "top_k": 10}`) |
| `/api/models` | GET | List loaded models and stats |
| `/api/health` | GET | Health check |

## Sample Questions

- What is rotary drilling and how does it work?
- What are the OSHA confined space entry requirements?
- How do electric submersible pumps operate?
- What are the primary recovery mechanisms in reservoir engineering?
- How to prevent blowouts during drilling operations?
- What is process safety management (PSM)?

## Topics Covered

- **Drilling**: Rotary drilling, drilling fluids, wellbore stability, directional drilling, casing design, well control, bit selection, well abandonment
- **Safety**: PPE requirements, hot work permits, confined space entry, hazardous area classification, PSM, fall protection, environmental compliance, oil spill response
- **Equipment**: Centrifugal pumps, heat exchangers, compressors, pressure vessels, ESP systems, rod pumping, pipelines, separators, offshore platforms
- **Production**: Primary/EOR recovery, well testing, flow assurance, artificial lift, gas processing, reservoir characterization, workovers, gas lift, LNG
- **Reservoir**: Reservoir engineering, multiphase flow, stimulation design, formation damage, EOR screening, seismic methods

## Technology Stack

- Python 3.8+
- Flask (web framework)
- scikit-learn (TF-IDF vectorization)
- NumPy (numerical operations)
- Pandas (data handling)
- joblib (model persistence)

## Elaborado por Ing. Kelvin Cabrera
