# User Guide - LLM Oil & Gas Assistant

## Overview

The LLM Oil & Gas Assistant is an intelligent question-answering system for oil and gas industry documentation. It uses retrieval-augmented generation (RAG) with TF-IDF vectorization to search a 60+ document knowledge base and generate answers using extractive QA.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd llm-oil-gas-assistant
pip install -r requirements.txt
```

### Train the Knowledge Base

```bash
python train.py
```

This generates a synthetic knowledge base and builds TF-IDF indices. Models are saved to `outputs/models/`.

### Start the API Server

```bash
python app.py
```

The server starts on `http://localhost:5015`.

### Open the Dashboard

Navigate to `http://localhost:5015` in your browser to access the web interface.

### Run Tests

```bash
python test_api.py
```

## Dashboard Features

- **Ask Questions**: Enter natural language questions about oil & gas topics
- **Knowledge Base Search**: Search documents by keyword or phrase
- **Text Summarization**: Paste text to get an extractive summary
- **Model Statistics**: View loaded models and knowledge base stats

## Knowledge Base Topics

| Category | Topics |
|----------|--------|
| Drilling | Rotary drilling, drilling fluids, wellbore stability, directional drilling, casing design, well control, bit selection, well abandonment |
| Safety | PPE requirements, hot work permits, confined space entry, hazardous area classification, PSM, fall protection, environmental compliance, oil spill response |
| Equipment | Centrifugal pumps, heat exchangers, compressors, pressure vessels, ESP systems, rod pumping, pipelines, separators, offshore platforms |
| Production | Primary/EOR recovery, well testing, flow assurance, artificial lift, gas processing, reservoir characterization, workovers, gas lift, LNG |
| Reservoir | Reservoir engineering, multiphase flow, stimulation design, formation damage, EOR screening, seismic methods |

## Sample Questions

- What is rotary drilling and how does it work?
- What are the OSHA confined space entry requirements?
- How do electric submersible pumps operate?
- What are the primary recovery mechanisms in reservoir engineering?
- How to prevent blowouts during drilling operations?
- What is process safety management (PSM)?

## API Usage

### Using curl

**Ask a question:**
```bash
curl -X POST http://localhost:5015/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hydraulic fracturing?", "top_k": 5}'
```

**Summarize text:**
```bash
curl -X POST http://localhost:5015/api/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your long text here...", "num_sentences": 3}'
```

**Search knowledge base:**
```bash
curl -X POST http://localhost:5015/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "well control blowout prevention", "top_k": 10}'
```

### Using Python

```python
import requests

# Ask a question
response = requests.post("http://localhost:5015/api/ask", json={
    "query": "What is hydraulic fracturing?",
    "top_k": 5
})
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")

# Summarize text
response = requests.post("http://localhost:5015/api/summarize", json={
    "text": "Your long text here...",
    "num_sentences": 3
})
print(response.json()["summary"])

# Search knowledge base
response = requests.post("http://localhost:5015/api/search", json={
    "query": "casing design standards",
    "top_k": 5
})
for doc in response.json()["results"]:
    print(f"- {doc['title']} (score: {doc['score']:.4f})")
```

### Using JavaScript

```javascript
const response = await fetch("http://localhost:5015/api/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What is hydraulic fracturing?" })
});
const data = await response.json();
console.log(data.answer);
```
