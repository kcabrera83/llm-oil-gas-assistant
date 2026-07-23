# API Documentation - LLM Oil & Gas Assistant

## Base URL

```
http://localhost:5015
```

## Endpoints

### GET /

Serves the web interface (HTML dashboard).

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "rag_ready": true,
  "models_loaded": ["RAG Engine (TF-IDF)", "QA Model (Extractive)", "Summarizer (Extractive)"],
  "version": "1.0.0"
}
```

---

### GET /api/models

Returns loaded model information and knowledge base statistics.

**Response:**
```json
{
  "loaded_models": ["RAG Engine (TF-IDF)", "QA Model (Extractive)", "Summarizer (Extractive)"],
  "stats": {
    "total_documents": 60,
    "total_chunks": 300,
    "vocabulary_size": 4500,
    "qa_model": {
      "documents_loaded": 60
    }
  }
}
```

---

### POST /api/ask

Ask a question using RAG + extractive QA models. Retrieves relevant documents and generates answers.

**Request:**
```json
{
  "query": "What is hydraulic fracturing?",
  "top_k": 5
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | The question to ask |
| top_k | integer | No | 5 | Number of documents to retrieve |

**Response:**
```json
{
  "query": "What is hydraulic fracturing?",
  "answer": "Hydraulic fracturing is a well stimulation technique in which rock is fractured by a pressurized liquid...",
  "confidence": 0.8742,
  "sources": ["Drilling & Completion", "Reservoir Engineering", "Production Operations"],
  "num_retrieved": 5,
  "qa_answer": "Hydraulic fracturing is a well stimulation technique in which rock is fractured by a pressurized liquid...",
  "qa_confidence": 0.9123,
  "processing_time_ms": 45.23
}
```

**Error Response:**
```json
{"error": "Query is required"}
```
Status: `400 Bad Request`

---

### POST /api/summarize

Summarize a given text using extractive summarization.

**Request:**
```json
{
  "text": "Rotary drilling is the most common method of drilling wells today. The drill string consists of the drill pipe, drill collars, and the drill bit. The rotary table provides the rotational force needed to turn the drill string...",
  "num_sentences": 3
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| text | string | Yes | - | Text to summarize |
| num_sentences | integer | No | 3 | Number of sentences in summary |

**Response:**
```json
{
  "summary": "Rotary drilling is the most common method of drilling wells today. The drill string consists of the drill pipe, drill collars, and the drill bit. The rotary table provides the rotational force.",
  "num_sentences": 3,
  "original_length": 350,
  "processing_time_ms": 12.56
}
```

**Error Response:**
```json
{"error": "Text is required"}
```
Status: `400 Bad Request`

---

### POST /api/search

Search the knowledge base for documents matching a query.

**Request:**
```json
{
  "query": "well control blowout prevention",
  "top_k": 10
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | Yes | - | Search query |
| top_k | integer | No | 10 | Maximum results to return |

**Response:**
```json
{
  "query": "well control blowout prevention",
  "results": [
    {
      "title": "Well Control & Blowout Prevention",
      "category": "Safety",
      "content": "Well control refers to the regulation of...",
      "score": 0.8234
    }
  ],
  "total_results": 10,
  "processing_time_ms": 23.11
}
```

**Error Response:**
```json
{"error": "Query is required"}
```
Status: `400 Bad Request`

---

### GET /api/docs

Returns OpenAPI 3.0.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "LLM Oil & Gas Assistant", "version": "1.0.0"},
  "paths": {
    "/api/health": {"get": {"summary": "Health check"}},
    "/api/models": {"get": {"summary": "Model info"}},
    "/api/ask": {"post": {"summary": "Ask a question using RAG + QA models"}},
    "/api/summarize": {"post": {"summary": "Summarize text"}},
    "/api/search": {"post": {"summary": "Search knowledge base"}}
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid parameters |
| 500 | Internal server error |
