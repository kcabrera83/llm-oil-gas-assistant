# Architecture - LLM Oil & Gas Assistant

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (HTML)                   │
│                    Port 5015 /                           │
├─────────────────────────────────────────────────────────┤
│                    Flask API Layer                       │
│         /api/ask  /api/summarize  /api/search            │
├──────────┬──────────────┬──────────────┬─────────────────┤
│ RAG      │ QA Model     │ Summarizer   │ Text Processor  │
│ Engine   │ (Extractive) │ (Extractive) │ (TF-IDF)        │
├──────────┴──────────────┴──────────────┴─────────────────┤
│              Knowledge Base (60+ documents)               │
│         Synthetic + Pre-built JSON documents              │
└─────────────────────────────────────────────────────────┘
```

## Components

### Data Layer

- **Knowledge Base**: 60+ documents covering Drilling, Safety, Equipment, Production, and Reservoir engineering
- **Data Generator**: Creates synthetic oil & gas documents with realistic terminology
- **Document Format**: JSON with title, category, content, and source fields

### Model Layer

#### RAG Engine (`RAGEngine`)
- **Algorithm**: TF-IDF vectorization with cosine similarity
- **Vectorization**: scikit-learn `TfidfVectorizer` with English stop words removed
- **Chunking**: Documents split into overlapping text chunks for better retrieval
- **Retrieval**: Cosine similarity search across all indexed chunks
- **Answer Generation**: Extractive - selects most relevant passage as answer
- **Confidence Score**: Based on cosine similarity of top retrieval results

#### QA Model (`QAModel`)
- **Algorithm**: Extractive question answering
- **Approach**: Matches query terms against document sentences, scores by overlap
- **Answer Span**: Extracts the highest-scoring sentence from retrieved documents
- **Confidence**: TF-IDF weighted term overlap between question and candidate answers

#### Summarizer (`Summarizer`)
- **Algorithm**: Extractive summarization using sentence scoring
- **Scoring**: TF-IDF weighted keyword frequency per sentence
- **Ranking**: Sentences ranked by score, top-N selected
- **Order**: Selected sentences returned in original document order

#### Text Processor (`TextProcessor`)
- **Tokenization**: Custom word tokenizer with lowercase normalization
- **TF-IDF**: Term frequency-inverse document frequency computation
- **Similarity**: Cosine similarity between TF-IDF vectors
- **Chunking**: Configurable overlap-based text chunking

### API Layer

- **Framework**: Flask (Python)
- **Serialization**: JSON request/response
- **Model Loading**: joblib deserialization at startup
- **Port**: 5015

### Dashboard Layer

- **Frontend**: HTML/CSS/JavaScript (single page)
- **Visualization**: Chart.js for statistics
- **Style**: Dark-themed responsive UI

## Data Flow

### Ask a Question (RAG Pipeline)

```
1. User Query
   ↓
2. TF-IDF Vectorization (TextProcessor)
   ↓
3. Cosine Similarity Search (RAG Engine)
   ↓
4. Top-K Document Retrieval
   ↓
5. Extractive QA Answer Generation (QAModel)
   ↓
6. Answer + Confidence + Sources
   ↓
7. Dashboard Display
```

### Summarization Pipeline

```
1. Input Text
   ↓
2. Sentence Tokenization
   ↓
3. TF-IDF Sentence Scoring
   ↓
4. Top-N Sentence Selection
   ↓
5. Summary Output (original order)
```

### Search Pipeline

```
1. Search Query
   ↓
2. TF-IDF Vectorization
   ↓
3. Cosine Similarity Search
   ↓
4. Ranked Results with Scores
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Web Framework | Flask |
| ML Library | scikit-learn |
| Numerical | NumPy |
| Data Handling | Pandas |
| Model Persistence | joblib |
| Frontend | HTML/CSS/JS + Chart.js |

## Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/rag_engine.joblib` | Serialized RAG engine with TF-IDF matrix |
| `outputs/models/qa_model.joblib` | Serialized QA model with document index |
