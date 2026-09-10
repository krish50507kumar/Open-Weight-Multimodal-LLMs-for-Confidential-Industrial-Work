# SIH 26117 — Sovereign On-Premise Agentic AI Workbench

**Organisation:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Category:** Software · Smart India Hackathon

---

## Overview

A fully on-premise, air-gap-compatible AI workbench that lets industrial staff query **confidential internal documents** using open-weight LLMs — no data ever leaves the local server.

## Architecture

```
User (Browser)
     │
     ▼
Streamlit UI (app.py)
     │
     ├─► Ingest pipeline:  PDF → extract → chunk → embed → ChromaDB
     │
     └─► Query pipeline:   Question → Planner Agent → Researcher Agent
                                      → ChromaDB (semantic search)
                                      → Answer Agent (Ollama LLM)
                                      ↓
                               Audit Log (local JSONL)
```

## Features

| Feature | Status |
|---|---|
| PDF upload & ingestion (chunked, embedded) | ✅ |
| Semantic search with ChromaDB | ✅ |
| Multi-agent pipeline (Planner → Researcher → Answerer) | ✅ |
| Chat with conversation history | ✅ |
| Image/vision support (multimodal LLM) | ✅ |
| Model selection (any Ollama model) | ✅ |
| Audit trail (local JSONL) | ✅ |
| 100% on-premise, zero cloud calls | ✅ |
| Web UI (Streamlit) | ✅ |

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install & start Ollama

Download from https://ollama.com and then pull a model:

```bash
ollama pull gemma2:2b        # lightweight text model
ollama pull gemma3:4b        # vision + text
ollama pull llava:7b         # vision model
```

### 3. Run the web app

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

### 4. Ingest documents

- Upload PDFs via the **sidebar → Upload Documents**
- Click **Ingest PDFs**

### 5. Ask questions

Type in the chat box. The three-agent pipeline (Planner, Researcher, Answerer) will retrieve relevant chunks and answer from your documents.

### 6. Image analysis

Upload an image in **Upload Image (Vision)** in the sidebar, then ask a question about it. Requires a vision-capable model (`llava`, `gemma3:4b`, etc.)

---

## CLI Usage (original)

You can still use the original CLI:

```bash
# Ingest PDFs (put PDFs in data/raw_pdfs/ first)
python ingest.py

# Chat via CLI
python main.py
```

## Data Privacy

- All data stored in `data/` folder on your machine
- ChromaDB vector store: `data/chroma_db/`
- PDFs: `data/raw_pdfs/`
- Audit log: `data/audit_log.jsonl`
- **No telemetry. No external API calls.**

## Project Structure

```
SIH/
├── app.py              ← Streamlit web UI (NEW)
├── main.py             ← CLI entry point
├── ingest.py           ← PDF ingestion CLI
├── requirements.txt    ← All dependencies
├── README.md
├── src/
│   ├── rag.py          ← Multi-agent RAG pipeline
│   ├── chat.py         ← Ollama LLM wrapper
│   ├── embed.py        ← Sentence-transformers embeddings
│   ├── extract.py      ← PDF text extraction
│   ├── chunker.py      ← Text chunking
│   ├── vectorstore.py  ← ChromaDB interface
│   └── instructions.py ← System prompt
└── data/
    ├── raw_pdfs/       ← Uploaded PDFs
    ├── chroma_db/      ← Vector store (auto-created)
    └── audit_log.jsonl ← Audit trail (auto-created)
```
