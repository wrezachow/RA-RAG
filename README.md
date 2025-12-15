# RA-RAG (Research Assistant RAG)

RA-RAG is a **fully local Retrieval-Augmented Generation (RAG) AI agent** that allows you to ask questions about research papers (PDFs) and receive answers grounded strictly in the paper’s content.

The system runs **entirely offline** using **Ollama**, **LangChain**, and **Chroma**, with PDFs converted into structured CSV text chunks that act as the source of truth.

No cloud APIs. No data leaves your machine.

---

## Features

- Research paper ingestion from **PDF**
- Clean text extraction and chunking
- CSV-based intermediate representation (transparent & inspectable)
- Semantic search using **Chroma**
- Local inference with **Ollama (llama3)**
- Fully private and offline
- Answers grounded in retrieved excerpts with citations
- Deterministic single-paper workflow

---

## Requirements

- Python **3.10+**
- [Ollama](https://ollama.com/) installed and running locally

---

## Installation

### 1️⃣ Clone the repository
```
git clone [link]
cd RA-RAG
```
### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Pull required Ollama models
```
ollama pull llama3
ollama pull mxbai-embed-large
```

## Usage
### 1️⃣ Provide a PDF

Place a research paper inside the papers/ folder (or provide any valid path).

Example:
papers/sample.pdf

### 2️⃣ Run the agent
```
python main.py
```

### 3️⃣ Follow the prompts
```
PDF path (e.g. papers/paper.pdf):
Output CSV (press Enter for default):
```
If you press Enter, the system automatically creates a CSV next to the PDF:
```
papers/ijerph-18-11382_chunks.csv
```
### 4️⃣ Ask questions
```
Ask your question:
```
RA-RAG retrieves the most relevant excerpts from the paper and generates an answer only from those excerpts, with page and chunk references.

## How It Works

1. The PDF is parsed and cleaned
2. ext is split into overlapping chunks
3. Chunks are stored in a CSV file
4. Chunks are embedded and indexed in Chroma
5. A retriever selects relevant passages
6. A local LLM generates an answer grounded in the retrieved text

### Design choice:
The CSV is the runtime data source. The PDF is only used during ingestion.

## Models Used

| Purpose | Model |
|------|------|
| Chat / reasoning | `llama3` |
| Embeddings | `mxbai-embed-large` |

Embedding models can be swapped (e.g. `nomic-embed-text`) depending on speed vs quality needs.

---

## Notes & Limitations

- First run for a paper takes longer due to embedding creation
- To re-embed a paper, delete `chroma_langchain_db/` and rerun
- Image-only / scanned PDFs require OCR (not yet implemented)

---

## Roadmap

- Multi-paper querying
- Section-aware chunking (Abstract / Methods / Results)
- Hallucination safeguards
- Incremental ingestion without DB rebuilds
- GUI / Canvas-style interface

---

## Motivation

Many RAG demos:
- hide data behind APIs
- mix ingestion and inference
- obscure how retrieval works

RA-RAG keeps everything **local, explicit, and inspectable**, making it useful for:
- academic research
- private document analysis
- learning RAG systems deeply
- portfolio and systems design work
