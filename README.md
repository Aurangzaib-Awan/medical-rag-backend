# Medical RAG Backend

A **Retrieval-Augmented Generation (RAG)** backend for medical question answering, built with LangChain, a local vector store, and a FastAPI/Streamlit interface. Upload medical documents, embed them into a vector index, and ask questions grounded in your own knowledge base.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)

---

## Overview

This project implements a RAG pipeline tailored for medical documents. Instead of relying on a general-purpose LLM that may hallucinate clinical details, the system retrieves relevant chunks from your document corpus first, then uses the LLM to generate a grounded answer.

**Core flow:**
```
Medical PDF/Text → Chunking → Embeddings → Vector Store → Retrieval → LLM → Answer
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  User Query                 │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Query Embedding   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Vector Store Search │  ◄── my_langchain_index/
          │  (Similarity Search) │
          └──────────┬──────────┘
                     │ Top-K Chunks
          ┌──────────▼──────────┐
          │    LLM (with RAG    │
          │    context prompt)  │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Grounded Answer   │
          └─────────────────────┘
```

---

## Project Structure

```
medical-rag-backend/
│
├── app.py                  # Streamlit / API entry point — user-facing interface
├── main.py                 # Core RAG chain setup and query handling
├── vector.py               # Document loading, chunking, embedding & index creation
├── requirements.txt        # Python dependencies
├── my_langchain_index/     # Persisted LangChain vector store (FAISS/Chroma index)
└── .gitignore
```

| File | Responsibility |
|------|---------------|
| `vector.py` | Loads documents, splits into chunks, generates embeddings, saves the vector index |
| `main.py` | Loads the persisted index, builds the retrieval chain, handles Q&A logic |
| `app.py` | Frontend interface (Streamlit) — takes user input, calls the chain, displays results |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- A Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

### Install dependencies

```bash
git clone https://github.com/Aurangzaib-Awan/medical-rag-backend.git
cd medical-rag-backend
pip install -r requirements.txt
```

### Set environment variables

```bash
export GOOGLE_API_KEY=your_api_key_here
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

---

## Usage

### Step 1 — Add your medical documents

Place your PDF or text files in the project root (or a `docs/` folder), then build the vector index:
```bash
python vector.py
```

This will populate the `my_langchain_index/` directory with the persisted embeddings.

### Step 2 — Launch the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, type a medical question, and get a retrieval-grounded answer.

---

## How It Works

1. **Ingestion (`vector.py`)** — Medical documents (PDFs or text) are loaded, split into overlapping chunks using LangChain's text splitters, and embedded using an embedding model (e.g., OpenAI `text-embedding-ada-002`). The resulting vectors are saved to a local FAISS/Chroma index under `my_langchain_index/`.

2. **Retrieval (`main.py`)** — At query time, the user's question is embedded and matched against stored vectors via similarity search. The top-K most relevant chunks are retrieved.

3. **Generation (`main.py`)** — Retrieved chunks are injected into a prompt template alongside the user's question and passed to the LLM. The model generates an answer grounded strictly in the retrieved context — reducing hallucination on clinical content.

4. **Interface (`app.py`)** — A Streamlit UI ties everything together, providing a simple chat-like experience for end users.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)
![Gemma](https://img.shields.io/badge/Gemma-4285F4?style=flat&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-blue?style=flat)

---

## Author

**Aurangzaib Shehzad Awan**  
[GitHub](https://github.com/Aurangzaib-Awan)
