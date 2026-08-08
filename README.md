# Wire — Live News RAG

A full-stack Retrieval-Augmented Generation (RAG) system that fetches live news, indexes it semantically, and answers questions using only real, current articles — not a language model's memorized (and potentially outdated) knowledge.

Built as a learning project to understand how modern AI systems ground their answers in real data instead of hallucinating.

## How it works

```
NewsAPI (live articles)
      ↓
Embed with sentence-transformers (local, free)
      ↓
Store in ChromaDB (local vector database, with metadata)
      ↓
Semantic search on user's question (meaning-based, not keyword-based)
      ↓
Local LLM (llama3.2 via Ollama) generates an answer grounded ONLY in retrieved articles
      ↓
Dashboard (FastAPI + HTML/JS) — ask questions, see sourced answers
```

## Features

- **Live ingestion** — pulls current articles from NewsAPI on demand or on a schedule
- **Semantic search** — finds relevant articles by meaning, not just keyword match
- **Grounded generation** — the LLM only answers from retrieved context, and honestly says so when it doesn't have enough information, instead of guessing
- **Source attribution** — every answer links back to the real articles it was based on
- **Fully local & free** — embeddings and generation both run on-device (no paid API required)
- **Auto-refresh loop** — keeps the article database current without manual re-fetching

## Tech stack

| Layer | Tool |
|---|---|
| Data source | NewsAPI |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB |
| LLM | llama3.2 via Ollama (local) |
| Backend | FastAPI |
| Frontend | HTML / CSS / vanilla JS |

## Project structure

```
news/
├── step1_fetch_news.py      # fetch live headlines from NewsAPI
├── step2_embed_store.py     # embed articles, store in ChromaDB
├── step3_retrieve.py        # semantic search over stored articles
├── step4_generate.py        # retrieve + generate a grounded answer (CLI)
├── step5_scheduler.py       # auto-refresh loop, fetches on a timer
├── backend.py                # FastAPI server wrapping the RAG pipeline
├── static/
│   └── index.html            # dashboard frontend
├── .env                      # NewsAPI key (not committed)
└── .gitignore
```

## Setup

1. Clone the repo and install dependencies:
   ```
   python -m pip install requests chromadb sentence-transformers fastapi uvicorn python-dotenv
   ```

2. Install [Ollama](https://ollama.com/download) and pull the model:
   ```
   ollama pull llama3.2
   ```

3. Get a free [NewsAPI](https://newsapi.org/register) key, then create a `.env` file:
   ```
   NEWSAPI_KEY=your_key_here
   ```

4. Fetch and index some articles:
   ```
   python step2_embed_store.py
   ```

5. Start the backend:
   ```
   python -m uvicorn backend:app --reload
   ```

6. Open `static/index.html` in your browser and start asking questions.

## What I learned building this

- RAG can only answer from what's actually in the vector database — retrieval quality is capped by ingestion, no amount of clever prompting fixes missing data
- Semantic search finds conceptually related content even without exact keyword overlap
- Grounding the LLM in retrieved context (and instructing it to admit uncertainty) is what prevents hallucinated answers
- Small local LLMs are a real trade-off — free and private, but noticeably weaker at nuanced reasoning than hosted models

## Possible next steps

- Narrow the topic (e.g. track news on specific companies) for a more personal use case
- Store full article text instead of just title + description, for richer answers
- Add filtering by recency/source in the dashboard UI
