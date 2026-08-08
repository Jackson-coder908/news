"""
Backend API for the News RAG dashboard.
Wraps the retrieve + generate logic (from step3/step4) behind a simple web API
so a frontend can call it.

Install: pip install fastapi uvicorn
Run:     uvicorn backend:app --reload
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="News RAG API")

# Allow the frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- SETUP (same as step3/step4) ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="news_articles")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


class QuestionRequest(BaseModel):
    question: str


def retrieve(question: str, top_k: int = 3):
    query_vector = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def build_prompt(question: str, documents: list, metadatas: list):
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        context_blocks.append(
            f"- {doc} (Source: {meta['source']}, Published: {meta['published_at']})"
        )
    context = "\n".join(context_blocks)

    return f"""You are a news assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer, say so honestly instead of guessing.
Always mention which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def ask_ollama(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


@app.get("/status")
def status():
    """Quick health check + article count, used by the dashboard header."""
    return {"total_articles": collection.count()}


@app.post("/ask")
def ask(req: QuestionRequest):
    """Full RAG pipeline: retrieve relevant articles, generate a grounded answer."""
    documents, metadatas = retrieve(req.question, top_k=3)

    if not documents:
        return {"answer": "No articles found in the database yet.", "sources": []}

    prompt = build_prompt(req.question, documents, metadatas)
    answer = ask_ollama(prompt)

    sources = [
        {"title": m["title"], "source": m["source"], "published_at": m["published_at"], "url": m["url"]}
        for m in metadatas
    ]

    return {"answer": answer, "sources": sources}
