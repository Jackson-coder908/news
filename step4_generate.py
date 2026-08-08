"""
Step 4: Generate a real answer using retrieved articles + a local LLM (via Ollama).
This completes the full RAG loop: Ingest -> Embed & Index -> Retrieve -> Augment & Generate.

Requires:
    - Ollama installed and running (ollama pull llama3.2)
    - Steps 2 already run at least once (so Chroma has data to search)
    pip install requests chromadb sentence-transformers python-dotenv
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ---- SETUP ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="news_articles")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def retrieve(question: str, top_k: int = 3):
    """Same retrieval logic as Step 3 - find the most relevant articles."""
    query_vector = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["documents"][0], results["metadatas"][0]


def build_prompt(question: str, documents: list, metadatas: list):
    """Combine the retrieved articles into a grounded prompt for the LLM."""
    context_blocks = []
    for doc, meta in zip(documents, metadatas):
        context_blocks.append(
            f"- {doc} (Source: {meta['source']}, Published: {meta['published_at']})"
        )
    context = "\n".join(context_blocks)

    prompt = f"""You are a news assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer, say so honestly instead of guessing.
Always mention which source(s) you used.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask_ollama(prompt: str):
    """Send the prompt to the local Ollama model and get a response."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    question = input("Ask a question about the news: ")

    print("\nRetrieving relevant articles...")
    documents, metadatas = retrieve(question, top_k=3)

    print("Building prompt and asking local LLM (this may take a moment)...\n")
    prompt = build_prompt(question, documents, metadatas)
    answer = ask_ollama(prompt)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
