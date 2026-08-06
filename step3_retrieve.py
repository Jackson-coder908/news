"""
Step 3: Retrieve relevant articles from the vector DB based on a natural language question.
Goal: prove semantic search works — find articles by MEANING, not exact keyword match.

Uses the same Chroma DB and embedding model from step2_embed_store.py.
Run step2 first so there's data to search!
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ---- SETUP ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="news_articles")


def search(question: str, top_k: int = 3):
    """Embed the question, then find the most semantically similar articles."""
    query_vector = embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    return results


def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # lower = more similar

    print(f"\nTop {len(documents)} matches:\n")
    print("-" * 60)

    for doc, meta, dist in zip(documents, metadatas, distances):
        print(f"Title: {meta['title']}")
        print(f"Source: {meta['source']}")
        print(f"Published: {meta['published_at']}")
        print(f"Similarity score (lower = closer): {dist:.4f}")
        print(f"URL: {meta['url']}")
        print("-" * 60)


def main():
    question = input("Ask a question about the news: ")
    results = search(question, top_k=3)
    print_results(results)


if __name__ == "__main__":
    main()
