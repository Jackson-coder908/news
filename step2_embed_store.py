"""
Step 2: Embed news articles and store them in a local vector database (Chroma).
Goal: turn article text into searchable vectors, with recency metadata attached.

Install first (run in your Antigravity terminal):
    pip install chromadb sentence-transformers requests
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
NEWSAPI_KEY = "fa079c0d13774a38809346cdd01e3524"
QUERY = "semiconductor chips"
PAGE_SIZE = 10

# ---- SETUP ----
# Local embedding model — downloads once (~80MB), then runs offline, no API key needed.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma client — stores vectors in a local folder called "chroma_db"
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="news_articles")


def fetch_news(query: str, api_key: str, page_size: int = 10):
    """Fetch recent articles matching `query` from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["articles"]


def embed_and_store(articles):
    """Convert each article into a vector and store it in Chroma with metadata."""
    for i, article in enumerate(articles):
        # Combine title + description as the text we embed (richer signal than title alone)
        text = f"{article['title']}. {article.get('description') or ''}"

        # Skip articles with no usable text
        if not text.strip():
            continue

        # Generate the embedding vector for this article
        vector = embedder.encode(text).tolist()

        # Use the article URL as a unique ID (avoids duplicate storage on re-runs)
        doc_id = article["url"]

        collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "title": article["title"],
                "source": article["source"]["name"],
                "published_at": article["publishedAt"],
                "url": article["url"],
            }]
        )

    print(f"Stored/updated {len(articles)} articles in Chroma.")


def main():
    if NEWSAPI_KEY == "PASTE_YOUR_KEY_HERE":
        print("⚠️  Please paste your NewsAPI key into NEWSAPI_KEY before running.")
        return

    print("Fetching news...")
    articles = fetch_news(QUERY, NEWSAPI_KEY, PAGE_SIZE)

    print("Embedding and storing in Chroma...")
    embed_and_store(articles)

    print(f"\nTotal articles now in DB: {collection.count()}")


if __name__ == "__main__":
    main()
