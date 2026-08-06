"""
Step 2: Embed news articles and store them in a local vector database (Chroma).
Goal: turn article text into searchable vectors, with recency metadata attached.

Install first (run in your Antigravity terminal):
    pip install chromadb sentence-transformers requests python-dotenv
"""

import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
QUERY = "semiconductor chips"
PAGE_SIZE = 10

# ---- SETUP ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        text = f"{article['title']}. {article.get('description') or ''}"

        if not text.strip():
            continue

        vector = embedder.encode(text).tolist()
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
    if not NEWSAPI_KEY:
        print("please check .env file with NEWSAPI_KEY=your_key")
        return

    print("Fetching news...")
    articles = fetch_news(QUERY, NEWSAPI_KEY, PAGE_SIZE)

    print("Embedding and storing in Chroma...")
    embed_and_store(articles)

    print(f"\nTotal articles now in DB: {collection.count()}")


if __name__ == "__main__":
    main()
