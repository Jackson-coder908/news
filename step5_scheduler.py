"""
Step 5: Auto-refresh - keep fetching new articles automatically on a timer.
This is what makes the project actually "real-time" instead of a one-time script.

Runs step2's fetch+embed logic every N minutes, forever, until you stop it (Ctrl+C).
"""

import os
import time
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
QUERIES = ["technology", "semiconductor chips"]  # topics to keep refreshing
PAGE_SIZE = 10
REFRESH_INTERVAL_MINUTES = 10  # how often to fetch new articles

# ---- SETUP ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="news_articles")


def fetch_news(query: str, api_key: str, page_size: int = 10):
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
    count = 0
    for article in articles:
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
        count += 1
    return count


def run_refresh_cycle():
    """Fetch + store for every query in QUERIES, once."""
    total_new = 0
    for query in QUERIES:
        try:
            articles = fetch_news(query, NEWSAPI_KEY, PAGE_SIZE)
            stored = embed_and_store(articles)
            total_new += stored
            print(f"  [{query}] stored/updated {stored} articles")
        except Exception as e:
            print(f"  [{query}] fetch failed: {e}")

    print(f"Cycle complete. Total articles in DB: {collection.count()}")


def main():
    if not NEWSAPI_KEY:
        print("please check .env file with NEWSAPI_KEY=your_key")
        return

    print(f"Starting auto-refresh loop (every {REFRESH_INTERVAL_MINUTES} min). Press Ctrl+C to stop.\n")

    while True:
        print(f"--- Refresh cycle: {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        run_refresh_cycle()
        print(f"Sleeping for {REFRESH_INTERVAL_MINUTES} minutes...\n")
        time.sleep(REFRESH_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
