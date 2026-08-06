"""
Step 1: Fetch live news headlines from NewsAPI.
Goal: prove we can pull real, current articles before touching any AI/embedding stuff.
"""

import requests

# ---- CONFIG ----
NEWSAPI_KEY = "fa079c0d13774a38809346cdd01e3524"   # from https://newsapi.org
QUERY = "technology"                   # topic to search for
PAGE_SIZE = 10                         # how many articles to fetch

def fetch_news(query: str, api_key: str, page_size: int = 10):
    """Fetch recent articles matching `query` from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",   # newest first
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # crash loudly if something's wrong (bad key, rate limit, etc.)

    data = response.json()
    return data["articles"]


def main():
    if NEWSAPI_KEY == "PASTE_YOUR_KEY_HERE":
        print("⚠️  Please paste your NewsAPI key into NEWSAPI_KEY before running.")
        return

    articles = fetch_news(QUERY, NEWSAPI_KEY, PAGE_SIZE)

    print(f"\nFetched {len(articles)} articles for query: '{QUERY}'\n")
    print("-" * 60)

    for i, article in enumerate(articles, start=1):
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']['name']}")
        print(f"   Published: {article['publishedAt']}")
        print(f"   URL: {article['url']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
