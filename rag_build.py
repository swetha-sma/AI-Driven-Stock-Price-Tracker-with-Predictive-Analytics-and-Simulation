# rag_build.py — Build RAG store from live Finnhub news only (no local docs)
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from rag_fetch import fetch_and_normalize_news_for_watchlist

DB_PATH   = "rag_store"
COLL_NAME = "pricepal_docs"

client = chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    return client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

def clear_ticker(collection, tickers):
    try:
        collection.delete(where={"ticker": {"$in": [t.upper() for t in tickers]}})
    except Exception:
        pass

def build_index(watchlist=None, days=7):
    """
    Fetch & index ONLY API news (no local documents).
    watchlist = ["AAPL"] or ["TSLA","AMZN",...]
    """
    watchlist = [t.strip().upper() for t in (watchlist or []) if t.strip()]
    if not watchlist:
        return {"docs_indexed": 0, "tickers": []}

    collection = get_collection()

    # Fetch news from API
    api_records = fetch_and_normalize_news_for_watchlist(watchlist, days=days)
    if not api_records:
        return {"docs_indexed": 0, "tickers": watchlist}

    # Remove past records for this ticker
    clear_ticker(collection, watchlist)

    # Insert fresh news chunks
    ids = [f"rec-{i}" for i in range(len(api_records))]
    docs = [r["text"] for r in api_records]
    metas = [r["meta"] for r in api_records]

    collection.add(ids=ids, documents=docs, metadatas=metas)

    return {
        "collection": COLL_NAME,
        "docs_indexed": len(api_records),
        "tickers": watchlist,
        "when": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    print(build_index(["AAPL", "MSFT", "NVDA"], days=7))
