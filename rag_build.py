# rag_build.py — build RAG store from local docs + Finnhub news (smaller chunks, robust delete)
import os, glob
import chromadb
from chromadb.utils import embedding_functions

from rag_fetch import fetch_and_normalize_news_for_watchlist

# ---------- Config ----------
DB_PATH    = "rag_store"
COLL_NAME  = "pricepal_docs"
WATCHLIST  = ["AAPL", "MSFT", "NVDA"]   # add/remove tickers as you like
NEWS_DAYS  = 7                          # how many recent days of news to pull

client = chromadb.PersistentClient(path=DB_PATH)

def get_collection():
    return client.get_or_create_collection(
        name=COLL_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )

def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def simple_chunks(text: str, chunk_size=400, overlap=80):
    chunks, i, L = [], 0, len(text)
    while i < L:
        j = min(i + chunk_size, L)
        piece = text[i:j]
        if piece.strip():
            chunks.append(piece.strip())
        i += max(1, chunk_size - overlap)
    return chunks

def meta_from_path(path: str):
    name = os.path.basename(path)
    ticker = name.split("_")[0].upper() if "_" in name else "GEN"
    title = name.rsplit(".", 1)[0]
    return {"ticker": ticker, "title": title, "source": "local_doc", "source_path": path}

def clear_collection_safe(collection):
    try:
        existing = collection.get(include=["ids"])
        ids = existing.get("ids", []) if existing else []
        if ids:
            B = 500
            for i in range(0, len(ids), B):
                collection.delete(ids=ids[i:i+B])
    except Exception:
        try:
            client.delete_collection(COLL_NAME)
        except Exception:
            pass
        return get_collection()
    return collection

def build_local_docs():
    paths = glob.glob("docs/*.md") + glob.glob("docs/*.txt") + glob.glob("docs/*.pdf")
    records = []
    for p in paths:
        meta = meta_from_path(p)
        raw = load_text(p)
        header = f"[{meta['title']} | {meta['ticker']}] "
        for chunk in simple_chunks(raw, chunk_size=400, overlap=80):
            records.append({
                "text": header + chunk,
                "meta": meta
            })
    return records

def build_api_news():
    # returns list of {"text":..., "meta": {...}}
    return fetch_and_normalize_news_for_watchlist(WATCHLIST, days=NEWS_DAYS)

def main():
    collection = get_collection()
    collection = clear_collection_safe(collection)

    local_records = build_local_docs()
    api_records   = build_api_news()

    # Combine
    all_records = local_records + api_records

    if not all_records:
        print("No records found from docs or API. Add docs to ./docs or check your API key.")
        return

    # Prepare for upsert
    ids, docs, metas = [], [], []
    for i, r in enumerate(all_records):
        ids.append(f"rec-{i}")
        docs.append(r["text"])
        metas.append(r["meta"])

    collection.add(ids=ids, documents=docs, metadatas=metas)

    # Summary
    n_local = len(local_records)
    n_api   = len(api_records)
    print(f"Indexed {len(all_records)} records "
          f"(docs={n_local}, api_news={n_api}) into '{COLL_NAME}'. DB: {DB_PATH}")

if __name__ == "__main__":
    main()
