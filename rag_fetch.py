# rag_fetch.py — fetch & normalize API data for RAG (Finnhub company news)
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
import finnhub

load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing FINNHUB_API_KEY in .env")

_client = finnhub.Client(api_key=API_KEY)

# --------- utilities ---------
def _utc_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _clean_text(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split())

# --------- fetchers ----------
def fetch_company_news(ticker: str, days: int = 7) -> List[Dict[str, Any]]:
    """
    Get recent news/PR for a ticker from Finnhub.
    Returns raw items with fields: datetime (unix), headline, summary, source, url, id.
    """
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=max(1, days))
    data = _client.company_news(ticker.upper(), _utc_date_str(frm), _utc_date_str(to))
    # Finnhub returns a list; free tier is enough for a few tickers
    return data or []

def normalize_news_items(ticker: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn raw news into text + metadata dictionaries ready for RAG indexing.
    Output item:
      {
        "text": "Apple Q4: EPS beat ... (short)",
        "meta": {
            "ticker": "AAPL",
            "title": "Apple Q4 Earnings Beat",
            "source": "finnhub_news",
            "published_at": "2025-09-26",
            "url": "https://.../news",
        }
      }
    """
    out = []
    seen = set()
    for it in items:
        headline = _clean_text(it.get("headline", ""))
        summary  = _clean_text(it.get("summary", ""))  # may be empty on some items
        url      = it.get("url") or ""
        src      = _clean_text(it.get("source", "news"))
        unix_dt  = it.get("datetime") or 0

        # Deduplicate by headline+url
        key = (headline.lower(), url)
        if key in seen or not headline:
            continue
        seen.add(key)

        # Build short text (headline + 1–2 sentences)
        if summary:
            # keep it concise (~300 chars)
            base = f"{headline}. {summary}"
            text = base[:300] + ("…" if len(base) > 300 else "")
        else:
            text = headline

        # Convert publish time to date string (UTC)
        if unix_dt:
            dt = datetime.fromtimestamp(unix_dt, tz=timezone.utc)
            published = dt.strftime("%Y-%m-%d")
        else:
            published = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        out.append({
            "text": text,
            "meta": {
                "ticker": ticker.upper(),
                "title": headline[:120],
                "source": "finnhub_news",
                "published_at": published,
                "url": url,
            }
        })
    return out

def fetch_and_normalize_news_for_watchlist(tickers: list, days: int = 7) -> List[Dict[str, Any]]:
    """
    Convenience: fetch + normalize for multiple tickers.
    Returns a flat list of normalized records (each has text + meta).
    """
    all_records = []
    for t in tickers:
        try:
            raw = fetch_company_news(t, days=days)
            norm = normalize_news_items(t, raw)
            all_records.extend(norm)
        except Exception as e:
            # fail soft per ticker; continue others
            all_records.append({
                "text": f"[FETCH_ERROR] Could not fetch news for {t}: {e}",
                "meta": {"ticker": t.upper(), "source": "finnhub_news_error"}
            })
    return all_records
