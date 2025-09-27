# app.py — PricePal: live quotes (Finnhub) + simple 5m hint (Yahoo) + paper trades + RAG Q&A
import os, time, re, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo
from model_util import load_or_train_model, predict_next_prob_up

import streamlit as st
from dotenv import load_dotenv
import finnhub
import pandas as pd
import numpy as np
import joblib
import json

# RAG imports
import chromadb
from chromadb.utils import embedding_functions
from textwrap import shorten

# --------------- Setup ---------------
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

st.set_page_config(page_title="PricePal — Prices, 5m Hint & RAG", layout="wide")
st.title("PricePal — Live Prices, Simple 5-Minute Hint, and RAG Q&A")

if not API_KEY:
    st.error("FINNHUB_API_KEY not found in .env. Add `FINNHUB_API_KEY=...` and restart.")
    st.stop()

client = finnhub.Client(api_key=API_KEY)

# small in-memory cache for quotes (avoid rate limits)
CACHE = {}   # {sym: (ts, data)}
TTL = 45     # seconds

def fmt_time_unix_to_et(unix_ts: int) -> str:
    if not unix_ts:
        return "—"
    return datetime.fromtimestamp(unix_ts, tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")

def get_quote(symbol: str):
    """Return (data, from_cache). Finnhub fields: c (price), d (Δ), dp (Δ%), t (unix)."""
    s = symbol.upper().strip().lstrip("$")
    now = time.time()
    if s in CACHE and (now - CACHE[s][0]) < TTL:
        return CACHE[s][1], True
    data = client.quote(s)
    CACHE[s] = (now, data)
    return data, False

# --- Symbol search & resolve (Finnhub) ---
from functools import lru_cache

@lru_cache(maxsize=256)
def search_symbols(query: str):
    """Search by company name or partial ticker, return top matches."""
    try:
        # Finnhub recently named this endpoint /search.
        # Some SDK versions expose client.symbol_lookup, others symbol_search.
        try:
            res = client.symbol_lookup(query)      # try 1
        except AttributeError:
            res = client.symbol_search(query)      # fallback
        items = res.get("result", [])
        clean = []
        for it in items:
            sym = it.get("symbol", "")
            desc = it.get("description", "")
            exh = it.get("exchange", "")
            typ = it.get("type", "")
            # keep common US equity/ETF types for a clean demo
            if typ in ("Common Stock", "ETF", "ETP", "ADR") and sym and desc:
                clean.append({"symbol": sym, "desc": desc, "exchange": exh})
        return clean[:10]
    except Exception:
        return []

def resolve_to_symbols(user_text: str):
    """If user typed clear tickers (AAPL NVDA), return those; else do name search."""
    q = user_text.strip()
    if not q:
        return []
    import re
    typed = [t.strip().upper().lstrip("$") for t in re.split(r"[,\s]+", q) if t.strip()]
    if typed and all(1 <= len(t) <= 5 and t.isalnum() for t in typed):
        # Looks like pure tickers already
        return [{"symbol": t, "desc": "", "exchange": ""} for t in typed]
    # Otherwise, search by name
    return search_symbols(q)


# --------------- Feature functions (must match train.py) ---------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df):
    df = df.sort_values("ts").copy()
    df["ret1"]  = np.log(df["close"]).diff(1)
    df["ema5"]  = ema(df["close"], 5)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    df["range"] = df["high"] - df["low"]
    df["vol5"]  = df["ret1"].rolling(5).std()
    return df

# --------------- Load model (trained by train.py) ---------------
try:
    clf = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    FEATS = json.load(open("models/feature_list.json"))
    MODEL_READY = True
except Exception:
    MODEL_READY = False

# --------------- RAG: client + retrieval helpers ---------------
# Persistent Chroma client pointing to rag_store you built
try:
    _chroma_client = chromadb.PersistentClient(path="rag_store")
    _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    _docs = _chroma_client.get_or_create_collection(
        name="pricepal_docs",
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedder,
    )
except Exception as e:
    _docs = None
    st.warning(f"RAG store not available yet ({e}). Build it via the sidebar 'Rebuild RAG Index' or run rag_build.py.")

def detect_tickers(text: str):
    """Very simple ticker extractor: 'AAPL', 'MSFT', '$NVDA' -> ['AAPL', 'MSFT', 'NVDA']"""
    raw = re.findall(r"\$?[A-Z]{1,5}", text.upper())
    return list({r.lstrip("$") for r in raw})

def retrieve_chunks(question: str, k: int = 5, filter_by_tickers=True):
    """Query Chroma with optional ticker filter; return list of dicts: {text, meta, score}."""
    if _docs is None:
        return []
    where = {}
    if filter_by_tickers:
        tickers = detect_tickers(question)
        if tickers:
            where = {"$or": [{"ticker": t} for t in tickers]}
    res = _docs.query(query_texts=[question], n_results=k, where=where or None)
    out = []
    for text, meta, score in zip(res.get("documents", [[]])[0],
                                 res.get("metadatas", [[]])[0],
                                 res.get("distances", [[]])[0]):
        out.append({"text": text, "meta": meta, "score": float(score)})
    return out

def compose_answer_from_chunks(question: str, chunks: list, max_points: int = 5, max_words: int = 120):
    import re
    q_words = [w for w in re.findall(r"[A-Za-z]+", question.lower()) if len(w) > 2]

    def best_sentences(text, limit=1):
        sents = re.split(r"(?<=[.!?])\s+", text.strip().replace("\n", " "))
        scored = []
        for s in sents:
            w = re.findall(r"[A-Za-z]+", s.lower())
            scored.append((len(set(q_words) & set(w)), len(s), s))
        scored.sort(key=lambda x: (-x[0], x[1]))  # more overlap, then shorter
        return [t[2] for t in scored[:limit] if t[2]]

    bullets = []
    for ch in chunks:
        picks = best_sentences(ch["text"], limit=1)
        if picks:
            bullets.append("• " + picks[0].strip())
        if len(bullets) >= max_points:
            break

    if not bullets:
        return "I don’t have enough information in my notes to answer that confidently.", []

    # enforce ~120-word cap
    words = " ".join(bullets).split()
    if len(words) > max_words:
        kept, count = [], 0
        for b in bullets:
            w = b.split()
            if count + len(w) <= max_words:
                kept.append(b)
                count += len(w)
            else:
                break
        bullets = kept

    answer = "\n".join(bullets)
    titles = list({(ch["meta"] or {}).get("title") for ch in chunks if (ch["meta"] or {}).get("title")})
    return answer, titles


# ========== SIDEBAR: Utilities ==========
with st.sidebar:
    st.header("Utilities")
    # Reset paper ledger
    if st.button("Reset paper ledger"):
        try:
            if os.path.exists("ledger.csv"):
                os.remove("ledger.csv")
                st.success("Ledger reset.")
            else:
                st.info("No ledger found to reset.")
        except Exception as e:
            st.error(f"Could not reset ledger: {e}")

    # Rebuild RAG index (runs rag_build.py)
    if st.button("Rebuild RAG Index"):
        try:
            # Use the same interpreter/venv to run the script
            cmd = [sys.executable, "rag_build.py"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                st.success("RAG index rebuilt successfully.")
                st.caption(proc.stdout or "Done.")
            else:
                st.error("Failed to rebuild RAG index.")
                st.code(proc.stderr or proc.stdout)
        except Exception as e:
            st.error(f"Error while rebuilding index: {e}")

# ========== Top: highlight card (AAPL live) ==========
try:
    aapl, _ = get_quote("AAPL")
    price = aapl.get("c"); change = aapl.get("d"); pct = aapl.get("dp"); ts = aapl.get("t")
except Exception as e:
    st.error(f"Could not fetch AAPL price: {e}")
    st.stop()

c1, c2 = st.columns([1,1])
with c1:
    st.subheader("AAPL Price (live)")
    st.metric("AAPL", f"${price:,.2f}" if price else "—",
              f"{change:+.2f} ({pct:+.2f}%)" if change is not None and pct is not None else None)
with c2:
    st.write(f"**Last update:** {fmt_time_unix_to_et(ts)}")

st.divider()


st.subheader("Search any stock price")

qcol1, qcol2 = st.columns([3,1])
qtext = qcol1.text_input("Type a company name or ticker (e.g., Apple, NVDA, MSFT)", value="Apple")
limit = qcol2.slider("Max results", 1, 10, 5)

if st.button("Search"):
    candidates = resolve_to_symbols(qtext)[:limit]
    if not candidates:
        st.info("No matches found. Try a different name or ticker.")
    else:
        labels = [f"{c['symbol']} — {c['desc'] or 'No description'} ({c['exchange']})" for c in candidates]
        sel = st.multiselect("Select symbols to show:", labels, default=labels[:min(3, len(labels))])
        chosen = [lbl.split(" — ")[0].strip() for lbl in sel]

        rows, errors = [], []
        for s in chosen:
            try:
                q, cached = get_quote(s)
                price = q.get("c")
                if not price:
                    errors.append(s); continue
                rows.append({
                    "Symbol": s,
                    "Price": f"${price:,.2f}",
                    "Δ": f"{q.get('d', 0.0):+.2f}",
                    "Δ%": f"{q.get('dp', 0.0):+.2f}%",
                    "Last update (ET)": fmt_time_unix_to_et(q.get("t", 0)),
                    "Source": "cache" if cached else "live"
                })
            except Exception as e:
                errors.append(f"{s} ({e})")

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        if errors:
            st.warning("Skipped: " + ", ".join(errors))
            
# ========== Middle: multi-ticker price table ==========
st.subheader("Check multiple tickers at once")

left, right = st.columns([2,1])
with left:
    user_text = st.text_input("Type tickers (e.g., AAPL MSFT NVDA):", value="AAPL MSFT NVDA")
with right:
    quick = st.multiselect("Or pick:", ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","SPY"],
                           default=["AAPL","MSFT","NVDA"])

typed = [t.strip().upper().lstrip("$") for t in re.split(r"[,\s]+", user_text) if t.strip()]
symbols = list(dict.fromkeys(typed + quick))[:25]

if st.button("Get Prices", type="primary"):
    rows, errors = [], []
    for s in symbols or ["AAPL"]:
        try:
            q, cached = get_quote(s)
            price = q.get("c")
            if not price:
                errors.append(s); continue
            rows.append({
                "Symbol": s,
                "Price": f"${price:,.2f}",
                "Δ": f"{q.get('d', 0.0):+.2f}",
                "Δ%": f"{q.get('dp', 0.0):+.2f}%",
                "Last update (ET)": fmt_time_unix_to_et(q.get("t", 0)),
                "Source": "cache" if cached else "live"
            })
        except Exception as e:
            errors.append(f"{s} ({e})")

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No valid quotes returned.")

    if errors:
        st.warning("Skipped/invalid: " + ", ".join(errors))

st.caption("Source: Finnhub (free tier). Uses a small cache (~45s) to respect rate limits.")

st.divider()

# ========== Bottom: Simple 5-minute hint (AAPL, via Yahoo) + paper trade ==========
st.divider()
st.subheader("Predict next move (any ticker)")

pc1, pc2, pc3 = st.columns([1.2, 1, 1])
ticker_in = pc1.text_input("Ticker", value="AAPL").upper().strip()
threshold = pc2.slider("Confidence needed for LONG", 0.50, 0.65, 0.55, 0.01)
go = pc3.button("Predict")

if go:
    with st.spinner(f"Loading/training model for {ticker_in}…"):
        bundle = load_or_train_model(ticker_in)

    if bundle is None:
        st.warning(f"Not enough data to train a light model for {ticker_in}. Try another ticker or daily interval later.")
    else:
        p_up, ctx = predict_next_prob_up(ticker_in, bundle)
        stance = "LONG" if p_up >= threshold else "FLAT"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", ticker_in)
        c2.metric("P(up)", f"{p_up:.2f}")
        c3.metric("Stance", stance)
        c4.metric("Holdout acc.", f"{bundle.meta.get('holdout_accuracy', 0):.2f}")
        st.caption(f"Interval used: {bundle.meta['interval_used']} | Period: {bundle.meta['period_used']} | Samples: {bundle.meta['samples_total']}")

    # ---- Paper trade (pretend) ----
    st.markdown("### Paper trade (pretend)")
    if st.button("Simulate next trade (AAPL, 1 share)"):
        q_now, _ = get_quote("AAPL")
        fill_price = float(q_now.get("c", 0.0)) or 0.0
        side = "BUY" if action == "LONG" else "FLAT"
        fee = 0.01  # simple fixed fee
        ts_now = fmt_time_unix_to_et(q_now.get("t", 0))

        row = {
            "ts": ts_now, "symbol": "AAPL", "side": side, "qty": 1,
            "fill_price": fill_price, "fee": fee,
            "note": f"threshold={threshold:.2f}, p_up={p_up:.2f}" if 'p_up' in locals() else "no_prob"
        }
        if os.path.exists("ledger.csv"):
            old = pd.read_csv("ledger.csv")
            new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
            new.to_csv("ledger.csv", index=False)
        else:
            pd.DataFrame([row]).to_csv("ledger.csv", index=False)
        st.success(f"Logged paper trade: {side} 1 AAPL @ ${fill_price:.2f}")

    st.markdown("#### Recent paper trades")
    if os.path.exists("ledger.csv"):
        led = pd.read_csv("ledger.csv")
        st.dataframe(led.tail(12), use_container_width=True, hide_index=True)
    else:
        st.info("No trades yet. Click the button above to log your first pretend trade.")

st.caption("Built for learning: live quotes via Finnhub; 5-minute candles via Yahoo; paper trades only (no real money).")

st.divider()

# ========== RAG Q&A ==========
st.subheader("Ask a company question (RAG from your docs)")

colq1, colq2 = st.columns([3,1])
question = colq1.text_input(
    "Ask in plain English (e.g., 'What does NVDA do?' or 'AAPL overview'):",
    value="What does AAPL do?"
)
filter_by_tickers = colq2.toggle(
    "Filter by detected tickers",
    value=True,
    help="Limits retrieval to docs tagged with tickers found in your question."
)

if st.button("Ask (RAG)"):
    with st.spinner("Searching your notes…"):
        hits = retrieve_chunks(question, k=5, filter_by_tickers=filter_by_tickers)
        if not hits:
            st.info("No relevant notes found. Add docs to ./docs and click 'Rebuild RAG Index' in the sidebar.")
        else:
            answer, sources = compose_answer_from_chunks(question, hits, max_points=5)
            st.write(answer)
            links = []
            for h in hits:
                m = h.get("meta") or {}
                title = m.get("title")
                url   = m.get("url")
                if title and title not in links:
                    if url:
                        links.append(f"[{title}]({url})")
                    else:
                        links.append(title)
            if links:
                st.markdown("**Sources:** " + " • ".join(links))
            with st.expander("Show retrieved chunks"):
                for i, h in enumerate(hits, 1):
                    meta = h.get("meta") or {}
                    st.markdown(f"**{i}. {meta.get('title','(untitled)')}** — ticker: {meta.get('ticker','GEN')}")
                    st.write(h["text"])

st.caption("RAG answers come from your local notes (./docs → rag_build.py → ./rag_store). Education only — not financial advice.")
