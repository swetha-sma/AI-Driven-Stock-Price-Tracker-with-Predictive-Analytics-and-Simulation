# app.py — PricePal: live quotes (Finnhub) + simple 5m hint (Yahoo) + paper trades + RAG Q&A
import os, time, re
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
import finnhub
import pandas as pd
import numpy as np
import joblib
import json

# Model utils
from model_util import load_or_train_model, predict_next_prob_up
from model_util import load_candles

def predict_next_price_point(ticker: str, bundle) -> tuple[float|None, float|None, dict]:
    """
    Returns (predicted_price, p_up, meta dict). If not enough data, returns (None, None, {...}).
    Method: expected log-return = (2*p_up - 1) * mean(|ret1| of recent window),
            next_price = last * exp(expected_log_return).
    """
    try:
        interval = bundle.meta.get("interval_used","5m")
        period   = {"5m":"5d","15m":"30d","1d":"1y"}.get(interval,"5d")

        df = load_candles(ticker, period=period, interval=interval)
        if df.empty or len(df) < 30:
            return None, None, {"err":"few_candles"}

        # probability from your classifier
        p_up, ctx = predict_next_prob_up(ticker, bundle)

        # recent volatility proxy from log returns
        df = df.sort_values("ts").copy()
        df["ret1"] = np.log(df["close"]).diff(1)
        recent = df["ret1"].dropna().tail(60)
        if recent.empty:
            return None, p_up, {"err":"no_recent_returns"}
        mean_abs = float(recent.abs().mean())

        last = float(df["close"].iloc[-1])
        exp_logret = (2.0 * float(p_up) - 1.0) * mean_abs
        pred_price = last * np.exp(exp_logret)

        meta = {
            "interval": interval,
            "period": period,
            "mean_abs_ret": mean_abs,
            "last_price": last
        }
        return pred_price, p_up, meta
    except Exception as e:
        return None, None, {"err": str(e)}


# RAG imports
import chromadb
from chromadb.utils import embedding_functions
from rag_build import build_index  # <- dynamic per-ticker rebuild

# --------------- Setup ---------------
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

st.set_page_config(page_title="PricePal — Stocks Chat & Tools", layout="wide")
st.title("PricePal — Stocks Chat & Tools")

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
    typed = [t.strip().upper().lstrip("$") for t in re.split(r"[,\s]+", q) if t.strip()]
    if typed and all(1 <= len(t) <= 5 and t.isalnum() for t in typed):
        # Looks like pure tickers already
        return [{"symbol": t, "desc": "", "exchange": ""} for t in typed]
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
    st.warning(f"RAG store not available yet ({e}). Build it via the sidebar below or run rag_build.py.")

def resolve_ticker_for_question(q: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9]+", q.lower())
    name_map = {
        "apple":"AAPL","aapl":"AAPL",
        "tesla":"TSLA","tsla":"TSLA",
        "microsoft":"MSFT","msft":"MSFT",
        "google":"GOOGL","alphabet":"GOOGL","googl":"GOOGL",
        "amazon":"AMZN","amzn":"AMZN",
        "nvidia":"NVDA","nvda":"NVDA","nvdia":"NVDA","nvida":"NVDA","nvidea":"NVDA",
        "meta":"META","facebook":"META",
        "netflix":"NFLX","adobe":"ADBE","intel":"INTC"
    }
    for w in words:
        if w in name_map:
            return name_map[w]

    det = detect_tickers(q)
    if det:
        return det[0]

    try:
        opts = search_symbols(q)
        if opts:
            return opts[0]["symbol"]
    except Exception:
        pass
    return None


def detect_tickers(text: str):
    raw = re.findall(r"\$?[A-Z]{2,5}", text.upper())  # min len 2
    blacklist = {"LATEST","WHAT","NEWS","ABOUT","STOCK","COMPANY","PRICE","PRICES","EARNINGS"}
    return list({r.lstrip("$") for r in raw if r not in blacklist})

def retrieve_chunks(question: str, k: int = 5, filter_by_tickers=True):
    """Query Chroma with optional ticker filter; return list of dicts: {text, meta, score}."""
    if _docs is None:
        return []
    where = None
    if filter_by_tickers:
        tickers = detect_tickers(question)
        if tickers:
            # Use $in to match any of the detected tickers
            where = {"ticker": {"$in": tickers}}
    res = _docs.query(query_texts=[question], n_results=k, where=where)
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

    st.subheader("Rebuild RAG (per ticker)")
    _rebuild_ticker = st.text_input("Ticker for RAG index (e.g., AAPL)", value="AAPL")
    _days = st.slider("Days of news to ingest", 3, 21, 7, 1)
    if st.button("Rebuild RAG Index (this ticker only)"):
        if _rebuild_ticker.strip():
            with st.spinner(f"Rebuilding index for {_rebuild_ticker.upper()} …"):
                summary = build_index(watchlist=[_rebuild_ticker.upper()], days=_days)
            st.success(f"Indexed {summary['docs_indexed']} records for {', '.join(summary['tickers'])}")
        else:
            st.warning("Enter a ticker first.")




# -------- Multi-ticker price table --------
st.subheader("Check stock prices - Multiple Tickers")
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

# -------- Predict next move + paper trade --------
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

        st.divider()

# ---- Local LLM via Ollama (Phi-3 Mini) ----
try:
    import ollama
except Exception as e:
    ollama = None
    st.warning(f"Ollama client not available: {e}. Start Ollama or run `pip install ollama`.")


def local_llm_answer(prompt: str, model: str = "phi3:mini") -> str:
    """
    Sends a prompt to a local Ollama model and returns the response text.
    Make sure Ollama is running (http://localhost:11434).
    """
    if ollama is None:
        return "(Local model unavailable. Is Ollama running? Try `ollama run phi3:mini ...` in a terminal.)"

    try:
        stream = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful stock assistant. Answer concisely and use plain English."},
            {"role": "user", "content": prompt}
        ])
        return stream["message"]["content"].strip()
    except Exception as e:
        return f"(Local model error: {e})"


# ======================= Chatbot (multi-turn) =======================
# ======================= Chatbot (multi-turn) =======================
st.divider()
st.subheader("Chatbot — Stocks & Companies (RAG)")

# Keep multi-turn chat history
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask about any stock/company (e.g., 'What is Tesla?' or 'Predict NVDA next move')")

if not user_msg:
    # nothing to do on this rerun, keep rest of the page rendering
    pass
else:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Decide the main ticker for this turn
    ticker = resolve_ticker_for_question(user_msg) or st.session_state.last_ticker
    st.session_state.last_ticker = ticker

    # Mode: predict vs info-only
    q_lower = user_msg.lower()
    predict_keywords = ("predict", "next move", "forecast", "estimate", "target", "probability")
    is_predict = any(k in q_lower for k in predict_keywords)

    with st.spinner("Thinking…"):
        # 1) Retrieve RAG context (auto-ingest if empty and we have a ticker)
        hits = retrieve_chunks(user_msg, k=5, filter_by_tickers=True)
        if not hits and ticker:
            try:
                build_index(watchlist=[ticker], days=7)
                hits = retrieve_chunks(user_msg, k=5, filter_by_tickers=True)
            except Exception as e:
                st.warning(f"Could not ingest news for {ticker}: {e}")

        context = "\n\n".join(ch["text"] for ch in hits[:6]) if hits else "(no context)"

        # 2) Live price (Finnhub)
        last_px = chg = chg_pct = None
        try:
            if ticker:
                qnow, _ = get_quote(ticker)
                last_px = qnow.get("c")
                chg, chg_pct = qnow.get("d"), qnow.get("dp")
        except Exception:
            pass

        # 3) If predict mode, compute model probability + short-term estimate
        p_up = pred_px = None
        interval_used = None
        if is_predict and ticker:
            try:
                bundle = load_or_train_model(ticker)
                if bundle:
                    interval_used = bundle.meta.get("interval_used", "5m")
                    pred_px, p_up, _meta = predict_next_price_point(ticker, bundle)
            except Exception:
                pass

        # 4) Ask local LLM for very short bullets (overview only)
        prompt = (
            "You are a concise stock analysis assistant.\n"
            "ANSWER RULES (follow exactly):\n"
            "- NO paragraphs.\n"
            "- Use short bullet points only.\n"
            "- Max 2 bullets for OVERVIEW.\n"
            "- If context is weak: say 'Context limited.'\n\n"
            "FORMAT (follow exactly):\n"
            "OVERVIEW:\n"
            "- bullet\n"
            "- bullet\n\n"
            f"USER QUESTION:\n{user_msg}\n\n"
            f"CONTEXT:\n{context}"
        )
        llm_sections = local_llm_answer(prompt)

    # ---------- Build short reply ----------
    def _extract_bullets(llm_text: str):
        items = []
        for line in (llm_text or "").splitlines():
            s = line.strip()
            if s.startswith(("-", "•")):
                items.append(s.lstrip("-• ").strip())
        return items

    overview = _extract_bullets(llm_sections)[:2] or ["Context limited."]

    # Clean price line
    if last_px is not None:
        price_line = f"**Current Price:** ${float(last_px):,.2f}"
        if chg is not None and chg_pct is not None:
            price_line += f" ({float(chg):+,.2f}, {float(chg_pct):+,.2f}%)"
    else:
        price_line = "**Current Price:** Not available"

    with st.chat_message("assistant"):
        # Always show overview bullets
        st.markdown("**Company Overview (RAG):**")
        st.markdown("\n".join(f"- {o}" for o in overview))

        # Always show current price
        st.markdown(price_line)

        # Only if predict mode: show probability + estimate
        if is_predict:
            c1, c2 = st.columns(2)
            if p_up is not None:
                direction = "UP" if p_up >= 0.5 else "DOWN"
                c1.metric("Next Move Probability", f"{int(round(p_up*100))}%", direction)
            else:
                c1.metric("Next Move Probability", "—")
            if pred_px is not None:
                c2.metric(f"Short-Term Estimate (next {interval_used or '15m'})", f"${float(pred_px):,.2f}")
            else:
                c2.metric("Short-Term Estimate", "—")

        # Always show retrieved chunks
        if hits:
            links = []
            for h in hits:
                meta = h.get("meta") or {}
                title = meta.get("title"); url = meta.get("url")
                if title and title not in links:
                    links.append(f"[{title}]({url})" if url else title)
            if links:
                st.markdown("**Sources:** " + " • ".join(links))

            with st.expander("Show retrieved chunks"):
                for i, h in enumerate(hits, 1):
                    meta = h.get("meta") or {}
                    st.markdown(f"**{i}. {meta.get('title','(untitled)')}** — ticker: {meta.get('ticker','GEN')}")
                    st.write(h["text"])

    # Push a compact, single-line summary into history (for the scroll)
    compact = f"Overview: {', '.join(overview)}"
    if last_px is not None:
        compact += f" | Price: ${float(last_px):,.2f}"
    if is_predict and (p_up is not None):
        compact += f" | P(up): {int(round(p_up*100))}%"
    if is_predict and (pred_px is not None):
        compact += f" | Est: ${float(pred_px):,.2f}"
    st.session_state.chat.append({"role": "assistant", "content": compact})
