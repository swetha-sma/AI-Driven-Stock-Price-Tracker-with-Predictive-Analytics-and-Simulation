# model_util.py — per-ticker trainer/loader for next-bar direction
import os, json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ---------- Features (match your app/train.py style) ----------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

FEATS = ["ret1","ret3","ret6","ema5","ema20","rsi14","range","vol5"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts").copy()
    df["ret1"]  = np.log(df["close"]).diff(1)
    df["ret3"]  = np.log(df["close"]).diff(3)
    df["ret6"]  = np.log(df["close"]).diff(6)
    df["ema5"]  = ema(df["close"], 5)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    df["range"] = df["high"] - df["low"]
    df["vol5"]  = df["ret1"].rolling(5).std()
    df["y"]     = (df["close"].shift(-1) > df["close"]).astype(int)  # next bar up?
    return df.dropna().reset_index(drop=True)

# ---------- Data loading ----------
def load_candles(ticker: str, period="5d", interval="5m") -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    import yfinance as yf

    hist = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if hist is None or len(hist) == 0:
        return pd.DataFrame()

    # If MultiIndex columns (e.g., ('Open','AAPL')), flatten to first level
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] if isinstance(c, tuple) else c for c in hist.columns]

    # Helper to pick the first available column name among variants
    def pick(*names):
        for n in names:
            if n in hist.columns:
                return n
        return None

    col_open   = pick("Open", "open", "OPEN")
    col_high   = pick("High", "high", "HIGH")
    col_low    = pick("Low", "low", "LOW")
    col_close  = pick("Close", "close", "CLOSE", "Adj Close", "AdjClose", "adjclose")
    col_volume = pick("Volume", "volume", "VOLUME")

    # Build a normalized frame; require at least 'close'
    df = pd.DataFrame()
    if col_open is not None:   df["open"]   = hist[col_open]
    if col_high is not None:   df["high"]   = hist[col_high]
    if col_low  is not None:   df["low"]    = hist[col_low]
    if col_close is not None:  df["close"]  = hist[col_close]
    if col_volume is not None: df["volume"] = hist[col_volume]

    # Must have close; drop rows without it
    if "close" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["close"])

    # Move index -> UTC timestamp column "ts"
    df = df.reset_index()
    ts_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
    df = df.rename(columns={ts_col: "ts"})

    # Ensure timezone uniformity (yfinance returns timezone-aware for intraday)
    try:
        if hasattr(df["ts"].dt, "tz") and df["ts"].dt.tz is not None:
            df["ts"] = df["ts"].dt.tz_convert("UTC")
    except Exception:
        # If it's date (no time), leave as-is
        pass

    return df


@dataclass
class ModelBundle:
    clf: LogisticRegression
    scaler: StandardScaler
    feats: list
    meta: dict

MODELS_DIR = "models"

def _ticker_dir(ticker: str) -> str:
    return os.path.join(MODELS_DIR, ticker.upper())

def _paths(ticker: str):
    d = _ticker_dir(ticker)
    mpath = os.path.join(d, "model.pkl")
    spath = os.path.join(d, "scaler.pkl")
    jpath = os.path.join(d, "meta.json")
    return mpath, spath, jpath


from datetime import datetime, timezone, timedelta

def _file_age_hours(path: str) -> float:
    try:
        mtime = os.path.getmtime(path)
        return (datetime.now(timezone.utc) - datetime.fromtimestamp(mtime, tz=timezone.utc)).total_seconds() / 3600.0
    except Exception:
        return 1e9  # very old if missing

def _is_stale(meta_path: str, max_age_hours: int = 24) -> bool:
    # Prefer meta.json timestamp; fallback to file mtime
    try:
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            ts = meta.get("trained_at")
            if ts:
                trained_at = datetime.fromisoformat(ts.replace("Z","+00:00"))
                age = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600.0
                return age > max_age_hours
    except Exception:
        pass
    # If meta missing or unreadable, check file mtimes via _file_age_hours
    return _file_age_hours(meta_path) > max_age_hours


# ---------- Train (with simple fallbacks) ----------
def train_for_ticker(ticker: str) -> Optional[ModelBundle]:
    """
    Tries 5m/5d; if not enough rows, tries 15m/30d; then daily/1y.
    Returns ModelBundle or None if insufficient data.
    """
    attempts = [
        ("5m","5d",   500),
        ("15m","30d", 500),
        ("1d","1y",   150),
    ]
    df = None; used = None
    for interval, period, min_rows in attempts:
        tmp = load_candles(ticker, period=period, interval=interval)
        if len(tmp) >= min_rows:
            df = tmp; used = (interval, period)
            break
    if df is None:
        return None

    feats = make_features(df)
    if len(feats) < 200:  # still too small to be meaningful
        return None

    X = feats[FEATS].values
    y = feats["y"].values
    split = int(len(feats)*0.7)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr_s, ytr)
    acc = float(accuracy_score(yte, clf.predict(Xte_s)))
    mpath, spath, jpath = _paths(ticker)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)

    joblib.dump(clf, mpath)
    joblib.dump(scaler, spath)
    meta = {
        "ticker": ticker.upper(),
        "features": FEATS,
        "interval_used": used[0],
        "period_used": used[1],
        "samples_total": int(len(feats)),
        "holdout_accuracy": acc,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with open(jpath, "w") as f:
        json.dump(meta, f)

    return ModelBundle(clf=clf, scaler=scaler, feats=FEATS, meta=meta)

# ---------- Load (with auto-train) ----------
def load_or_train_model(ticker: str, max_age_hours: int = 24) -> Optional[ModelBundle]:
    mpath, spath, jpath = _paths(ticker)

    # If all files exist and model is fresh, load it
    if os.path.exists(mpath) and os.path.exists(spath) and os.path.exists(jpath) and not _is_stale(jpath, max_age_hours):
        try:
            clf = joblib.load(mpath)
            scaler = joblib.load(spath)
            meta = json.load(open(jpath))
            return ModelBundle(clf=clf, scaler=scaler, feats=meta.get("features", FEATS), meta=meta)
        except Exception:
            pass  # fall through to retrain

    # Otherwise: (missing or stale) → train fresh
    return train_for_ticker(ticker)


# ---------- Predict last bar for a ticker ----------
def predict_next_prob_up(ticker: str, bundle: ModelBundle) -> Tuple[float, dict]:
    # Use the same interval/period used during training for consistency
    interval = bundle.meta.get("interval_used", "5m")
    period   = {"5m":"5d","15m":"30d","1d":"1y"}.get(interval, "5d")
    df = load_candles(ticker, period=period, interval=interval)
    feats = make_features(df)
    if feats.empty:
        return 0.5, {"err": "no_features"}
    X_last = feats[bundle.feats].iloc[[-1]].values
    X_last_s = bundle.scaler.transform(X_last)
    p_up = float(bundle.clf.predict_proba(X_last_s)[0,1])
    return p_up, {"interval": interval, "period": period, "rows": int(len(feats))}
