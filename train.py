# train.py — uses Yahoo Finance (yfinance) for 5-minute candles
import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

SYMBOL = "AAPL"
FEATS = ["ret1","ema5","ema20","rsi14","range","vol5"]

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df):
    df = df.copy()
    df["ret1"]  = np.log(df["close"]).diff(1)
    df["ema5"]  = ema(df["close"], 5)
    df["ema20"] = ema(df["close"], 20)
    df["rsi14"] = rsi(df["close"], 14)
    df["range"] = df["high"] - df["low"]
    df["vol5"]  = df["ret1"].rolling(5).std()
    df["y"]     = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)

# ------------- Download AAPL 5m candles (~30 days) -------------
# Yahoo supports intraday 5m for recent period; timezone is local; we’ll normalize.
hist = yf.download(
    tickers=SYMBOL,
    period="30d",       # last ~30 days
    interval="5m",      # 5-minute bars
    auto_adjust=False,
    progress=False
)

if hist.empty:
    raise SystemExit("Yahoo returned no data. Try again or reduce period to '14d'.")

# Normalize columns
hist = hist.rename(columns={
    "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
})
# Index is DatetimeIndex; store as UTC if tz-aware
if hist.index.tz is not None:
    hist.index = hist.index.tz_convert("UTC")
hist = hist.reset_index().rename(columns={"Datetime":"ts"})

# Build features & train/test split
df_feat = make_features(hist)
X = df_feat[FEATS].values
y = df_feat["y"].values
split = int(len(df_feat)*0.7)
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

clf = LogisticRegression(max_iter=200)
clf.fit(Xtr_s, ytr)
acc = accuracy_score(yte, clf.predict(Xte_s))
print(f"Holdout accuracy: {acc:.3f} on {len(df_feat)} samples (5m)")

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
pd.Series(FEATS).to_json("models/feature_list.json", orient="values")
print("Saved: models/model.pkl, models/scaler.pkl, models/feature_list.json")
