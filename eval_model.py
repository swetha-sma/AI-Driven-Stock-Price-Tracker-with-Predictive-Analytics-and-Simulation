import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

from model_util import load_or_train_model, load_candles, make_features

def ensure_reports():
    os.makedirs("reports", exist_ok=True)
    return "reports"

def plot_roc(y, y_prob, path):
    fpr, tpr, _ = roc_curve(y, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0,1],[0,1],"--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_calibration(y, y_prob, path):
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Predicted P(up)")
    plt.ylabel("Observed P(up)")
    plt.title("Calibration Curve")
    plt.savefig(path)
    plt.close()

def evaluate(ticker="TSLA"):
    bundle = load_or_train_model(ticker)
    if bundle is None:
        print("Could not train or load a model for this ticker.")
        return

    interval = bundle.meta["interval_used"]
    period = bundle.meta["period_used"]
    print(f"Evaluating {ticker} on its training window: {interval} / {period}")

    df = load_candles(ticker, period=period, interval=interval)
    feats = make_features(df)

    X = feats[bundle.feats].values
    y = feats["y"].values

    X_scaled = bundle.scaler.transform(X)
    y_prob = bundle.clf.predict_proba(X_scaled)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    brier = brier_score_loss(y, y_prob)

    print("\nModel Performance (Same Window as Training):")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision (UP) : {prec:.4f}")
    print(f"Recall (UP)    : {rec:.4f}")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"Brier Score    : {brier:.4f}")
    print("Confusion Matrix:\n", cm)

    coefs = pd.DataFrame({
        "feature": bundle.feats,
        "coef": bundle.clf.coef_.ravel()
    }).sort_values("coef", key=abs, ascending=False)
    print("\nTop Feature Influences:")
    print(coefs.to_string(index=False))

    reports = ensure_reports()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    plot_roc(y, y_prob, f"{reports}/{ticker}_roc_{ts}.png")
    plot_calibration(y, y_prob, f"{reports}/{ticker}_cal_{ts}.png")
    coefs.to_csv(f"{reports}/{ticker}_coeffs_{ts}.csv", index=False)

    print("\nSaved reports to /reports folder.")

if __name__ == "__main__":
    evaluate("TSLA")
