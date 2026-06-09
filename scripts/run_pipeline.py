"""One-shot pipeline runner: collect -> score -> features -> train -> predict.

This is what the scheduler (GitHub Actions) and n8n call. It writes two artifacts
the dashboard reads:
  data/processed/dataset.parquet      (features + targets, full history)
  data/processed/predictions.parquet  (latest prediction per ticker)
  reports/metrics.json                (headline metrics for the dashboard)

Run:  python -m scripts.run_pipeline
Env:  USE_SYNTHETIC=planted|random  -> skip live collection, use generated data
                                        (handy for CI smoke tests with no API keys)
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RAW, PROCESSED, ROOT
from src.features.build import build

REPORTS = ROOT / "reports"


def collect():
    mode = os.getenv("USE_SYNTHETIC", "").strip()
    if mode in ("planted", "random"):
        print(f"[collect] synthetic mode: {mode}")
        subprocess.run([sys.executable, "make_data.py", mode], check=True,
                       env={**os.environ, "PYTHONPATH": "."})
        return
    print("[collect] live mode")
    subprocess.run([sys.executable, "-m", "src.data.collect_prices"], check=True)
    subprocess.run([sys.executable, "-m", "src.data.collect_stocktwits"], check=True)


def score():
    subprocess.run([sys.executable, "-m", "src.sentiment.score"], check=True)


def train_and_predict():
    df, feature_cols = build()
    df = df.sort_values("date")
    X, y = df[feature_cols], df["target_up"].astype(int)

    # Honest CV metrics for the dashboard
    tscv = TimeSeriesSplit(n_splits=5)
    accs, aucs, bases = [], [], []
    for tr, te in tscv.split(X):
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        est.fit(X.iloc[tr], y.iloc[tr])
        proba = est.predict_proba(X.iloc[te])[:, 1]
        accs.append(accuracy_score(y.iloc[te], (proba > 0.5).astype(int)))
        aucs.append(roc_auc_score(y.iloc[te], proba) if y.iloc[te].nunique() > 1 else np.nan)
        bases.append(max(y.iloc[tr].mean(), 1 - y.iloc[tr].mean()))

    # Final model on all data -> predict latest row per ticker
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    final.fit(X, y)
    latest = df.sort_values("date").groupby("ticker").tail(1).copy()
    latest["pred_up_proba"] = final.predict_proba(latest[feature_cols])[:, 1]
    preds = latest[["date", "ticker", "close", "sent_mean", "msg_volume",
                    "pred_up_proba"]].sort_values("pred_up_proba", ascending=False)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "dataset.parquet", index=False)
    preds.to_parquet(PROCESSED / "predictions.parquet", index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "cv_accuracy": round(float(np.mean(accs)), 4),
        "cv_baseline": round(float(np.mean(bases)), 4),
        "cv_auc": round(float(np.nanmean(aucs)), 4),
        "edge": round(float(np.mean(accs) - np.mean(bases)), 4),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[metrics]", json.dumps(metrics, indent=2))
    return metrics


def main():
    collect()
    score()
    metrics = train_and_predict()
    # Regenerate the static dashboard
    subprocess.run([sys.executable, "-m", "scripts.build_report"], check=True)
    print("[done] pipeline complete")
    return metrics


if __name__ == "__main__":
    main()
