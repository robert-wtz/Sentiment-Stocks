"""One-shot pipeline runner: collect -> score -> features -> train -> predict.

Writes:
  data/processed/dataset.parquet      full feature matrix with multi-horizon targets
  data/processed/predictions.parquet  latest signal per ticker per horizon
  reports/metrics.json                headline metrics for the dashboard

Run:  python -m scripts.run_pipeline
Env:  USE_SYNTHETIC=planted|random  -> skip live collection (CI / no API keys)
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

from src.config import RAW, PROCESSED, ROOT, HORIZONS
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


def _cv_metrics(X, y):
    """Walk-forward CV. Returns (mean_accuracy, mean_baseline, mean_auc)."""
    tscv = TimeSeriesSplit(n_splits=5)
    accs, aucs, bases = [], [], []
    for tr, te in tscv.split(X):
        if y.iloc[te].nunique() < 2:
            continue
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        est.fit(X.iloc[tr], y.iloc[tr])
        proba = est.predict_proba(X.iloc[te])[:, 1]
        accs.append(accuracy_score(y.iloc[te], (proba > 0.5).astype(int)))
        aucs.append(roc_auc_score(y.iloc[te], proba))
        bases.append(max(y.iloc[tr].mean(), 1 - y.iloc[tr].mean()))
    if not accs:
        return 0.5, 0.5, 0.5
    return float(np.mean(accs)), float(np.mean(bases)), float(np.nanmean(aucs))


def train_and_predict():
    df, feature_cols, target_cols = build()
    df = df.sort_values("date")

    horizon_metrics = {}
    pred_frames = []

    for target_col, horizon in zip(target_cols, HORIZONS):
        sub = df.dropna(subset=[target_col]).copy()
        X = sub[feature_cols]
        y = sub[target_col].astype(int)

        acc, base, auc = _cv_metrics(X, y)
        horizon_metrics[f"{horizon}d"] = {
            "cv_accuracy": round(acc, 4),
            "cv_baseline": round(base, 4),
            "cv_auc":      round(auc, 4),
            "edge":        round(acc - base, 4),
        }
        print(f"[{horizon}d] acc={acc:.3f} base={base:.3f} edge={acc-base:+.3f} auc={auc:.3f}")

        # Final model on all available data → predict latest row per ticker
        final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        final.fit(X, y)
        latest = sub.sort_values("date").groupby("ticker").tail(1).copy()
        latest["pred_up_proba"] = final.predict_proba(latest[feature_cols])[:, 1]
        latest["horizon"] = f"{horizon}d"
        pred_frames.append(
            latest[["date", "ticker", "horizon", "close", "sent_mean",
                     "msg_volume", "pred_up_proba"]]
        )

    preds = (
        pd.concat(pred_frames, ignore_index=True)
        .sort_values(["horizon", "pred_up_proba"], ascending=[True, False])
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "dataset.parquet", index=False)
    preds.to_parquet(PROCESSED / "predictions.parquet", index=False)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "horizons": horizon_metrics,
        # Top-level convenience fields (20d as the primary horizon)
        "cv_accuracy": horizon_metrics[f"{HORIZONS[0]}d"]["cv_accuracy"],
        "cv_baseline": horizon_metrics[f"{HORIZONS[0]}d"]["cv_baseline"],
        "cv_auc":      horizon_metrics[f"{HORIZONS[0]}d"]["cv_auc"],
        "edge":        horizon_metrics[f"{HORIZONS[0]}d"]["edge"],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[metrics]", json.dumps(metrics, indent=2))
    return metrics


def main():
    collect()
    score()
    metrics = train_and_predict()
    subprocess.run([sys.executable, "-m", "scripts.build_report"], check=True)
    print("[done] pipeline complete")
    return metrics


if __name__ == "__main__":
    main()
