"""Phase 4 — train & evaluate. The rigor that makes this a real ML project.

Non-negotiables enforced here:
  - TimeSeriesSplit, NEVER random K-fold (random splits leak the future).
  - Compare against DUMB baselines (majority class). If you can't beat them, the
    signal isn't there — and reporting that honestly is a strength.
  - Logistic regression first (interpretable), then LightGBM.

Run:  python -m src.models.train
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED
from src.features.build import build


def majority_baseline(y_train, y_test):
    pred = np.full(len(y_test), int(round(y_train.mean())))
    return accuracy_score(y_test, pred)


def evaluate(model_name, model, X, y, needs_scaling=False):
    """Walk-forward CV. Each split trains on the past, tests on the future."""
    tscv = TimeSeriesSplit(n_splits=5)
    accs, aucs, base_accs = [], [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        est = make_pipeline(StandardScaler(), model) if needs_scaling else model
        est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)

        accs.append(accuracy_score(y_te, pred))
        # AUC undefined if a test fold is single-class; guard it.
        aucs.append(roc_auc_score(y_te, proba) if y_te.nunique() > 1 else np.nan)
        base_accs.append(majority_baseline(y_tr, y_te))

    print(f"\n{model_name}")
    print(f"  accuracy : {np.mean(accs):.3f}  (baseline {np.mean(base_accs):.3f})")
    print(f"  roc auc  : {np.nanmean(aucs):.3f}")
    edge = np.mean(accs) - np.mean(base_accs)
    verdict = "beats baseline" if edge > 0.01 else "NO edge over baseline"
    print(f"  edge     : {edge:+.3f}  -> {verdict}")
    return {"model": model_name, "acc": np.mean(accs),
            "baseline": np.mean(base_accs), "auc": np.nanmean(aucs)}


def main():
    try:
        df = pd.read_parquet(PROCESSED / "dataset.parquet")
        feature_cols = [c for c in df.columns if c.startswith(("sent_", "msg_", "vol_"))]
    except FileNotFoundError:
        df, feature_cols = build()

    df = df.sort_values("date")  # CRITICAL: chronological order for TimeSeriesSplit
    X = df[feature_cols]
    y = df["target_up"].astype(int)

    print(f"rows: {len(df)} | class balance (up): {y.mean():.3f}")

    results = [
        evaluate("LogisticRegression",
                 LogisticRegression(max_iter=1000), X, y, needs_scaling=True),
    ]
    try:
        from lightgbm import LGBMClassifier
        results.append(
            evaluate("LightGBM",
                     LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, verbose=-1), X, y)
        )
    except ImportError:
        print("\n(lightgbm not installed — skipping)")

    print("\n=== summary ===")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
