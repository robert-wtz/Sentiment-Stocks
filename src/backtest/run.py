"""Phase 5 — backtest the signal vs buy-and-hold, WITH transaction costs.

A backtest is just: "if I had acted on the signal, what would have happened?"
The discipline is to (a) only act on information available at decision time, and
(b) subtract realistic costs, or you'll fool yourself.

This is a teaching backtest, not a trading system. Strategy: go long for the next
day when predicted P(up) > threshold, else hold cash.

Run:  python -m src.backtest.run
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED

COST_PER_TRADE = 0.0005  # 5 bps each time the position changes


def walk_forward_signals(df, feature_cols, threshold=0.5, min_train=200):
    """Expanding-window predictions: predict day t using a model trained on < t."""
    df = df.sort_values("date").reset_index(drop=True)
    preds = np.full(len(df), np.nan)
    for t in range(min_train, len(df)):
        train = df.iloc[:t]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[feature_cols], train["target_up"].astype(int))
        preds[t] = model.predict_proba(df.iloc[[t]][feature_cols])[0, 1]
    df["pred_up"] = preds
    df["position"] = (df["pred_up"] > threshold).astype(float)
    return df.dropna(subset=["pred_up"])


def backtest(df):
    df = df.copy()
    df["turnover"] = df.groupby("ticker")["position"].diff().abs().fillna(0)
    df["cost"] = df["turnover"] * COST_PER_TRADE
    df["strat_ret"] = df["position"] * df["next_ret"] - df["cost"]

    summary = []
    for tic, g in df.groupby("ticker"):
        strat = (1 + g["strat_ret"]).prod() - 1
        hold = (1 + g["next_ret"]).prod() - 1
        summary.append({"ticker": tic, "strategy": strat, "buy_hold": hold,
                        "edge": strat - hold, "n_days": len(g)})
    return pd.DataFrame(summary)


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    feature_cols = [c for c in df.columns if c.startswith(("sent_", "msg_", "vol_"))]
    signaled = walk_forward_signals(df, feature_cols)
    print(backtest(signaled).to_string(index=False))
    print("\nReminder: a positive edge on a handful of tickers over a short window "
          "is NOT proof of a strategy. Check it survives more tickers, more time, "
          "and higher costs before believing it.")
