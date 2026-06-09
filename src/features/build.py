"""Phase 3 — build the daily feature matrix. THE LEAKAGE-SENSITIVE PART.

Read this carefully — it's the heart of the learning.

Two leakage hazards handled here:
  1. INTRADAY CUTOFF. A message posted at 3pm on day D is fine to use for predicting
     day D+1. But a message posted after market close on day D really belongs to D+1's
     trading session. To keep the starter simple we assign each message to the trading
     date by its calendar (UTC) date and predict the NEXT day's return — so all of a
     day's text is used to predict the following day. Good enough to start; a stricter
     version aligns to market close in US/Eastern.
  2. ROLLING FEATURES must only look BACKWARD. We use .shift() and trailing windows,
     never centered or forward windows.

Run:  python -m src.features.build
Output: data/processed/dataset.parquet (one row per ticker per day, features + target)
"""
import pandas as pd

from src.config import RAW, PROCESSED

SENT_COL = "sentiment_vader"  # switch to sentiment_finbert to compare


def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = scored["created_at"].dt.tz_convert("UTC").dt.date
    g = scored.groupby(["ticker", "date"])
    out = g.agg(
        sent_mean=(SENT_COL, "mean"),
        sent_std=(SENT_COL, "std"),
        msg_volume=(SENT_COL, "size"),
    ).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    out["sent_std"] = out["sent_std"].fillna(0.0)
    return out


def add_backward_features(df: pd.DataFrame) -> pd.DataFrame:
    """All trailing — only past information. Sorted by date per ticker first."""
    df = df.sort_values(["ticker", "date"]).copy()
    grp = df.groupby("ticker")
    df["sent_mean_lag1"] = grp["sent_mean"].shift(1)
    df["sent_momentum"] = df["sent_mean"] - df["sent_mean_lag1"]
    df["sent_mean_roll3"] = grp["sent_mean"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    df["vol_roll3"] = grp["msg_volume"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    return df


def build() -> pd.DataFrame:
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    scored = pd.read_parquet(PROCESSED / "scored.parquet")

    sent = add_backward_features(daily_sentiment(scored))

    # Inner join: only keep ticker-days where we have BOTH price and text.
    df = prices.merge(sent, on=["ticker", "date"], how="inner")

    feature_cols = [
        "sent_mean", "sent_std", "msg_volume",
        "sent_mean_lag1", "sent_momentum", "sent_mean_roll3", "vol_roll3",
    ]
    df = df.dropna(subset=feature_cols + ["target_up"])
    return df, feature_cols


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset, cols = build()
    dataset.to_parquet(PROCESSED / "dataset.parquet", index=False)
    print(f"dataset: {len(dataset)} rows, features: {cols}")
    print(dataset[["date", "ticker", "target_up"] + cols].tail())
