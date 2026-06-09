"""Phase 3 — build the feature matrix for long-term sentiment signals.

Leakage rules (same as before, stricter horizon):
  - All features use .shift(1) so they only contain info from BEFORE the prediction date.
  - Rolling windows are trailing-only (never centered or forward).
  - Target for horizon H on day D = did close[D+H] > close[D]?
    This means the last H rows per ticker have no valid target and are dropped.

Run:  python -m src.features.build
Output: data/processed/dataset.parquet
"""
import pandas as pd

from src.config import RAW, PROCESSED, HORIZONS

SENT_COL = "sentiment_vader"


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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All trailing windows — only past information ever used."""
    df = df.sort_values(["ticker", "date"]).copy()
    grp = df.groupby("ticker")

    # Lag-1 sentiment and short momentum (same as before)
    df["sent_lag1"]       = grp["sent_mean"].shift(1)
    df["sent_momentum"]   = df["sent_mean"] - df["sent_lag1"]

    # Medium-term sentiment trend: rolling mean over past 10 and 30 trading days
    df["sent_roll10"]     = grp["sent_mean"].transform(lambda s: s.shift(1).rolling(10).mean())
    df["sent_roll30"]     = grp["sent_mean"].transform(lambda s: s.shift(1).rolling(30).mean())

    # Sentiment acceleration: is the 10-day trend rising faster than the 30-day?
    df["sent_accel"]      = df["sent_roll10"] - df["sent_roll30"]

    # Volume trend: is the stock getting talked about more over the past 10/30 days?
    df["vol_roll10"]      = grp["msg_volume"].transform(lambda s: s.shift(1).rolling(10).mean())
    df["vol_roll30"]      = grp["msg_volume"].transform(lambda s: s.shift(1).rolling(30).mean())
    df["vol_accel"]       = df["vol_roll10"] - df["vol_roll30"]

    # Sentiment vs price divergence: sentiment rising but price flat/falling = potential signal
    df["price_roll10"]    = grp["close"].transform(lambda s: s.shift(1).rolling(10).mean())
    df["price_roll30"]    = grp["close"].transform(lambda s: s.shift(1).rolling(30).mean())
    df["price_trend"]     = df["price_roll10"] / df["price_roll30"] - 1  # >0 = uptrend
    df["sent_price_div"]  = df["sent_accel"] - df["price_trend"]         # divergence

    return df


def add_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add one target column per horizon: did close go up H trading days from now?"""
    df = df.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        df[f"target_{h}d"] = (
            df.groupby("ticker")["close"]
            .shift(-h)                          # future close H days ahead
            .gt(df["close"])                    # True if higher than today
            .astype("Int64")
        )
    return df


def build() -> tuple[pd.DataFrame, list[str], list[str]]:
    prices = pd.read_parquet(RAW / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    scored = pd.read_parquet(PROCESSED / "scored.parquet")

    sent = daily_sentiment(scored)
    df = prices.merge(sent, on=["ticker", "date"], how="inner")
    df = add_features(df)
    df = add_targets(df, HORIZONS)

    feature_cols = [
        "sent_mean", "sent_std", "msg_volume",
        "sent_lag1", "sent_momentum",
        "sent_roll10", "sent_roll30", "sent_accel",
        "vol_roll10", "vol_roll30", "vol_accel",
        "price_trend", "sent_price_div",
    ]
    target_cols = [f"target_{h}d" for h in HORIZONS]

    df = df.dropna(subset=feature_cols)
    # Keep rows that have at least one valid target
    df = df[df[target_cols].notna().any(axis=1)]
    return df, feature_cols, target_cols


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset, cols, targets = build()
    dataset.to_parquet(PROCESSED / "dataset.parquet", index=False)
    print(f"dataset: {len(dataset)} rows, features: {cols}, targets: {targets}")
    print(dataset[["date", "ticker"] + targets].tail())
