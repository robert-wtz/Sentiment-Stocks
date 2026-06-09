"""Phase 1a — collect daily price data. This one always works; start here.

Run:  python -m src.data.collect_prices
Output: data/raw/prices.parquet  (columns: date, ticker, open, high, low, close, volume)
"""
import pandas as pd
import yfinance as yf

from src.config import TICKERS, PRICE_START, PRICE_END, RAW


def fetch_prices(tickers=TICKERS, start=PRICE_START, end=PRICE_END) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [warn] no data for {t}")
            continue
        df = df.reset_index()
        # yfinance sometimes returns a MultiIndex on columns for single tickers
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns=str.lower)
        df["ticker"] = t
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
        print(f"  {t}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def add_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Next-day return + its sign. This is the LABEL.

    Leakage note: the target for day D uses close[D+1]. When you later join
    sentiment, the FEATURES for day D must use only info available up to D's close.
    """
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices["next_ret"] = prices["next_close"] / prices["close"] - 1
    prices["target_up"] = (prices["next_ret"] > 0).astype("Int64")
    return prices


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    prices = add_targets(fetch_prices())
    out = RAW / "prices.parquet"
    prices.to_parquet(out, index=False)
    print(f"saved {len(prices)} rows -> {out}")
