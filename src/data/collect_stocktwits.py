"""Phase 1b — collect text messages per ticker.

⚠️ VERIFY ACCESS EARLY. StockTwits restricted their public API; the endpoint below
may require an approved app / OAuth token (set STOCKTWITS_TOKEN in .env). If you get
401/403/429, that's expected — see the fallback note at the bottom of this file.

The collector is deliberately behind a tiny interface (`fetch_messages`) so you can
swap in Reddit or news headlines later WITHOUT touching sentiment/features/models.

Run:  python -m src.data.collect_stocktwits
Output: data/raw/messages.parquet (columns: id, ticker, created_at, body, source)
"""
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import TICKERS, RAW

load_dotenv()
TOKEN = os.getenv("STOCKTWITS_TOKEN", "").strip()
BASE = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


def fetch_messages(symbol: str, max_id: int | None = None) -> list[dict]:
    """Return a list of normalized message dicts for one ticker (one page).

    Normalized schema: id, ticker, created_at (UTC), body, source.
    Keeping this schema stable is what makes the source swappable.
    """
    params = {}
    if max_id:
        params["max"] = max_id
    if TOKEN:
        params["access_token"] = TOKEN

    resp = requests.get(BASE.format(symbol=symbol), params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(
            f"StockTwits returned {resp.status_code} for {symbol}. "
            f"If 401/403, your access likely needs an approved token. "
            f"If 429, you are rate-limited — back off. Body: {resp.text[:200]}"
        )
    data = resp.json()
    out = []
    for m in data.get("messages", []):
        out.append(
            {
                "id": m["id"],
                "ticker": symbol,
                "created_at": pd.to_datetime(m["created_at"], utc=True),
                "body": m["body"],
                "source": "stocktwits",
            }
        )
    return out


def collect(tickers=TICKERS, pages_per_ticker: int = 3) -> pd.DataFrame:
    rows: list[dict] = []
    for t in tickers:
        max_id = None
        for page in range(pages_per_ticker):
            try:
                batch = fetch_messages(t, max_id=max_id)
            except RuntimeError as e:
                print(f"  [stop] {t} page {page}: {e}")
                break
            if not batch:
                break
            rows.extend(batch)
            max_id = min(m["id"] for m in batch) - 1  # paginate backwards in time
            print(f"  {t} page {page}: +{len(batch)} (total {len(rows)})")
            time.sleep(1.0)  # be polite / avoid rate limits
    return pd.DataFrame(rows)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        print(
            "No messages collected. Verify STOCKTWITS_TOKEN, or swap to a fallback "
            "source (see collect_reddit_fallback.py) — the rest of the pipeline is "
            "source-agnostic as long as you produce the same columns."
        )
    else:
        df = df.drop_duplicates("id")
        out = RAW / "messages.parquet"
        df.to_parquet(out, index=False)
        print(f"saved {len(df)} messages -> {out}")

# ---------------------------------------------------------------------------
# FALLBACK if StockTwits access fails:
#   Produce a DataFrame with the SAME columns
#   (id, ticker, created_at[UTC], body, source) from any source:
#     - Reddit via `praw` (r/wallstreetbets, search by ticker)
#     - News headlines via an RSS feed or Finnhub free tier
#   Save to data/raw/messages.parquet and everything downstream just works.
# ---------------------------------------------------------------------------
