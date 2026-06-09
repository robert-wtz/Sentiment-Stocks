"""Central config. Edit TICKERS and dates here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Start small. More tickers = more text to collect = slower iteration.
TICKERS = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]

# History window for prices. Text history will be much shorter (APIs cap how far
# back you can pull), which is itself a real-world data limitation to reckon with.
PRICE_START = "2023-01-01"
PRICE_END = None  # None = today
