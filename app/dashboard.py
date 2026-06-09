"""Portfolio centerpiece — Streamlit dashboard.

Run:  streamlit run app/dashboard.py

Shows, per ticker: latest sentiment, the model's current prediction, and — most
importantly — its HONEST historical accuracy vs the baseline. Don't hide weak
results; framing them well is the portfolio win.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED  # noqa: E402

st.set_page_config(page_title="Sentiment Stock Signal", layout="wide")
st.title("📈 Sentiment-Driven Stock Signal")
st.caption("A learning project on whether social sentiment predicts next-day moves. "
           "Results are reported honestly — including when the signal is weak.")

dataset_path = PROCESSED / "dataset.parquet"
if not dataset_path.exists():
    st.warning("Run the pipeline first: collect_prices → collect_stocktwits → "
               "sentiment.score → features.build")
    st.stop()

df = pd.read_parquet(dataset_path)
df["date"] = pd.to_datetime(df["date"])

tickers = sorted(df["ticker"].unique())
tic = st.selectbox("Ticker", tickers)
g = df[df["ticker"] == tic].sort_values("date")

c1, c2, c3 = st.columns(3)
c1.metric("Latest sentiment (mean)", f"{g['sent_mean'].iloc[-1]:+.3f}")
c2.metric("Message volume (latest day)", int(g["msg_volume"].iloc[-1]))
c3.metric("Days of data", len(g))

st.subheader("Sentiment over time")
st.line_chart(g.set_index("date")[["sent_mean"]])

st.subheader("Price (close)")
st.line_chart(g.set_index("date")[["close"]])

st.info("To wire in live predictions, import your trained model and the latest "
        "feature row here. Keep showing the baseline-relative accuracy so viewers "
        "can judge the signal honestly.")
