# Sentiment-Driven Stock Signal Analyzer

**Core question:** Does aggregated social sentiment about a stock predict its
next-day return *better than chance*?

The goal of this project is to **learn ML fundamentals deeply** by building every
stage of the pipeline yourself — not to make money. The portfolio value is in the
rigor: honest validation, leakage-free features, and baseline-relative evaluation.

> ⚠️ **Read this first — the #1 trap.** It is very easy to accidentally "predict"
> the future by leaking it into your training data. This repo is structured to make
> that hard: timestamps are tracked everywhere, and modeling uses `TimeSeriesSplit`,
> never random K-fold. If your model looks amazing, assume leakage until proven
> otherwise.

## Pipeline

```
data/        -> collect prices (yfinance) + text (StockTwits) with timestamps
sentiment/   -> score text two ways: VADER (lexicon) vs FinBERT (transformer)
features/    -> aggregate sentiment per ticker per day, join to prices, NO leakage
models/      -> logistic regression -> LightGBM, evaluated against dumb baselines
backtest/    -> simple strategy vs buy-and-hold, with transaction costs
app/         -> Streamlit dashboard (portfolio centerpiece)
```

## Suggested order of work

1. `src/data/collect_prices.py`   — get this working first, it always works
2. `src/data/collect_stocktwits.py` — verify your API access EARLY
3. `src/sentiment/score.py`        — VADER baseline, then FinBERT
4. `src/features/build.py`         — the leakage-sensitive part
5. `src/models/train.py`           — baselines first, always
6. `src/backtest/run.py`
7. `app/dashboard.py`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your StockTwits token if you have one
```

## The honest-result clause

If your model can't beat the majority-class baseline, **say so in the README and
the dashboard.** "I built a rigorous pipeline and found the signal is weak" is a
stronger portfolio piece than a suspiciously profitable backtest.
