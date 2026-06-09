"""Phase 2 — score each message's sentiment. THE ML CORE of the learning project.

Build TWO scorers and compare them. That comparison (lexicon vs transformer) is
where you learn when model complexity actually pays off.

Run:  python -m src.sentiment.score
Output: data/processed/scored.parquet (messages + sentiment_vader [, sentiment_finbert])
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import RAW, PROCESSED

_vader = SentimentIntensityAnalyzer()


def score_vader(texts: list[str]) -> list[float]:
    """Lexicon baseline. Returns compound score in [-1, 1]. Fast, interpretable."""
    return [_vader.polarity_scores(t or "")["compound"] for t in texts]


def score_finbert(texts: list[str]) -> list[float]:
    """Transformer scorer. Heavier — install torch + transformers first.

    Learning goals here: tokenization, batching, model outputs as probabilities.
    Returns a signed score = P(positive) - P(negative) in [-1, 1] for comparability
    with VADER.
    """
    from transformers import pipeline  # local import so VADER path needs no torch

    clf = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
    )
    scores = []
    for i in range(0, len(texts), 32):  # batch to keep memory sane
        batch = [t or "" for t in texts[i : i + 32]]
        for r in clf(batch):
            label = r["label"].lower()
            s = r["score"]
            scores.append(s if label == "positive" else -s if label == "negative" else 0.0)
    return scores


def run(use_finbert: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(RAW / "messages.parquet")
    df["sentiment_vader"] = score_vader(df["body"].tolist())
    if use_finbert:
        df["sentiment_finbert"] = score_finbert(df["body"].tolist())
    return df


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_df = run(use_finbert=False)  # flip to True once torch/transformers installed
    out = PROCESSED / "scored.parquet"
    out_df.to_parquet(out, index=False)
    print(f"scored {len(out_df)} messages -> {out}")
