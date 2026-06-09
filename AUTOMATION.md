# Automation — GitHub Actions + Pages

Everything runs through GitHub. No n8n, no local machine left on, no tunnels.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  GitHub Actions                         │
│                                                         │
│  Triggers:                                              │
│    cron       weekdays 22:30 UTC (after market close)   │
│    manual     Actions tab → "Run workflow"              │
│    push       if src/ scripts/ requirements.txt changes │
│    pull_request  smoke-test on synthetic data           │
│                                                         │
│  Jobs:                                                  │
│    pipeline → collect → score → features                │
│             → train → predict → build HTML              │
│             → commit parquet + index.html → Pages       │
│    deploy   → GitHub Pages (your portfolio link)        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              https://<user>.github.io/<repo>/
              (static HTML, self-contained, always current)
```

## One-time GitHub setup

1. **Push the repo** to GitHub (public or private).

2. **Enable GitHub Pages**
   Settings → Pages → Source: **GitHub Actions**
   *(not "Deploy from a branch" — the workflow uses the `deploy-pages` action)*

3. **Add secrets** (Settings → Secrets and variables → Actions → New repository secret)

   | Secret | Value | Required? |
   |--------|-------|-----------|
   | `STOCKTWITS_TOKEN` | Your StockTwits access token | Only for live mode |

   No other secrets needed — the workflow uses the built-in `GITHUB_TOKEN`
   for committing and deploying.

## How to trigger a run

**Automatic:** Just push the repo. The cron fires weekdays at 22:30 UTC.

**Manual (run now):**
Actions tab → "sentiment-pipeline" → "Run workflow" → choose mode:
- `live` — real StockTwits + yfinance data (needs the secret)
- `planted` — synthetic data with a planted signal (great for testing)
- `random` — synthetic random data (verifies the model finds nothing)

**On code change:** Editing anything under `src/`, `scripts/`, or
`requirements.txt` on `main` triggers a run automatically.

**On PR:** Always runs with `planted` synthetic data — no secrets needed,
no real API calls.

## Reading run results (no dashboard visit required)

Every run writes a **job summary** visible directly in the Actions tab:

```
## Sentiment Pipeline — Run Summary

| Metric      | Value  |
|-------------|--------|
| CV Accuracy | 61.9%  |
| Baseline    | 52.7%  |
| Edge        | +9.2%  |
| ROC AUC     | 0.646  |
| Rows        | 994    |
| Tickers     | AAPL, TSLA |
```

Followed by the deployed dashboard URL. You see this without opening the dashboard.

## What gets committed back to the repo

Each run commits:
```
data/processed/dataset.parquet       full feature matrix
data/processed/predictions.parquet   latest prediction per ticker
reports/index.html                   regenerated dashboard (self-contained HTML)
reports/metrics.json                 headline metrics
```

This means `git log` doubles as a run log, and you can `git diff` any two runs
to see exactly what changed in the data.

## Offline-safe dashboard

If `reports/vendor/chart.umd.min.js` exists, the build script inlines it so the
dashboard works with zero internet — no CDN dependency.
The file degrades gracefully either way: if the chart library is missing, the KPI
cards and signal table still render; only the charts show a short notice.

To vendor chart.js once (run locally, then commit):
```bash
mkdir -p reports/vendor
# download https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.min.js
# save as reports/vendor/chart.umd.min.js
git add -f reports/vendor/chart.umd.min.js
```

## Local run (no Actions, no internet needed)

```bash
pip install -r requirements.txt

# Smoke test with synthetic data:
USE_SYNTHETIC=planted python -m scripts.run_pipeline
open reports/index.html

# Real data:
python -m scripts.run_pipeline
```
