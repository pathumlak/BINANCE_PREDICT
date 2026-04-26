# Binance ML Predictor — Phase 1: Data Ingestion

This is the **Phase 1** deliverable of a research project on multimodal
crypto-price prediction (CNN over chart images + NLP over news + numerical
baselines). Phase 1 is *only* concerned with getting clean, reproducible
data on disk. No models live here yet.

---

## What this phase does

Three pipelines, all writing to a single Parquet dataset under `data/`:

| Pipeline | Source | Cadence | Output |
| --- | --- | --- | --- |
| Historical OHLCV | Binance REST `/api/v3/klines` | One-shot, resumable | `data/ohlcv/<PAIR>/<INTERVAL>/<YYYY-MM>.parquet` |
| Live OHLCV | Binance WebSocket `kline_<interval>` | Long-lived | Same partitions (appended) |
| News | RSS (CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine) + CryptoPanic | Run on cron every few minutes | `data/news/<source>/<YYYY-MM-DD>.parquet` |

**Tracked assets (configurable in `config.yaml`):** BTC, ETH, BNB, SOL, XRP — all vs USDT.
**Intervals:** 1m, 5m, 15m, 1h, 4h, 1d.
**History depth:** from each pair's listing date forward.

Every row carries **two timestamps** — the event time (`open_time` / `published_at`) and our ingestion time (`ingested_at`). This is non-negotiable: it's what lets later phases prove they aren't using information that wasn't yet public at prediction time.

---

## Project layout

```
binance stock prediction/
├── README.md
├── requirements.txt
├── config.yaml              # all knobs live here
├── .env.example             # copy to .env if you have API keys
├── src/
│   ├── config.py            # typed config loader
│   ├── ingest/
│   │   ├── historical.py    # REST loader (paginated, resumable)
│   │   ├── live.py          # WebSocket streamer
│   │   └── news.py          # RSS + CryptoPanic
│   └── utils/
│       └── storage.py       # Parquet schema + read/write helpers
├── scripts/
│   ├── fetch_historical.py  # entry point — historical pull
│   ├── stream_live.py       # entry point — live WS streamer
│   ├── fetch_news.py        # entry point — news pull
│   └── smoke_test.py        # tiny end-to-end check
└── data/                    # generated, .gitignored
    ├── ohlcv/
    └── news/
```

---

## Setup

```bash
# 1. (Optional but recommended) create a venv
python -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. (Optional) copy .env.example to .env and add API keys.
#    Phase 1 works with zero keys.
cp .env.example .env
```

---

## Smoke test (always run this first)

Pulls 30 days of daily BTC candles + one page of news. Should complete in well under a minute and proves your network can reach Binance + the RSS feeds.

```bash
python scripts/smoke_test.py
```

Expected tail of output:

```
SUCCESS  Smoke: OHLCV sanity checks passed
SUCCESS  Smoke: news no-lookahead invariant holds
SUCCESS  Smoke test complete.
```

If this passes you're cleared for the full historical pull.

---

## Full historical download

```bash
python scripts/fetch_historical.py
```

This will pull every pair × interval combination from `config.yaml`.
Rough timing on a home connection (be patient — 1m candles dominate):

| Pair | 1m candles ≈ | Approx time |
| --- | --- | --- |
| BTCUSDT (since Aug 2017) | ~4.5 M | 25–40 min |
| ETHUSDT (since Aug 2017) | ~4.5 M | 25–40 min |
| BNBUSDT (since Nov 2017) | ~4.3 M | 25–40 min |
| SOLUSDT (since Aug 2020) | ~3.0 M | 20–30 min |
| XRPUSDT (since May 2018) | ~4.1 M | 25–40 min |

**Resumable:** if you Ctrl-C halfway, just rerun. The loader checks the latest `open_time` already on disk and continues from there.

**Total disk:** roughly 4–6 GB of Parquet across all pairs/intervals.

---

## Live streaming

In a separate terminal once the historical pull is done (or even alongside it):

```bash
python scripts/stream_live.py
```

This connects to a single multiplexed WebSocket carrying all 30 streams (5 pairs × 6 intervals), buffers closed candles, and flushes them to the same monthly Parquet partitions every 30 seconds. Reconnects automatically with exponential backoff.

Stop it with `Ctrl-C` — it will run a final flush before exiting.

---

## News ingestion

One-shot:

```bash
python scripts/fetch_news.py
```

Long-lived CryptoPanic poller (good for a sidecar process or a `tmux` window):

```bash
python scripts/fetch_news.py --loop
```

Or schedule it: a cron entry like `*/5 * * * *` runs the one-shot fetch every five minutes. RSS feeds typically update only a few times per hour, so polling more often than that is wasted work.

---

## Inspecting the data

```python
import pandas as pd

# Read all 1h BTC candles in 2024
df = pd.read_parquet("data/ohlcv/BTCUSDT/1h/")
df = df[df.open_time.dt.year == 2024]
print(df.head())
print(df.describe())

# Read all news from the last 7 days
import glob
from pathlib import Path
files = sorted(Path("data/news").glob("*/*.parquet"))[-7:]
news = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(news[['source','title','tickers','published_at']].head(20))
```

---

## Schema

### OHLCV (`OHLCV_SCHEMA` in `src/utils/storage.py`)

| Column | Type | Notes |
| --- | --- | --- |
| `open_time` | timestamp[ms, UTC] | Candle start (Binance event time) |
| `open` / `high` / `low` / `close` | float64 | Prices in quote currency (USDT) |
| `volume` | float64 | Base-asset volume |
| `close_time` | timestamp[ms, UTC] | Candle end |
| `quote_volume` | float64 | Quote-asset volume |
| `trades` | int64 | Trade count in candle |
| `taker_buy_base` / `taker_buy_quote` | float64 | Aggressive buy volume |
| `ingested_at` | timestamp[ms, UTC] | When *we* pulled it |

### News

| Column | Type | Notes |
| --- | --- | --- |
| `id` | str | sha1(url)[:16] — stable across runs |
| `source` | str | "CoinDesk", "CryptoPanic", … |
| `title` | str | Headline |
| `url` | str | Article URL (dedup key) |
| `summary` | str | HTML-stripped excerpt (RSS only) |
| `published_at` | timestamp[UTC] | Per source |
| `ingested_at` | timestamp[UTC] | When *we* saw it |
| `tickers` | list[str] | Detected symbols (BTC/ETH/…) |
| `raw_sentiment` | object/null | Source-specific signal (e.g. CryptoPanic vote dict). Filled in Phase 4 by FinBERT/CryptoBERT. |

---

## Common gotchas

* **Binance geo-blocking.** `api.binance.com` is unreachable from some regions (US, UK). If you hit `451 Unavailable For Legal Reasons`, use a VPN or switch the base URL in `historical.py` / `live.py` to `api.binance.us` (US-compliant subset, fewer pairs).
* **Wall-clock vs candle clock.** Binance candles are timestamped in UTC. Don't reformat to local time before persisting — every downstream phase assumes UTC.
* **Don't pre-filter at ingestion time.** This phase deliberately stores *everything* a source returns, including articles that may be irrelevant. Filtering belongs in Phase 4 (sentiment) and Phase 5 (fusion). Storage is cheap; ingestion bugs are expensive.

---

# Phase 2 — Feature Engineering & Baseline Models

Phase 2 turns the raw OHLCV dataset from Phase 1 into a supervised
learning problem and trains four baselines through a leak-proof
walk-forward cross-validation harness.

## Scope (configurable in `scripts/train_baselines.py`)

* **Target** — binary direction over the next candle: `1` if `close[t+1] > close[t]`, else `0`.
* **Horizon** — 1 candle.
* **Pairs** — all 5 majors at the `1h` interval.
* **Baselines** — naive (majority + persistence), XGBoost, LSTM, PatchTST.

## Layout

```
src/
├── features/
│   ├── numerical.py     # returns, rolling stats, RSI/MACD/ATR/BB%B, vol features, time
│   ├── targets.py       # direction & return labels
│   └── dataset.py       # loads OHLCV → (X, y, close) per pair
├── eval/
│   ├── walk_forward.py  # expanding-window CV, gap-protected
│   ├── metrics.py       # accuracy/F1/MCC/log-loss + sign-aware Sharpe & PnL
│   └── dm_test.py       # Diebold-Mariano test (HLN small-sample correction)
├── models/
│   ├── base.py
│   ├── baseline_naive.py
│   ├── baseline_xgboost.py
│   ├── baseline_lstm.py
│   └── baseline_patchtst.py   # in-house compact PatchTST
└── runner.py            # orchestrator: per (model, pair) train + persist

scripts/
├── train_baselines.py     # full sweep
├── summarize_baselines.py # comparison table + DM significance matrix
└── smoke_phase2.py        # one-model smoke test (~30 s)
```

## Setup (additional deps)

```bash
pip install -r requirements.txt   # adds scikit-learn, xgboost, scipy, torch
```

## Smoke test (always run first)

```bash
python scripts/smoke_phase2.py
```

Runs XGBoost on BTCUSDT 1h with 3 folds. Should print fold metrics and end with `smoke OK`.

## Full sweep

```bash
# Naive + XGBoost across all 5 pairs (~5 min on CPU)
python scripts/train_baselines.py --models naive_majority naive_persistence xgboost

# Add deep models (LSTM ~5–10 min/pair on CPU, PatchTST ~10–15 min/pair on CPU)
python scripts/train_baselines.py --models lstm patchtst
```

Results land under `experiments/baselines/<PAIR>/<MODEL>/`:
* `predictions.parquet` — per-row `(open_time, fold, y_true, y_pred, y_prob, next_ret)`
* `metrics.json` — per-fold and aggregate metrics

## Aggregating results

```bash
python scripts/summarize_baselines.py
python scripts/summarize_baselines.py --pair BTCUSDT   # focus DM matrix on one pair
```

Prints a comparison table sorted by accuracy and a Diebold-Mariano
signed-p-value matrix per pair (negative p ⇒ row beats column;
`|p| < 0.05` ⇒ significant difference).

`experiments/baselines/summary.csv` is written for easy plotting.

## Reading the metrics

| Metric | Meaning | What's good |
| --- | --- | --- |
| `accuracy` | Hit rate on direction | Crypto: anything > 0.52 is *interesting*, > 0.55 is suspicious of a leak — re-check. |
| `mcc` | Matthews correlation coef | Less fooled by class imbalance; treat 0.05+ as a real signal. |
| `log_loss` | Calibration of `y_prob` | Lower = better. |
| `sharpe_per_bar` | Mean / std of (position × next_ret) | > 0 ⇒ profitable directionality before fees. Annualise for headline numbers. |
| `pnl_log` | Cumulative log-return of always-on long/short following predictions | Sign matters more than magnitude (no fees / sizing here). |

## Sanity rules of thumb

* If `accuracy > 0.6`, **assume a leak first.** Recheck `make_direction_label` and the warmup-row drop in `build_features`.
* If `xgboost < naive_majority`, your dataset has a near-pure label imbalance and the model is doing nothing. Check `y.mean()`.
* If `lstm == patchtst == 0.5`, training collapsed (often: features all NaN after standardise, or windowing produced empty arrays). Inspect `_LSTMNet.forward` / `_PatchEmbed` shapes.

## What's next — Phase 3 preview

Phase 3 introduces the chart-CNN: render candlestick windows as images,
train a small ResNet/EfficientNet head, learn embeddings reusable in
Phase 6's pattern-matching engine. The CNN's predictions plug into the
same evaluation harness built here, so we can DM-test it directly
against XGBoost / LSTM / PatchTST.
