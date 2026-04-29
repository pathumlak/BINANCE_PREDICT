"""Phase 4 smoke test — sentiment ensemble + aggregator end-to-end.

Steps:
  1. Confirm both backbones load and score five hand-picked headlines.
  2. Ensure leakage invariant holds on the news on disk.
  3. Aggregate per-1h-bar sentiment features for BTCUSDT.

Should run in ~2 minutes the first time (model downloads), <30 s thereafter.

    python scripts/smoke_phase4.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import load_config  # noqa: E402
from src.features.dataset import load_ohlcv  # noqa: E402
from src.nlp.aggregate import (  # noqa: E402
    AggregateConfig,
    aggregate_for_bars,
    load_all_scored_news,
)
from src.nlp.leakage import assert_news_publication_before_ingest  # noqa: E402
from src.nlp.sentiment import score_texts  # noqa: E402


HEADLINES = [
    "Bitcoin surges past $80,000 on record ETF inflows",
    "Ethereum L2 activity hits all-time high as fees drop",
    "Major exchange hack drains $200M from user wallets",
    "Regulators warn of tighter crypto rules in Q3",
    "Solana network outage halts trading for two hours",
]


def main():
    # 1) Sentiment
    logger.info("loading CryptoBERT + FinBERT… (first run downloads ~880MB)")
    scores = score_texts(HEADLINES)
    logger.success("ensemble scores:")
    for s in scores:
        print(f"  ens={s.ensemble:+.3f}  conf={s.confidence:.2f}  "
              f"cb={s.cryptobert:+.2f}  fb={s.finbert:+.2f}  | {s.text}")

    # 2) Leakage check on whatever news is on disk
    cfg = load_config()
    news_root = cfg.storage.root_path / "news"
    news = load_all_scored_news(news_root)
    if news.empty:
        logger.warning("no news on disk yet — run fetch_news.py first to test leakage + aggregator")
        return
    assert_news_publication_before_ingest(news)
    scored = news[news.get("sentiment").notna()] if "sentiment" in news.columns else pd.DataFrame()
    logger.success(f"news on disk: {len(news):,} total, {len(scored):,} already scored")

    # 3) Aggregate for BTCUSDT
    if scored.empty:
        logger.warning("no SCORED articles yet — run scripts/score_news.py first to test aggregator")
        return
    try:
        ohlcv = load_ohlcv("BTCUSDT", "1h")
    except FileNotFoundError:
        logger.warning("BTCUSDT 1h OHLCV not on disk — skipping aggregation step")
        return
    feats = aggregate_for_bars(scored, ohlcv[["open_time"]].copy(),
                               pair="BTCUSDT",
                               cfg=AggregateConfig(lookback=timedelta(hours=24)))
    nz = (feats["sent_count"] > 0).sum()
    logger.success(f"aggregated to {len(feats):,} bars  ({nz:,} non-empty)")
    print(feats[feats["sent_count"] > 0].head(5).to_string(index=False))
    logger.success("smoke phase 4 OK")


if __name__ == "__main__":
    main()
