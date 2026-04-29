"""Aggregate scored news into per-bar sentiment feature parquet.

Output: ``data/features_sentiment/<PAIR>/1h.parquet`` — one row per
OHLCV bar carrying the columns documented in :mod:`src.nlp.aggregate`.

Usage:
    python scripts/build_sentiment_features.py
    python scripts/build_sentiment_features.py --pairs BTCUSDT --interval 1h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import timedelta  # noqa: E402

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


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-hours", type=int, default=24)
    args = parser.parse_args()

    news_root = cfg.storage.root_path / "news"
    feat_root = cfg.storage.root_path / "features_sentiment"
    feat_root.mkdir(parents=True, exist_ok=True)

    news = load_all_scored_news(news_root)
    if news.empty:
        logger.warning("No news on disk yet — run fetch_news.py + score_news.py first.")
        sys.exit(0)
    assert_news_publication_before_ingest(news)
    logger.info(f"loaded {len(news):,} news articles")

    cfg_agg = AggregateConfig(
        lookback=timedelta(hours=args.lookback_hours),
        include_macro=True,
    )

    for pair in args.pairs:
        try:
            ohlcv = load_ohlcv(pair, args.interval)
        except FileNotFoundError:
            logger.warning(f"{pair} {args.interval}: no OHLCV — skipped")
            continue
        feats = aggregate_for_bars(news, ohlcv[["open_time"]].copy(), pair, cfg_agg)

        # Persist
        out_dir = feat_root / pair
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.interval}.parquet"
        feats.to_parquet(out_path, compression="snappy", index=False)
        non_empty = (feats["sent_count"] > 0).sum()
        logger.success(
            f"{pair} {args.interval}: {len(feats):,} bars  "
            f"({non_empty:,} with at least one article)  →  {out_path}"
        )
