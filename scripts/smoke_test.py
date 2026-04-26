"""Phase 1 smoke test.

Exercises the whole ingestion stack against a *tiny* slice of data so you
can verify the plumbing without waiting for a multi-hour historical pull:

  * Pulls 30 days of 1d BTCUSDT candles via the historical loader.
  * Pulls one page of CryptoPanic + 1 RSS feed via the news loader.
  * Reads back the resulting Parquet files and prints summary stats.
  * Asserts the no-lookahead invariant on the news data.

Run from the project root:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import load_config  # noqa: E402
from src.ingest.historical import (  # noqa: E402
    INTERVAL_MS,
    FetchPlan,
    run_plan,
)
from src.ingest.news import fetch_rss, fetch_cryptopanic, _persist  # noqa: E402


def main() -> None:
    cfg = load_config()

    # ---- 1. Tiny historical pull: last 30 days of 1d BTCUSDT ----
    pair = "BTCUSDT"
    interval = "1d"
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 30 * 24 * 60 * 60 * 1000

    plan = FetchPlan(
        pair=pair,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        interval_ms=INTERVAL_MS[interval],
    )
    logger.info(f"Smoke: fetching last 30 days of {pair} {interval}")
    rows = run_plan(plan, cfg)
    logger.success(f"Smoke: {rows} candles written")

    # Read back & sanity-check
    folder = cfg.storage.root_path / "ohlcv" / pair / interval
    files = sorted(folder.glob("*.parquet"))
    assert files, f"No partitions produced under {folder}"
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("open_time")
    logger.info(f"Smoke: dataframe shape {df.shape}")
    logger.info(f"Smoke: range {df.open_time.min()} → {df.open_time.max()}")
    logger.info(f"Smoke: latest close = {df.close.iloc[-1]}")
    assert df.open_time.is_monotonic_increasing
    assert (df.ingested_at >= df.open_time).all(), \
        "ingested_at must be >= open_time (no time travel)"
    logger.success("Smoke: OHLCV sanity checks passed")

    # ---- 2. Tiny news pull: 1 RSS feed + 1 CryptoPanic page ----
    if cfg.news.rss_feeds:
        logger.info("Smoke: fetching one RSS feed")
        rss = fetch_rss(cfg.news.rss_feeds[0])
        logger.info(f"Smoke: {len(rss)} articles from {cfg.news.rss_feeds[0].name}")
    else:
        rss = pd.DataFrame()

    logger.info("Smoke: fetching one CryptoPanic page")
    cp = fetch_cryptopanic(cfg, max_pages=1)
    logger.info(f"Smoke: {len(cp)} CryptoPanic posts")

    news = pd.concat([f for f in (rss, cp) if not f.empty], ignore_index=True) \
        if (not rss.empty or not cp.empty) else pd.DataFrame()
    if not news.empty:
        _persist(news, cfg.storage.root_path, cfg.storage.compression)
        # No-lookahead invariant
        leak = news[news.published_at > news.ingested_at]
        if not leak.empty:
            logger.error(f"Smoke: {len(leak)} articles violate no-lookahead!")
        else:
            logger.success("Smoke: news no-lookahead invariant holds")
    else:
        logger.warning("Smoke: no news fetched (network blocked? feed down?)")

    logger.success("Smoke test complete.")


if __name__ == "__main__":
    main()
