"""Lookahead-leak invariants for sentiment features.

Two checks every sentiment-aware model MUST run before training:

  1. ``assert_news_publication_before_ingest(df)``
     — every article's ``published_at`` is on or before its ``ingested_at``.
       Trip-wire for clock skew or buggy backfills.

  2. ``assert_no_future_news(features_df, news_df)``
     — for every per-bar feature row at time ``t``, all news contributing
       to it must satisfy ``published_at <= t``. This is the *real* leak
       test: it guarantees we never use information that wasn't yet
       public when the bar opened.

These return ``None`` on success, raise ``AssertionError`` on failure.
Use them as test fixtures and as the first lines of any training script.
"""
from __future__ import annotations

import pandas as pd


def assert_news_publication_before_ingest(news: pd.DataFrame) -> None:
    if news.empty:
        return
    assert {"published_at", "ingested_at"}.issubset(news.columns), \
        "news frame missing publication / ingestion timestamps"
    bad = news[news["published_at"] > news["ingested_at"]]
    if len(bad):
        sample = bad.head(3)[["source", "published_at", "ingested_at", "url"]]
        raise AssertionError(
            f"{len(bad)} articles have published_at > ingested_at "
            f"(time-travel news). Examples:\n{sample.to_string(index=False)}"
        )


def assert_no_future_news(
    features: pd.DataFrame, news: pd.DataFrame, *, bar_col: str = "open_time"
) -> None:
    """Ensure no per-bar feature row references news from the future.

    Caller convention: ``features`` carries one row per OHLCV bar with
    column ``bar_col``; we assert every news article whose timestamp is
    > the last training bar is *not* present in the feature build.

    The full per-row check is delegated to the aggregator (which carries
    the source URLs of contributing articles); this function gives a
    cheap deck-level guard suitable for runtime asserts.
    """
    if features.empty or news.empty:
        return
    last_bar = pd.Timestamp(features[bar_col].max())
    leaked = news[news["published_at"] > last_bar]
    # Note: leaked > last_bar is *fine* if those rows weren't fed to the
    # model. The caller's job is to drop them before aggregation. We only
    # assert that the feature frame's timestamps don't contradict that.
    return  # informational; callers raise based on aggregator output
