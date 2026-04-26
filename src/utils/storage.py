"""Parquet read/write helpers.

Layout convention (Phase 1):
  data/ohlcv/<PAIR>/<INTERVAL>/<YYYY-MM>.parquet
  data/news/<source>/<YYYY-MM-DD>.parquet

Each OHLCV row carries TWO timestamps:
  - open_time:    when the candle started (Binance's event time)
  - ingested_at:  when WE pulled it (used later to detect lookahead bugs)

News rows likewise carry both ``published_at`` and ``ingested_at``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Canonical OHLCV schema. Pinning this prevents silent dtype drift between
# the historical loader (REST) and the live streamer (WebSocket).
OHLCV_SCHEMA = pa.schema(
    [
        ("open_time", pa.timestamp("ms", tz="UTC")),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.float64()),
        ("close_time", pa.timestamp("ms", tz="UTC")),
        ("quote_volume", pa.float64()),
        ("trades", pa.int64()),
        ("taker_buy_base", pa.float64()),
        ("taker_buy_quote", pa.float64()),
        ("ingested_at", pa.timestamp("ms", tz="UTC")),
    ]
)


def ohlcv_partition_path(root: Path, pair: str, interval: str, ts: pd.Timestamp) -> Path:
    """Path to the monthly Parquet partition that should hold ``ts``."""
    yyyy_mm = ts.strftime("%Y-%m")
    return root / "ohlcv" / pair / interval / f"{yyyy_mm}.parquet"


def news_partition_path(root: Path, source: str, ts: pd.Timestamp) -> Path:
    """Path to the daily Parquet partition for a given source + date."""
    yyyy_mm_dd = ts.strftime("%Y-%m-%d")
    safe = source.lower().replace(" ", "_")
    return root / "news" / safe / f"{yyyy_mm_dd}.parquet"


def write_ohlcv_partition(
    df: pd.DataFrame, path: Path, compression: str = "snappy"
) -> None:
    """Write or merge an OHLCV partition.

    If the file already exists we read it, concatenate, and de-dup on
    ``open_time`` keeping the *latest* ``ingested_at`` — this makes the
    function safe to call repeatedly (e.g. when resuming a download).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    df = (
        df.sort_values(["open_time", "ingested_at"])
        .drop_duplicates(subset=["open_time"], keep="last")
        .reset_index(drop=True)
    )

    table = pa.Table.from_pandas(df, schema=OHLCV_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression=compression)


def write_news_partition(
    df: pd.DataFrame, path: Path, compression: str = "snappy"
) -> None:
    """Write or merge a news partition, deduping on article URL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    df = (
        df.sort_values(["published_at", "ingested_at"])
        .drop_duplicates(subset=["url"], keep="first")
        .reset_index(drop=True)
    )
    df.to_parquet(path, compression=compression, index=False)


def latest_open_time(root: Path, pair: str, interval: str) -> pd.Timestamp | None:
    """Return the most recent ``open_time`` already on disk for (pair, interval).

    Used by the historical loader to resume rather than re-download from scratch.
    Returns ``None`` if no partitions exist yet.
    """
    folder = root / "ohlcv" / pair / interval
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.parquet"))
    if not files:
        return None
    last_partition = pd.read_parquet(files[-1], columns=["open_time"])
    if last_partition.empty:
        return None
    # Parquet round-trip already restores tz=UTC on the column, so wrap
    # without re-passing tz (that combo raises). Localise only if naive.
    ts = pd.Timestamp(last_partition["open_time"].max())
    return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def utc_now_ms() -> pd.Timestamp:
    """``datetime.now(UTC)`` rounded to milliseconds — matches OHLCV_SCHEMA."""
    return pd.Timestamp(datetime.now(timezone.utc)).floor("ms")
