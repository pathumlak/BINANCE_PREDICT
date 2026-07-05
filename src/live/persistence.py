"""Persist live-stream closed candles and their CNN embeddings to disk.

Purpose
-------
Without this module, every live bar the paper trader receives lives only
in memory — restart the dashboard and everything is forgotten, forcing
a manual ``scripts/refresh_all.py --only ohlcv`` before predictions
have honest features again.

With this module, each closed candle is written straight into the same
canonical Parquet layout Phase 1 uses (``data/ohlcv/<PAIR>/<INTERVAL>/<YYYY-MM>.parquet``),
and each new CNN embedding is appended to
``experiments/embeddings/<PAIR>/<INTERVAL>/candle/embeddings.parquet``.
Restarting the dashboard now finds all bars-so-far already on disk.

Design notes
------------
* **Buffered writes.** We flush once every ``FLUSH_INTERVAL_S`` seconds
  or when the buffer holds more than ``FLUSH_MAX_ROWS`` rows — writing
  every single hour would spin the disk unnecessarily.
* **Same-schema safety.** We use the Phase 1 helpers
  (:func:`ohlcv_partition_path`, :func:`write_ohlcv_partition`) so the
  live stream is byte-compatible with what ``fetch_historical.py``
  would produce. Downstream loaders don't need to know the origin.
* **Embedding append.** Read-modify-write of the whole embeddings
  parquet each flush is fine at ~76 K rows × 128 dims (a few MB). If
  the dataset grows to millions of rows we'd want an append-optimised
  format, but this is a research artefact.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.utils.storage import (
    ohlcv_partition_path,
    utc_now_ms,
    write_ohlcv_partition,
)

FLUSH_INTERVAL_S = 60           # write to disk at most once per minute
FLUSH_MAX_ROWS = 32             # …or when buffer hits this many rows


# ---------------------------------------------------------------------------
class LivePersistor:
    """Thread-safe append-only persistence for candles + CNN embeddings."""

    def __init__(self, pair: str = "BTCUSDT", interval: str = "1h",
                 encoder: str = "candle") -> None:
        cfg = load_config()
        self.pair = pair
        self.interval = interval
        self.encoder = encoder
        self._root = cfg.storage.root_path
        self._compression = cfg.storage.compression
        self._emb_path = (
            PROJECT_ROOT / "experiments" / "embeddings"
            / pair / interval / encoder / "embeddings.parquet"
        )

        self._lock = threading.Lock()
        self._candle_buf: list[dict] = []
        self._emb_buf: list[tuple[pd.Timestamp, np.ndarray]] = []
        self._last_flush = datetime.now(timezone.utc)

        self.n_candles_persisted = 0
        self.n_embeddings_persisted = 0
        self.last_persist_at: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    def enqueue_candle(self, candle: dict) -> None:
        """Buffer a closed candle. Flushes automatically when big/old."""
        with self._lock:
            self._candle_buf.append(_normalise_candle(candle))
            self._maybe_flush_locked()

    def enqueue_embedding(self, open_time: pd.Timestamp, emb: np.ndarray) -> None:
        """Buffer a 128-dim embedding for a bar. Same flush policy."""
        with self._lock:
            self._emb_buf.append((pd.Timestamp(open_time), emb.astype(np.float32).copy()))
            self._maybe_flush_locked()

    def flush(self) -> None:
        """Force a flush of any buffered rows."""
        with self._lock:
            self._flush_locked()

    # ------------------------------------------------------------------
    def _maybe_flush_locked(self) -> None:
        age = (datetime.now(timezone.utc) - self._last_flush).total_seconds()
        big = (len(self._candle_buf) + len(self._emb_buf)) >= FLUSH_MAX_ROWS
        if big or age >= FLUSH_INTERVAL_S:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._candle_buf:
            self._flush_candles_locked()
        if self._emb_buf:
            self._flush_embeddings_locked()
        self._last_flush = datetime.now(timezone.utc)

    def _flush_candles_locked(self) -> None:
        rows = self._candle_buf
        self._candle_buf = []
        df = pd.DataFrame(rows)
        if df.empty:
            return
        # Group by month partition so a batch spanning a boundary
        # writes to the right file each.
        df["_month"] = df["open_time"].dt.strftime("%Y-%m")
        for month, sub in df.groupby("_month", sort=True):
            sub = sub.drop(columns="_month")
            ts = pd.Timestamp(sub["open_time"].iloc[0])
            path = ohlcv_partition_path(self._root, self.pair, self.interval, ts)
            write_ohlcv_partition(sub, path, compression=self._compression)
            self.n_candles_persisted += len(sub)
        self.last_persist_at = pd.Timestamp.utcnow()

    def _flush_embeddings_locked(self) -> None:
        rows = self._emb_buf
        self._emb_buf = []
        if not rows:
            return

        cols = [f"e{i}" for i in range(rows[0][1].shape[0])]
        new_df = pd.DataFrame([r[1] for r in rows], columns=cols)
        new_df.insert(0, "open_time", [pd.Timestamp(r[0]) for r in rows])

        self._emb_path.parent.mkdir(parents=True, exist_ok=True)
        if self._emb_path.exists():
            try:
                existing = pd.read_parquet(self._emb_path)
                existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
                combined = pd.concat([existing, new_df], ignore_index=True)
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
        combined = (
            combined.sort_values("open_time")
            .drop_duplicates(subset=["open_time"], keep="last")
            .reset_index(drop=True)
        )
        combined.to_parquet(self._emb_path, compression="snappy", index=False)
        self.n_embeddings_persisted += len(rows)
        self.last_persist_at = pd.Timestamp.utcnow()

    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            return {
                "candles_persisted": self.n_candles_persisted,
                "embeddings_persisted": self.n_embeddings_persisted,
                "candles_buffered": len(self._candle_buf),
                "embeddings_buffered": len(self._emb_buf),
                "last_persist_at": (
                    self.last_persist_at.isoformat()
                    if self.last_persist_at is not None else None
                ),
            }


# ---------------------------------------------------------------------------
def _normalise_candle(candle: dict) -> dict:
    """Coerce a live candle into the canonical Phase 1 OHLCV schema.

    Missing optional fields are back-filled with sensible defaults so
    the row is bit-compatible with ``fetch_historical.py`` output.
    """
    open_time = pd.Timestamp(candle["open_time"])
    if open_time.tzinfo is None:
        open_time = open_time.tz_localize("UTC")
    else:
        open_time = open_time.tz_convert("UTC")
    close_time = candle.get("close_time")
    if close_time is None:
        close_time = open_time + pd.Timedelta(minutes=59, seconds=59)
    else:
        close_time = pd.Timestamp(close_time)
        if close_time.tzinfo is None:
            close_time = close_time.tz_localize("UTC")
        else:
            close_time = close_time.tz_convert("UTC")

    volume = float(candle.get("volume", 0.0))
    return {
        "open_time": open_time,
        "open":   float(candle.get("open",   0.0)),
        "high":   float(candle.get("high",   0.0)),
        "low":    float(candle.get("low",    0.0)),
        "close":  float(candle.get("close",  0.0)),
        "volume": volume,
        "close_time":    close_time,
        "quote_volume":  float(candle.get("quote_volume",  0.0)),
        "trades":        int(candle.get("trades",  0)),
        "taker_buy_base":  float(candle.get("taker_buy_base",  volume * 0.5)),
        "taker_buy_quote": float(candle.get("taker_buy_quote", 0.0)),
        "ingested_at":   utc_now_ms(),
    }
