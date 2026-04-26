"""Live OHLCV streamer (Binance WebSocket).

Subscribes to the ``<symbol>@kline_<interval>`` stream for every
(pair, interval) in config, and appends *closed* candles to the same
Parquet partitions used by the historical loader.

We deliberately ignore in-progress candles — they'd add noise and the
schema would have to grow to mark them as provisional. Closed-candle-only
keeps the live stream a clean append to the historical dataset.
"""
from __future__ import annotations

import json
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import pandas as pd
from loguru import logger
from websocket import WebSocketApp  # via python-binance dependency tree

from src.config import Config, load_config
from src.utils.storage import (
    ohlcv_partition_path,
    utc_now_ms,
    write_ohlcv_partition,
)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="

# Flush buffered candles every N seconds. Frequent enough to feel "live"
# in downstream tooling, infrequent enough that we don't rewrite parquet
# files on every single candle close.
FLUSH_INTERVAL_S = 30


class LiveStreamer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.root = cfg.storage.root_path
        self.compression = cfg.storage.compression
        # Buffers keyed by (pair, interval). Lock guards mutation from the
        # WS callback thread vs the periodic flush thread.
        self._buffers: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self._lock = Lock()
        self._stop = False
        self._ws: WebSocketApp | None = None

    # ------------------------------------------------------------------
    #  Stream URL construction
    # ------------------------------------------------------------------
    def _stream_url(self) -> str:
        streams = [
            f"{p.lower()}@kline_{i}"
            for p in self.cfg.binance.pairs
            for i in self.cfg.binance.intervals
        ]
        return BINANCE_WS_BASE + "/".join(streams)

    # ------------------------------------------------------------------
    #  WebSocket callbacks
    # ------------------------------------------------------------------
    def _on_message(self, ws: WebSocketApp, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON WS frame: {raw[:120]}")
            return

        data = payload.get("data") or payload
        kline = data.get("k") if isinstance(data, dict) else None
        if not kline:
            return

        # k.x = "is this kline closed?". Skip in-progress.
        if self.cfg.binance.websocket.only_closed and not kline.get("x", False):
            return

        pair = kline["s"]
        interval = kline["i"]
        candle = {
            "open_time": pd.to_datetime(kline["t"], unit="ms", utc=True),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "close_time": pd.to_datetime(kline["T"], unit="ms", utc=True),
            "quote_volume": float(kline["q"]),
            "trades": int(kline["n"]),
            "taker_buy_base": float(kline["V"]),
            "taker_buy_quote": float(kline["Q"]),
            "ingested_at": utc_now_ms(),
        }
        with self._lock:
            self._buffers[(pair, interval)].append(candle)

    def _on_error(self, ws: WebSocketApp, err: Any) -> None:
        logger.error(f"WS error: {err}")

    def _on_close(self, ws: WebSocketApp, code: int, msg: str) -> None:
        logger.warning(f"WS closed (code={code}): {msg}")

    def _on_open(self, ws: WebSocketApp) -> None:
        n = len(self.cfg.binance.pairs) * len(self.cfg.binance.intervals)
        logger.success(f"WS open — subscribed to {n} streams")

    # ------------------------------------------------------------------
    #  Periodic flush
    # ------------------------------------------------------------------
    def _flush_loop(self) -> None:
        while not self._stop:
            time.sleep(FLUSH_INTERVAL_S)
            self._flush_once()

    def _flush_once(self) -> None:
        with self._lock:
            snapshot = {k: v for k, v in self._buffers.items() if v}
            self._buffers.clear()
        if not snapshot:
            return

        for (pair, interval), rows in snapshot.items():
            df = pd.DataFrame(rows)
            df = df.sort_values("open_time").reset_index(drop=True)
            # All rows in this batch *might* span a month boundary near the 1st.
            # Group by month and write each partition.
            df["_month"] = df["open_time"].dt.strftime("%Y-%m")
            for month, sub in df.groupby("_month"):
                sub = sub.drop(columns="_month")
                ts = pd.Timestamp(sub["open_time"].iloc[0])
                path = ohlcv_partition_path(self.root, pair, interval, ts)
                write_ohlcv_partition(sub, path, compression=self.compression)
            logger.info(f"flushed {len(df)} {pair} {interval} candles")

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def run(self) -> None:
        flush_thread = Thread(target=self._flush_loop, daemon=True)
        flush_thread.start()

        backoff = self.cfg.binance.websocket.reconnect_initial
        max_backoff = self.cfg.binance.websocket.reconnect_max

        while not self._stop:
            url = self._stream_url()
            logger.info(f"Connecting to {url[:80]}…")
            self._ws = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            # ping the server every 60s to keep NAT/firewalls happy
            self._ws.run_forever(ping_interval=60, ping_timeout=10)

            if self._stop:
                break
            logger.info(f"Reconnecting in {backoff}s…")
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

        # Final flush before exit
        self._flush_once()
        logger.success("Live streamer stopped cleanly.")

    def stop(self) -> None:
        self._stop = True
        if self._ws is not None:
            self._ws.close()


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    streamer = LiveStreamer(cfg)

    def _sigint(signum, frame):  # noqa: ARG001
        logger.info("Caught SIGINT — shutting down")
        streamer.stop()

    signal.signal(signal.SIGINT, _sigint)
    streamer.run()


if __name__ == "__main__":
    main()
