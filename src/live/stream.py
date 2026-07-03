"""Binance WebSocket consumer for the paper-trading orchestrator.

The historical ingestor in :mod:`src.ingest.live` is a long-lived
script that writes Parquet partitions; here we want something different:
an async-friendly producer that pushes each *closed* candle into an
``asyncio.Queue`` so the FastAPI orchestrator can consume it.

Threading model
---------------
The official ``websocket-client`` library is synchronous and exposes
``WebSocketApp.run_forever``. We host it in a daemon thread; its
``on_message`` callback bridges to asyncio via
``loop.call_soon_threadsafe(queue.put_nowait, candle)``.

This pattern keeps us free of an extra async-websocket dependency, and
mirrors how the existing :class:`LiveStreamer` is structured so we don't
fork the websocket plumbing twice.

Auto-reconnect
--------------
``WebSocketApp.run_forever(reconnect=1)`` will retry every second with
its built-in backoff. We additionally wrap that call in a while-loop so
even hard failures (TLS errors, etc.) bring the stream back up.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

import pandas as pd
from loguru import logger
from websocket import WebSocketApp

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"


def _kline_to_candle(kline: dict) -> dict:
    """Convert a Binance kline payload to our internal candle dict.

    Includes the full Phase 1 OHLCV schema (taker_buy_base etc.) so that
    downstream feature engineering — which requires those columns to
    avoid being dropped by ``df.dropna()`` — has everything it needs.
    """
    return {
        "open_time":      pd.to_datetime(kline["t"], unit="ms", utc=True),
        "open":           float(kline["o"]),
        "high":           float(kline["h"]),
        "low":            float(kline["l"]),
        "close":          float(kline["c"]),
        "volume":         float(kline["v"]),
        "close_time":     pd.to_datetime(kline["T"], unit="ms", utc=True),
        "quote_volume":   float(kline.get("q", 0.0)),
        "trades":         int(kline.get("n", 0)),
        "taker_buy_base": float(kline.get("V", float(kline["v"]) * 0.5)),
        "taker_buy_quote": float(kline.get("Q", 0.0)),
    }


class CandleStream:
    """Async producer of closed BTC 1h candles (or whatever pair/interval).

    Usage::

        stream = CandleStream(pair="BTCUSDT", interval="1h")
        await stream.start()
        candle = await stream.queue.get()
        ...
        await stream.stop()
    """

    def __init__(self, pair: str = "BTCUSDT", interval: str = "1h",
                 max_queue_size: int = 64) -> None:
        self.pair = pair.upper()
        self.interval = interval
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue_size)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[WebSocketApp] = None
        self._stop_event = threading.Event()
        self._connected = False
        self._last_message_at: Optional[pd.Timestamp] = None
        self._messages_seen = 0

    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def messages_seen(self) -> int:
        return self._messages_seen

    @property
    def last_message_at(self) -> Optional[pd.Timestamp]:
        return self._last_message_at

    # ------------------------------------------------------------------
    def _url(self) -> str:
        return f"{BINANCE_WS_BASE}/{self.pair.lower()}@kline_{self.interval}"

    def _on_open(self, _ws: WebSocketApp) -> None:
        self._connected = True
        logger.success(f"[CandleStream] connected: {self.pair} {self.interval}")

    def _on_close(self, _ws: WebSocketApp, code: int, msg: str) -> None:
        self._connected = False
        logger.warning(f"[CandleStream] closed (code={code}): {msg}")

    def _on_error(self, _ws: WebSocketApp, err: object) -> None:
        logger.error(f"[CandleStream] error: {err}")

    def _on_message(self, _ws: WebSocketApp, raw: str) -> None:
        self._messages_seen += 1
        self._last_message_at = pd.Timestamp.utcnow()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"[CandleStream] non-JSON frame: {raw[:120]!r}")
            return

        kline = payload.get("k")
        if not kline:
            return

        candle = _kline_to_candle(kline)
        candle["pair"] = self.pair
        candle["interval"] = self.interval
        # ``x`` = "is this kline closed?". We forward EVERY event now
        # (with the marker) so the chart can animate live ticks while
        # the paper trader still filters for is_closed only.
        candle["is_closed"] = bool(kline.get("x", False))

        if self._loop is None:
            return
        # Hand off to the asyncio loop. ``put_nowait`` is OK because the
        # queue's maxsize gives us back-pressure; if it fills, we drop
        # the oldest entry rather than block the WS thread.
        try:
            self._loop.call_soon_threadsafe(self._enqueue, candle)
        except RuntimeError:
            # loop closed while we were dispatching — happens on shutdown
            pass

    def _enqueue(self, candle: dict) -> None:
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(candle)

    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Body of the daemon thread — reconnect loop around run_forever."""
        backoff = 1.0
        while not self._stop_event.is_set():
            self._ws = WebSocketApp(
                self._url(),
                on_open=self._on_open,
                on_close=self._on_close,
                on_error=self._on_error,
                on_message=self._on_message,
            )
            try:
                # ``ping_interval`` keeps the connection alive through NATs
                # and lets Binance gracefully time-out idle sessions.
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:                              # noqa: BLE001
                logger.error(f"[CandleStream] run_forever crashed: {e}")
            if self._stop_event.is_set():
                break
            # Exponential backoff capped at 60 s.
            sleep = min(60.0, backoff)
            logger.info(f"[CandleStream] reconnecting in {sleep:.1f}s")
            time.sleep(sleep)
            backoff = min(60.0, backoff * 2)

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.get_event_loop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"CandleStream-{self.pair}", daemon=True,
        )
        self._thread.start()
        logger.info(f"[CandleStream] started thread for {self.pair} {self.interval}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:                                   # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("[CandleStream] stopped")


# ---------------------------------------------------------------------------
class MulticastCandleStream:
    """One WebSocket to Binance, many consumers.

    Wraps :class:`CandleStream` and fans every tick out to any number of
    subscribed ``asyncio.Queue`` instances. Used by the FastAPI app so
    the paper-trading orchestrator AND every open Server-Sent-Events
    connection can each independently receive the same live feed
    without opening N sockets to Binance.
    """

    def __init__(self, pair: str = "BTCUSDT", interval: str = "1h") -> None:
        self._stream = CandleStream(pair=pair, interval=interval)
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._pump_task: Optional[asyncio.Task] = None
        self.pair = pair
        self.interval = interval

    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._stream.is_connected

    @property
    def messages_seen(self) -> int:
        return self._stream.messages_seen

    @property
    def n_subscribers(self) -> int:
        return len(self._subscribers)

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._pump_task is not None and not self._pump_task.done():
            return
        await self._stream.start()
        self._pump_task = asyncio.create_task(self._pump())
        logger.success(
            f"[MulticastCandleStream] started {self.pair} {self.interval}"
        )

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        await self._stream.stop()
        for q in list(self._subscribers):
            self._subscribers.discard(q)
        logger.info("[MulticastCandleStream] stopped")

    # ------------------------------------------------------------------
    def subscribe(self, maxsize: int = 128) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(q)

    # ------------------------------------------------------------------
    async def _pump(self) -> None:
        """Consume the wrapped stream and fan out to every subscriber."""
        try:
            while True:
                candle = await self._stream.queue.get()
                for q in list(self._subscribers):
                    if q.full():
                        # drop oldest to keep the newest — better latency
                        try:
                            _ = q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        q.put_nowait(candle)
                    except asyncio.QueueFull:
                        pass
        except asyncio.CancelledError:
            raise
