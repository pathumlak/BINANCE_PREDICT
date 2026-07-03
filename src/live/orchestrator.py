"""Phase 8 orchestrator — wires the live stream into the paper trader.

Single async task with three responsibilities:

  1. Wait for the next closed candle from :class:`CandleStream`.
  2. Hand it to :class:`InferenceService.append_live_bar` to recompute
     features / embedding and produce a prediction.
  3. Run the paper-trader state machine
     (:mod:`src.live.paper_trader`) over (candle, prediction).

The orchestrator is meant to be owned by FastAPI's lifespan: started
when the user clicks "Start", stopped when they click "Stop" or when
the app shuts down. It is **single-process, in-memory only** — restart
the server and you reset the books. That's exactly what a research
artefact should do; persisting paper P&L across runs would invite
treating it as a brokerage.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd
from loguru import logger

from src.api.inference import InferenceService
from src.live.paper_trader import (
    PaperTraderState,
    accuracy_stats,
    step as paper_step,
    validity_stats,
)
from src.live.stream import CandleStream, MulticastCandleStream


class LiveOrchestrator:
    """One-per-process coroutine driver for paper trading.

    Two supported modes:
      * **standalone** (used by the smoke): owns its own :class:`CandleStream`.
      * **multicast** (used by the dashboard): subscribes to a shared
        :class:`MulticastCandleStream` that also feeds the SSE endpoint.
    """

    def __init__(self, inference: InferenceService,
                 initial_balance: float = 100.0,
                 multicast: Optional[MulticastCandleStream] = None) -> None:
        self.inference = inference
        self.state = PaperTraderState(initial_balance=initial_balance,
                                      balance=initial_balance)
        self.stream: Optional[CandleStream] = None
        self.multicast = multicast
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self.is_running:
            logger.info("[orchestrator] already running")
            return
        if self.multicast is not None:
            self._queue = self.multicast.subscribe()
        else:
            self.stream = CandleStream(pair=self.inference.pair,
                                       interval=self.inference.interval)
            await self.stream.start()
            self._queue = self.stream.queue

        self.state.is_running = True
        self.state.started_at = pd.Timestamp.utcnow()
        self._task = asyncio.create_task(self._run())
        logger.success("[orchestrator] started")

    async def stop(self) -> None:
        self.state.is_running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self.multicast is not None and self._queue is not None:
            self.multicast.unsubscribe(self._queue)
        if self.stream is not None:
            await self.stream.stop()
            self.stream = None
        self._queue = None
        logger.info("[orchestrator] stopped")

    async def reset(self, initial_balance: float = 100.0) -> None:
        self.state.reset(initial_balance=initial_balance)
        logger.info(f"[orchestrator] state reset (balance ${initial_balance:.2f})")

    # ------------------------------------------------------------------
    async def _run(self) -> None:
        """Main loop: consume candles, predict, trade. Cancellable.

        Filters for ``is_closed=True`` so that the multicast (which
        forwards every tick) doesn't cause a mid-candle prediction.
        """
        assert self._queue is not None
        try:
            while True:
                candle = await self._queue.get()
                if not candle.get("is_closed", True):
                    continue
                try:
                    self._process_candle(candle)
                except Exception as e:                              # noqa: BLE001
                    logger.exception(f"[orchestrator] candle processing failed: {e}")
        except asyncio.CancelledError:
            raise

    # Synchronous body — called from the async loop. Keeps the trading
    # state-machine deterministic and easy to test from the smoke.
    def _process_candle(self, candle: dict) -> None:
        ts = pd.Timestamp(candle["open_time"]).tz_convert("UTC") \
            if pd.Timestamp(candle["open_time"]).tzinfo is not None \
            else pd.Timestamp(candle["open_time"], tz="UTC")
        try:
            prediction = self.inference.append_live_bar(candle)
        except Exception as e:                                      # noqa: BLE001
            logger.error(f"[orchestrator] inference failed at {ts}: {e}")
            return

        decision = paper_step(self.state, candle, prediction)
        d = decision["decision"]
        b = decision["balance"]
        ct = decision["closed_trade"]
        op = decision["opened_position"]
        msg = (
            f"[orchestrator] {ts}  close=${candle['close']:.2f}  "
            f"P(up)={prediction['p_up']:.3f}  → {d.upper()}  bal=${b:.2f}"
        )
        if ct is not None:
            msg += (
                f"  | closed {ct.direction} pnl=${ct.pnl_dollars:+.2f} "
                f"({ct.pnl_pct*100:+.2f}%)"
            )
        if op is not None:
            msg += f"  | opened {op.direction} ${op.dollars:.2f} "
            msg += f"@{op.entry_price:.2f}"
        logger.info(msg)

    # ------------------------------------------------------------------
    # Convenience for the HTTP layer
    # ------------------------------------------------------------------
    def snapshot(self, n_trades: int = 20) -> dict:
        s = self.state
        stats = validity_stats(s, window=50)
        acc = accuracy_stats(s)
        open_pos = None
        if s.open_position is not None:
            p = s.open_position
            open_pos = {
                "direction": p.direction,
                "entry_time": p.entry_time.isoformat(),
                "entry_price": p.entry_price,
                "dollars": p.dollars,
                "size_pct": p.size_pct,
                "p_up": p.p_up,
                "regime": p.regime,
            }
        recent = []
        for t in s.closed_trades[-n_trades:]:
            recent.append({
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "dollars": t.dollars,
                "pnl_dollars": t.pnl_dollars,
                "pnl_pct": t.pnl_pct,
                "hit": t.hit,
                "regime": t.regime,
            })
        # Cap equity-curve payload size — full history would balloon.
        eq = s.equity_curve[-500:]
        # Prefer the multicast connection status (shared with the SSE
        # feed) over the standalone-mode stream, so the dashboard's
        # "🟢 running" badge reflects the actual live-data source.
        if self.multicast is not None:
            stream_status = {
                "connected": self.multicast.is_connected,
                "messages_seen": self.multicast.messages_seen,
            }
        elif self.stream is not None:
            stream_status = {
                "connected": self.stream.is_connected,
                "messages_seen": self.stream.messages_seen,
            }
        else:
            stream_status = {"connected": False, "messages_seen": 0}

        return {
            "is_running": self.is_running,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "initial_balance": s.initial_balance,
            "balance": s.balance,
            "bars_seen": s.bars_seen,
            "bars_traded": s.bars_traded,
            "bars_abstained": s.bars_abstained,
            "open_position": open_pos,
            "recent_trades": recent,
            "last_signal": s.last_signal,
            "stats": stats,
            "accuracy": acc,
            "equity_curve": [
                {"open_time": t.isoformat(), "balance": float(b)}
                for t, b in eq
            ],
            "stream": stream_status,
        }
