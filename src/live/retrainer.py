"""Nightly refit of the fast models — self-sustaining pipeline.

Runs as an asyncio background task inside the dashboard's lifespan.
Every ``RETRAIN_INTERVAL_S`` seconds (default 24 h) it:

  1. Asks the LivePersistor to flush any buffered candles / embeddings.
  2. Calls ``InferenceService.refit_fast_models()`` — which shells out to
     ``scripts/fit_hmm_regimes.py`` and ``scripts/build_faiss_index.py``,
     then reloads OHLCV / features / embeddings / regimes / FAISS in
     memory and refits the anchor fusion model.

Fast in this context means "seconds to a minute" (HMM ~10 s, FAISS ~5 s,
fusion refit ~5 s). The Chart-CNN retraining takes 15-25 min and is
explicitly **not** part of the auto loop — a user should trigger it
by hand via ``python scripts/refresh_all.py`` when convenient.

The scheduler also exposes a ``trigger_now()`` coroutine so the
dashboard's "Refit now" button can force an immediate run.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from src.api.inference import InferenceService

# 24 hours between automatic refits. Keeping this configurable via
# the RetrainerScheduler constructor.
RETRAIN_INTERVAL_S = 24 * 3600


class RetrainerScheduler:
    """Periodic + on-demand refit driver."""

    def __init__(
        self,
        inference: InferenceService,
        orchestrator=None,
        interval_seconds: int = RETRAIN_INTERVAL_S,
    ) -> None:
        self.inference = inference
        self.orchestrator = orchestrator
        self.interval_seconds = interval_seconds

        self._task: Optional[asyncio.Task] = None
        self._trigger_evt = asyncio.Event()
        self._stop = False

        self.last_run_at: Optional[datetime] = None
        self.last_result: Optional[dict] = None
        self.is_running_refit = False
        self._next_run_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    @property
    def is_scheduled(self) -> bool:
        return self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self.is_scheduled:
            return
        self._stop = False
        self._task = asyncio.create_task(self._loop(), name="retrainer")
        logger.success(
            f"[retrainer] scheduled — first fire in {self.interval_seconds//3600}h"
        )

    async def stop(self) -> None:
        self._stop = True
        self._trigger_evt.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def trigger_now(self) -> dict:
        """Immediately run a refit (blocks the caller until done)."""
        # Directly invoke instead of just setting the event, so the
        # caller can await the result.
        return await self._run_refit_once()

    def request_trigger(self) -> None:
        """Non-blocking: nudge the scheduler awake between naps."""
        self._trigger_evt.set()

    # ------------------------------------------------------------------
    async def _loop(self) -> None:
        try:
            while not self._stop:
                self._next_run_at = datetime.now(timezone.utc) + \
                    asyncio.get_event_loop().time().__class__(0) * 0  # placeholder
                # A dumb but correct wait: sleep in short slices so
                # trigger_now can wake us mid-nap.
                try:
                    await asyncio.wait_for(
                        self._trigger_evt.wait(),
                        timeout=self.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                self._trigger_evt.clear()
                if self._stop:
                    break
                try:
                    await self._run_refit_once()
                except Exception as e:                                # noqa: BLE001
                    logger.exception(f"[retrainer] refit crashed: {e}")
        except asyncio.CancelledError:
            raise

    async def _run_refit_once(self) -> dict:
        """Do a single refit off the main event loop."""
        if self.is_running_refit:
            return {"ok": False, "errors": ["another refit is already running"]}

        self.is_running_refit = True
        logger.info("[retrainer] refit starting")
        try:
            # Persist any buffered live bars so the retrain sees them.
            if (self.orchestrator is not None
                    and self.orchestrator.persistor is not None):
                self.orchestrator.persistor.flush()
            # The refit calls subprocess and disk IO — push it off the
            # loop so the SSE feed and the paper trader don't stall.
            result = await asyncio.to_thread(self.inference.refit_fast_models)
        finally:
            self.is_running_refit = False
        self.last_run_at = datetime.now(timezone.utc)
        self.last_result = result
        logger.success(
            f"[retrainer] refit done in "
            f"{sum(result.get('timings_seconds', {}).values()):.1f}s — "
            f"{result.get('total_bars_now', 0):,} bars"
        )
        return result

    # ------------------------------------------------------------------
    def status(self) -> dict:
        now = datetime.now(timezone.utc)
        eta_s: Optional[float] = None
        if self.last_run_at is not None:
            elapsed = (now - self.last_run_at).total_seconds()
            eta_s = max(0.0, self.interval_seconds - elapsed)
        return {
            "scheduled": self.is_scheduled,
            "interval_seconds": self.interval_seconds,
            "is_running_refit": self.is_running_refit,
            "last_run_at": (self.last_run_at.isoformat()
                            if self.last_run_at is not None else None),
            "last_result": self.last_result,
            "seconds_to_next_auto_refit": eta_s,
        }
