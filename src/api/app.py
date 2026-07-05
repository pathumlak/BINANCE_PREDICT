"""FastAPI app factory for the Phase 7 dashboard.

Spinup is intentionally slow: the lifespan event loads OHLCV, features,
embeddings, regimes, FAISS index, and fits the anchor fusion model
before the first HTTP request is accepted. On a typical CPU it takes
5–15 s; after that every request is fast.

Run it with::

    python scripts/run_dashboard.py

or::

    uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.inference import InferenceService
from src.api.routes import router
from src.live.orchestrator import LiveOrchestrator
from src.live.retrainer import RetrainerScheduler
from src.live.stream import MulticastCandleStream

log = logging.getLogger("phase7.app")

# Folder layout for the bundled single-page UI.
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(pair: str = "BTCUSDT", interval: str = "1h") -> FastAPI:
    """FastAPI factory. ``--factory`` mode lets uvicorn call us per worker."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("[Phase 7] starting up — building InferenceService...")
        app.state.inference = InferenceService(pair=pair, interval=interval)
        # Phase 8 — one always-on Binance WebSocket, fanned out to
        # (a) the paper-trading orchestrator (closed candles only) and
        # (b) any open Server-Sent-Events client (every tick), so the
        # browser never has to reach Binance directly.
        app.state.multicast = MulticastCandleStream(
            pair=pair, interval=interval,
        )
        try:
            await app.state.multicast.start()
        except Exception:                                # noqa: BLE001
            log.exception(
                "multicast start failed — SSE feed will be unavailable "
                "but the rest of the dashboard still works"
            )
        app.state.orchestrator = LiveOrchestrator(
            inference=app.state.inference, initial_balance=100.0,
            multicast=app.state.multicast,
        )
        # Nightly refit of HMM + FAISS + fusion so the model stays fresh
        # on newly-arrived bars without any manual scripts/refresh_all.py.
        app.state.retrainer = RetrainerScheduler(
            inference=app.state.inference,
            orchestrator=app.state.orchestrator,
        )
        try:
            await app.state.retrainer.start()
        except Exception:                                # noqa: BLE001
            log.exception("retrainer start failed — dashboard still runs")

        log.info("[Phase 7] ready")
        yield
        log.info("[Phase 7] shutting down")
        try:
            await app.state.retrainer.stop()
        except Exception:                                # noqa: BLE001
            log.exception("retrainer shutdown failed")
        try:
            await app.state.orchestrator.stop()
        except Exception:                                # noqa: BLE001
            log.exception("orchestrator shutdown failed")
        try:
            if app.state.orchestrator.persistor is not None:
                app.state.orchestrator.persistor.flush()
        except Exception:                                # noqa: BLE001
            log.exception("final persistence flush failed")
        try:
            await app.state.multicast.stop()
        except Exception:                                # noqa: BLE001
            log.exception("multicast shutdown failed")

    app = FastAPI(
        title="Multimodal Crypto Direction Predictor — Dashboard",
        version="0.7.0",
        description=(
            "Phase 7 of the thesis. Exposes the Phase 5 fusion model + "
            "Phase 6 pattern engine over HTTP, plus a vanilla-JS frontend "
            "for live exploration."
        ),
        lifespan=lifespan,
    )

    app.include_router(router)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True),
                  name="static")
    else:                                                 # pragma: no cover
        log.warning(f"[Phase 7] static dir missing at {STATIC_DIR}; "
                    f"API-only mode")

    return app


def _create_app_from_env() -> FastAPI:
    """uvicorn --factory entry point reading config from env vars.

    ``scripts/run_dashboard.py`` sets ``DASHBOARD_PAIR`` and
    ``DASHBOARD_INTERVAL`` before invoking uvicorn, so each spawned
    worker (especially under ``--reload``) gets identical config.
    """
    return create_app(
        pair=os.environ.get("DASHBOARD_PAIR", "BTCUSDT"),
        interval=os.environ.get("DASHBOARD_INTERVAL", "1h"),
    )
