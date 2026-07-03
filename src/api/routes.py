"""HTTP routes for the Phase 7 dashboard backend."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from src.api.inference import InferenceService
from src.api.models import (
    CandleResponse,
    ConformalSet,
    HealthResponse,
    PredictionResponse,
    RegimeResponse,
    SimilarMatchResponse,
    SimilarResponse,
)


router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
def _svc(req: Request) -> InferenceService:
    """Pluck the singleton out of app.state — set by the lifespan hook."""
    svc = getattr(req.app.state, "inference", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="inference service not ready")
    return svc


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def _parse_ts(value: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"invalid timestamp {value!r}: {e}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health(req: Request) -> HealthResponse:
    svc = _svc(req)
    t0, t1 = svc.time_range
    return HealthResponse(
        pair=svc.pair, interval=svc.interval,
        total_bars=svc.total_bars, n_features=svc.n_features,
        time_range_start=_iso(t0), time_range_end=_iso(t1),
        anchor_cut_row=svc.anchor_cut,
    )


@router.get("/pairs", response_model=list[str])
def pairs(req: Request) -> list[str]:
    return _svc(req).list_pairs()


# ---------------------------------------------------------------------------
@router.get("/candles", response_model=list[CandleResponse])
def candles(
    req: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = Query(500, ge=1, le=10_000),
) -> list[CandleResponse]:
    svc = _svc(req)
    rows = svc.get_candles(
        start=_parse_ts(start) if start else None,
        end=_parse_ts(end) if end else None,
        limit=limit,
    )
    return [
        CandleResponse(
            open_time=_iso(r.open_time),
            open=r.open, high=r.high, low=r.low,
            close=r.close, volume=r.volume,
        )
        for r in rows
    ]


@router.get("/regimes", response_model=list[RegimeResponse])
def regimes(
    req: Request,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = Query(500, ge=1, le=10_000),
) -> list[RegimeResponse]:
    svc = _svc(req)
    rows = svc.get_regimes(
        start=_parse_ts(start) if start else None,
        end=_parse_ts(end) if end else None,
        limit=limit,
    )
    return [
        RegimeResponse(
            open_time=_iso(r["open_time"]),
            regime=r["regime"],
            regime_name=r["regime_name"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
@router.get("/predict", response_model=PredictionResponse)
def predict(req: Request, open_time: str) -> PredictionResponse:
    svc = _svc(req)
    ts = _parse_ts(open_time)
    try:
        out = svc.predict(ts)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return PredictionResponse(
        open_time=_iso(out["open_time"]),
        p_up=out["p_up"], label=out["label"],
        conformal_set=ConformalSet(
            down=out["conformal_set"]["down"],
            up=out["conformal_set"]["up"],
        ),
        regime=out["regime"], regime_name=out["regime_name"],
        is_in_demo_window=out["is_in_demo_window"],
    )


@router.get("/similar", response_model=SimilarResponse)
def similar(
    req: Request,
    open_time: str,
    k: int = Query(10, ge=1, le=100),
    regime_filter: bool = True,
) -> SimilarResponse:
    svc = _svc(req)
    ts = _parse_ts(open_time)
    try:
        rows = svc.similar(ts, k=k, regime_filter=regime_filter)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SimilarResponse(
        query_open_time=_iso(ts),
        k=k, regime_filter=regime_filter,
        matches=[
            SimilarMatchResponse(
                open_time=_iso(m.open_time),
                similarity=m.similarity,
                regime=m.regime,
                regime_name=m.regime_name,
            )
            for m in rows
        ],
    )


@router.get("/similar/detailed")
def similar_detailed(
    req: Request,
    open_time: str,
    k: int = Query(5, ge=1, le=20),
    regime_filter: bool = True,
) -> dict:
    """Similar matches enriched with their 64-bar OHLC windows.

    The frontend renders each match as a small candlestick sparkline so
    the viewer can see the actual historical pattern that the CNN
    considered similar.
    """
    svc = _svc(req)
    ts = _parse_ts(open_time)
    try:
        rows = svc.similar_with_windows(ts, k=k, regime_filter=regime_filter)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "query_open_time": _iso(ts),
        "k": k,
        "regime_filter": regime_filter,
        "matches": [
            {
                "open_time": _iso(m["open_time"]),
                "similarity": float(m["similarity"]),
                "regime": m["regime"],
                "regime_name": m["regime_name"],
                "window": m["window"],
                "next_close": m["next_close"],
                "next_direction": m["next_direction"],
            }
            for m in rows
        ],
    }


@router.get("/news")
def news(
    req: Request,
    limit: int = Query(25, ge=1, le=200),
    ticker: Optional[str] = None,
) -> dict:
    """Return the most-recent scored news articles.

    ``ticker`` (e.g. ``"BTC"``) narrows to articles mentioning that symbol.
    Sentiment score is in [-1, +1] from the CryptoBERT + FinBERT ensemble.
    """
    svc = _svc(req)
    items = svc.get_news(limit=limit, min_ticker=ticker)
    return {"count": len(items), "items": items}


@router.get("/gradcam")
def gradcam(req: Request, open_time: str) -> Response:
    svc = _svc(req)
    ts = _parse_ts(open_time)
    try:
        png = svc.gradcam_png(ts)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:                              # CNN not loaded
        raise HTTPException(status_code=503, detail=str(e))
    return Response(content=png, media_type="image/png")


# ---------------------------------------------------------------------------
# Phase 8 — live paper-trading endpoints
# ---------------------------------------------------------------------------
def _orch(req: Request):
    orch = getattr(req.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="orchestrator not ready")
    return orch


@router.get("/paper/state")
def paper_state(req: Request) -> dict:
    """Full snapshot of the paper-trader for the dashboard poll loop."""
    return _orch(req).snapshot()


@router.post("/paper/start")
async def paper_start(req: Request) -> dict:
    orch = _orch(req)
    await orch.start()
    return {"is_running": orch.is_running, "balance": orch.state.balance}


@router.post("/paper/stop")
async def paper_stop(req: Request) -> dict:
    orch = _orch(req)
    await orch.stop()
    return {"is_running": orch.is_running, "balance": orch.state.balance}


@router.post("/paper/reset")
async def paper_reset(req: Request, initial_balance: float = 100.0) -> dict:
    orch = _orch(req)
    await orch.reset(initial_balance=initial_balance)
    return {"is_running": orch.is_running, "balance": orch.state.balance}


@router.get("/paper/predictions")
def paper_predictions(req: Request, limit: int = Query(200, ge=1, le=2000)) -> dict:
    """Recent closed-bar predictions with their outcomes (for the
    dashboard's accuracy card + rolling-accuracy chart)."""
    orch = _orch(req)
    preds = orch.state.predictions[-limit:]
    rows = []
    for p in preds:
        rows.append({
            "open_time": p.open_time.isoformat(),
            "close_price": p.close_price,
            "p_up": p.p_up,
            "predicted_direction": p.predicted_direction,
            "cp_singleton": p.cp_singleton,
            "decision": p.decision,
            "regime": p.regime,
            "actual_direction": p.actual_direction,
            "correct": p.correct,
        })
    return {"count": len(rows), "predictions": rows}


# ---------------------------------------------------------------------------
# Live tick relay — Server-Sent Events. The browser subscribes to
# /api/live/stream and receives every kline update the backend gets
# from Binance, without having to reach ``stream.binance.com`` itself.
# ---------------------------------------------------------------------------
@router.get("/live/stream")
async def live_stream(req: Request):
    """Same-origin SSE endpoint that broadcasts Binance kline ticks."""
    mc = getattr(req.app.state, "multicast", None)
    if mc is None:
        raise HTTPException(status_code=503, detail="multicast not ready")

    q = mc.subscribe()

    async def event_gen():
        # Immediately tell the client we're alive so the LIVE badge can
        # go green before the first candle arrives.
        yield f"event: hello\ndata: {json.dumps({'connected': mc.is_connected})}\n\n"
        try:
            while True:
                if await req.is_disconnected():
                    break
                try:
                    candle = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Keepalive comment stops upstream proxies / browsers
                    # from timing out the connection during quiet minutes.
                    yield ": keepalive\n\n"
                    continue

                payload = {
                    "open_time": pd.Timestamp(candle["open_time"]).isoformat(),
                    "open":  float(candle["open"]),
                    "high":  float(candle["high"]),
                    "low":   float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"]),
                    "is_closed": bool(candle.get("is_closed", False)),
                }
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            mc.unsubscribe(q)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # nginx-friendly, harmless elsewhere
        },
    )
