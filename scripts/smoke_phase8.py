"""Phase 8 smoke — exercises the live paper-trading pipeline without
touching the real Binance WebSocket.

What it checks:
  1. The dashboard's lifespan wires up both the InferenceService and
     the LiveOrchestrator (no live stream is started in the smoke).
  2. ``InferenceService.append_live_bar`` succeeds on a *synthetic*
     closed candle that mimics what the Binance WS event would feed us:
       * features get rebuilt for the new bar
       * the CNN embeds the chart window
       * the fusion model produces a probability + conformal set
  3. ``paper_trader.step`` runs end-to-end: settles any prior position,
     applies the confidence gate, sizes by inverse-vol, opens or
     abstains, updates the balance.
  4. Running through ~30 synthetic candles produces a coherent state
     snapshot (recent trades, equity curve, validity verdict).

Usage::

    python scripts/smoke_phase8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402


def _synth_candle(prev_close: float, t: pd.Timestamp, rng: np.random.Generator) -> dict:
    """Generate a moderately-volatile fake closed candle near ``prev_close``.

    Includes ``taker_buy_base`` so the engineered-features pipeline
    doesn't drop the new row via dropna() on the taker-buy-share feature.
    """
    ret = rng.normal(loc=0.0, scale=0.004)         # ~0.4 % hourly vol
    close = prev_close * float(np.exp(ret))
    high = max(prev_close, close) * (1 + abs(rng.normal(0, 0.001)))
    low = min(prev_close, close) * (1 - abs(rng.normal(0, 0.001)))
    volume = float(rng.uniform(50, 200))
    quote_vol = volume * close
    return {
        "open_time": t.isoformat(),
        "open": float(prev_close),
        "high": float(high),
        "low":  float(low),
        "close": float(close),
        "volume": volume,
        "close_time": (t + pd.Timedelta(minutes=59, seconds=59)).isoformat(),
        "quote_volume": float(quote_vol),
        "trades": int(rng.integers(100, 2000)),
        "taker_buy_base": float(volume * rng.uniform(0.35, 0.65)),
        "taker_buy_quote": float(quote_vol * rng.uniform(0.35, 0.65)),
    }


def main() -> int:
    logger.info("Phase 8 smoke: booting dashboard app in-process...")
    app = create_app(pair="BTCUSDT", interval="1h")

    with TestClient(app) as client:
        orch = app.state.orchestrator
        svc = app.state.inference
        assert orch is not None, "orchestrator missing from app.state"

        # Use the timestamp AFTER the latest bar we have on disk so we
        # don't collide with historical rows.
        latest_ts = svc.time_range[1]
        last_close = float(svc._ohlcv.loc[latest_ts, "close"])
        logger.info(f"latest historical bar: {latest_ts}  close=${last_close:,.2f}")

        rng = np.random.default_rng(42)
        n_synth = 30
        for i in range(n_synth):
            t = latest_ts + pd.Timedelta(hours=i + 1)
            candle = _synth_candle(last_close, t, rng)
            try:
                # Drive the orchestrator's sync path directly — skipping the
                # actual WebSocket but using the EXACT code path that runs
                # in production.
                orch._process_candle(candle)
            except Exception as e:                            # noqa: BLE001
                logger.exception(f"candle {i} failed: {e}")
                return 1
            last_close = candle["close"]

        # Pull the snapshot via HTTP exactly like the dashboard does.
        r = client.get("/api/paper/state")
        assert r.status_code == 200, r.text
        snap = r.json()

        logger.info(
            f"after {snap['bars_seen']} bars: "
            f"balance=${snap['balance']:.2f}  "
            f"trades={snap['bars_traded']}  abstained={snap['bars_abstained']}"
        )
        stats = snap.get("stats", {})
        logger.info(
            f"stats: n_closed={stats.get('n_closed')}  "
            f"win_rate={stats.get('win_rate')}  "
            f"sharpe/trade={stats.get('sharpe_per_trade')}  "
            f"pnl=${stats.get('pnl_total_dollars'):.2f}  "
            f"verdict='{stats.get('verdict')}'"
        )

        # Sanity assertions.
        assert snap["bars_seen"] == n_synth, f"expected {n_synth} bars, got {snap['bars_seen']}"
        # Each bar must produce a last_signal entry.
        assert snap["last_signal"] is not None, "last_signal never populated"
        # Either we traded *or* we abstained — the conformal gate is binary.
        assert snap["bars_traded"] + snap["bars_abstained"] == n_synth, (
            f"trades + abstentions ({snap['bars_traded']} + "
            f"{snap['bars_abstained']}) ≠ bars_seen ({n_synth})"
        )

        # Don't start the live stream in the smoke (no real WS). But the
        # /start endpoint should still be callable; the orchestrator
        # opens the stream lazily on real candles.
        r = client.post("/api/paper/start")
        assert r.status_code == 200, r.text
        r = client.post("/api/paper/stop")
        assert r.status_code == 200, r.text

        r = client.post("/api/paper/reset", params={"initial_balance": 100})
        assert r.status_code == 200
        r = client.get("/api/paper/state")
        post_reset = r.json()
        assert post_reset["balance"] == 100.0, f"reset didn't restore balance"
        assert post_reset["bars_seen"] == 0, f"reset didn't clear bar counter"

    logger.success("smoke phase 8 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
