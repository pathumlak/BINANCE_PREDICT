"""Phase 7 smoke — exercises every dashboard endpoint via TestClient.

Boots the FastAPI app in-process (no live uvicorn needed) so the
InferenceService is constructed exactly as it would be in production,
then hits each endpoint and asserts a basic shape on the response.

Should finish in 5–20 seconds (the slow bit is the anchor-fusion fit
that the lifespan event does — same speed as the dashboard's cold
start).

Usage::

    python scripts/smoke_phase7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402


def main() -> int:
    logger.info("Phase 7 smoke: spinning up FastAPI app in-process...")
    app = create_app(pair="BTCUSDT", interval="1h")

    with TestClient(app) as client:
        # 1. Health.
        r = client.get("/api/health")
        assert r.status_code == 200, r.text
        h = r.json()
        logger.info(
            f"health: pair={h['pair']}  total_bars={h['total_bars']:,}  "
            f"feats={h['n_features']}  anchor_cut={h['anchor_cut_row']:,}"
        )

        # 2. Pairs.
        r = client.get("/api/pairs")
        assert r.status_code == 200 and r.json() == [h["pair"]]

        # 3. Candles.
        r = client.get("/api/candles", params={"limit": 50})
        assert r.status_code == 200
        candles = r.json()
        assert len(candles) == 50, f"got {len(candles)} candles"
        assert {"open", "high", "low", "close", "volume", "open_time"} \
            <= set(candles[0].keys())
        logger.info(f"candles: {len(candles)} rows, last close = {candles[-1]['close']}")

        # 4. Regimes.
        r = client.get("/api/regimes", params={"limit": 50})
        assert r.status_code == 200
        regs = r.json()
        names = {x["regime_name"] for x in regs}
        logger.info(f"regimes: distinct names in window = {sorted(names)}")

        # 5. Predict. The VERY last bar has no next-bar label (it's the
        #    most recent OHLCV row) so engineered features drop it; the
        #    /api/predict endpoint correctly 404s on it. Use the
        #    second-to-last bar, which is guaranteed labelled.
        assert r.status_code == 200       # (regimes call from step 4)
        query_ts = candles[-2]["open_time"]
        # Confirm the API rejects the unlabelled tail bar — guards a
        # regression where someone "fixes" this by silently predicting on it.
        r = client.get("/api/predict", params={"open_time": candles[-1]["open_time"]})
        assert r.status_code == 404, (
            f"expected 404 on the unlabelled tail bar, got {r.status_code}: "
            f"{r.text}"
        )

        r = client.get("/api/predict", params={"open_time": query_ts})
        assert r.status_code == 200, r.text
        p = r.json()
        logger.info(
            f"predict @ {query_ts}: P(up)={p['p_up']:.4f}  label={p['label']}  "
            f"cp_set=(down={p['conformal_set']['down']}, "
            f"up={p['conformal_set']['up']})  regime={p['regime_name']}  "
            f"held_out={p['is_in_demo_window']}"
        )

        # 6. Similar.
        r = client.get("/api/similar", params={
            "open_time": query_ts, "k": 5, "regime_filter": True,
        })
        assert r.status_code == 200, r.text
        s = r.json()
        assert s["k"] == 5
        assert len(s["matches"]) == 5, f"expected 5 matches, got {len(s['matches'])}"
        logger.info(f"similar: top match = {s['matches'][0]['open_time']} "
                    f"sim={s['matches'][0]['similarity']:.4f}")

        # 7. Grad-CAM — optional (skipped with a warning if CNN missing).
        r = client.get("/api/gradcam", params={"open_time": query_ts})
        if r.status_code == 200:
            logger.info(f"grad-cam: {len(r.content):,} bytes PNG")
        elif r.status_code == 503:
            logger.warning(
                "grad-cam disabled — no CNN checkpoint. "
                "Run scripts/train_cnn_ewc.py to enable."
            )
        else:
            logger.error(f"grad-cam unexpected: {r.status_code} {r.text}")
            return 1

    logger.success("smoke phase 7 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
