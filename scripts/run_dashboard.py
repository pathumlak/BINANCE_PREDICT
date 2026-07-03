"""Launch the Phase 7 dashboard.

Usage::

    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --host 0.0.0.0 --port 8080 --reload

Open http://127.0.0.1:8000 in your browser.

The first request takes 5–15 s while the InferenceService loads OHLCV,
features, embeddings, regime labels, the FAISS index, the CNN
checkpoint, and fits the anchor fusion model. After that every request
is fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="dev-mode: restart on code change")
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    args = p.parse_args()

    # Export the constructor args to env so the factory can read them
    # under --reload, which spawns a fresh process per change.
    import os  # noqa: PLC0415
    os.environ["DASHBOARD_PAIR"] = args.pair
    os.environ["DASHBOARD_INTERVAL"] = args.interval

    import uvicorn  # noqa: PLC0415
    uvicorn.run(
        "src.api.app:_create_app_from_env",
        host=args.host, port=args.port,
        reload=args.reload, factory=True,
        log_level="info",
    )
