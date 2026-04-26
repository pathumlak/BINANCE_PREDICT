"""Phase 2 smoke test — runs the pipeline on a single (model, pair) combo
with very few folds so you can verify everything wires up before launching
the full multi-hour run.

Default: XGBoost on BTCUSDT 1h with 3 folds. Should finish in ~30s.

    python scripts/smoke_phase2.py
    python scripts/smoke_phase2.py --model lstm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.runner import run_one  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--pair", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    summary = run_one(
        model_name=args.model,
        pair=args.pair,
        interval=args.interval,
        n_folds=args.folds,
        train_frac=0.6,
    )
    logger.success("smoke OK")
    for k, v in summary.items():
        print(f"  {k}: {v}")
