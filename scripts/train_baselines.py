"""Entry point: train every Phase 2 baseline across all configured pairs.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --models xgboost lstm
    python scripts/train_baselines.py --pairs BTCUSDT --interval 1h --folds 8

Tip: do `--models naive_majority naive_persistence xgboost` first to
verify the harness, then add lstm / patchtst once you've checked the
classification metrics look sane.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.runner import MODEL_REGISTRY, run_all  # noqa: E402


DEFAULT_MODELS = [
    "naive_majority",
    "naive_persistence",
    "xgboost",
    "lstm",
    "patchtst",
    "cnn_candle",
    "cnn_gaf",
]


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Train Phase 2 baselines.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--train-frac", type=float, default=0.5)
    args = parser.parse_args()

    df = run_all(
        models=args.models,
        pairs=args.pairs,
        interval=args.interval,
        n_folds=args.folds,
        train_frac=args.train_frac,
    )
    if not df.empty:
        cols = ["pair", "model", "accuracy", "f1", "mcc",
                "sharpe_per_bar", "pnl_log", "elapsed_seconds"]
        print("\n" + df[cols].to_string(index=False))
