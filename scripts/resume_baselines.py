"""Resume an interrupted baseline sweep.

Looks at every ``experiments/baselines/<pair>/<model>/metrics.json`` already
on disk and runs ONLY the missing ``(model, pair)`` combinations.

Default scope = all 7 baselines × 5 pairs at 1h. Override with --models /
--pairs / --interval as you would with train_baselines.py.

    python scripts/resume_baselines.py
    python scripts/resume_baselines.py --models cnn_candle cnn_gaf
    python scripts/resume_baselines.py --dry-run        # just show what's pending
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.runner import EXPERIMENTS_ROOT, MODEL_REGISTRY, run_one  # noqa: E402


DEFAULT_MODELS = [
    "naive_majority",
    "naive_persistence",
    "xgboost",
    "lstm",
    "patchtst",
    "cnn_candle",
    "cnn_gaf",
]


def already_done(pair: str, model: str) -> bool:
    """A combo counts as 'done' iff its metrics.json was fully written."""
    return (EXPERIMENTS_ROOT / pair / model / "metrics.json").is_file()


if __name__ == "__main__":
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Resume baseline sweep, skipping finished combos.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true",
                        help="List pending combos without running anything.")
    args = parser.parse_args()

    pending: list[tuple[str, str]] = []
    done: list[tuple[str, str]] = []
    for pair in args.pairs:
        for model in args.models:
            (done if already_done(pair, model) else pending).append((pair, model))

    logger.info(f"already done : {len(done):>3} combos")
    for pair, model in done:
        logger.info(f"  ✓  {model:<20} / {pair}")

    logger.info(f"pending      : {len(pending):>3} combos")
    for pair, model in pending:
        logger.info(f"  ·  {model:<20} / {pair}")

    if args.dry_run or not pending:
        sys.exit(0)

    logger.info("starting pending runs…")
    for pair, model in pending:
        try:
            run_one(model, pair, args.interval,
                    n_folds=args.folds, train_frac=args.train_frac)
        except Exception as e:  # noqa: BLE001 — keep going if one combo fails
            logger.exception(f"FAILED {model} / {pair}: {e}")
    logger.success("resume sweep complete.")
