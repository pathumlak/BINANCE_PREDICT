"""Phase 5 smoke test — verifies the fusion pipeline runs end-to-end.

Runs ONE walk-forward fold of the full-fusion variant on BTCUSDT 1h.
Should take well under a minute on CPU (logistic regression is cheap).

Pre-condition (must already be on disk):
  * experiments/embeddings/BTCUSDT/1h/candle/embeddings.parquet
  * data/features_sentiment/BTCUSDT/1h.parquet

If either is missing the script prints the exact command to produce it
and exits with status 2.

What it checks:
  1. The fusion baseline loads its three modality blocks.
  2. The shapes line up after `reindex` to OHLCV `open_time`.
  3. MAPIE-style inductive-conformal calibration yields a usable threshold.
  4. Empirical coverage on the test fold is in the right ballpark (~ 1-alpha).
  5. Average prediction-set size is between 1 and 2.

Usage:
    python scripts/smoke_phase5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from scripts.train_fusion import _preflight, run_variant  # noqa: E402


def main() -> int:
    pair, interval = "BTCUSDT", "1h"
    logger.info("Phase 5 smoke: 1-fold fusion run on BTCUSDT 1h")

    _preflight(pair, interval, need_cnn=True, need_sent=True)

    # n_folds=3 with train_frac=0.6 mirrors smoke_phase2.py and leaves
    # enough headroom that the chronological gap between train and test
    # never pushes the final fold past the dataset end.
    agg = run_variant(
        variant="fusion_num_cnn_sent",
        pair=pair, interval=interval,
        n_folds=3, train_frac=0.6,
    )

    acc = agg.get("accuracy", float("nan"))
    cov = agg.get("cp_coverage", float("nan"))
    sset = agg.get("cp_avg_set_size", float("nan"))

    fail = False
    if not (0.3 <= acc <= 0.7):
        logger.error(f"accuracy {acc:.3f} out of sane range")
        fail = True
    if not (0.80 <= cov <= 0.99):
        # Nominal is 0.90 — wide tolerance for one fold.
        logger.error(f"conformal coverage {cov:.3f} far from 0.90 nominal")
        fail = True
    if not (1.0 <= sset <= 2.0):
        logger.error(f"avg set size {sset:.3f} out of [1, 2]")
        fail = True

    if fail:
        logger.error("smoke phase 5 FAILED")
        return 1

    logger.success(
        f"smoke phase 5 OK  acc={acc:.3f}  cp_cov={cov:.3f}  set_size={sset:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
