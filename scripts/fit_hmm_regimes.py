"""Fit 3-state Gaussian-HMM market regimes and persist per-bar labels.

The HMM is fit on the chronological training window (default first 50 %
of the data) using two features per bar: log-return + 24h rolling
realised volatility. The resulting states are then renumbered by
ascending mean return so the integer labels carry a stable meaning:

    0 = bear      (most-negative mean return)
    1 = sideways  (middle)
    2 = bull      (most-positive)

Output goes to ``data/regimes/<PAIR>/<INTERVAL>.parquet`` and is read by
``src.retrieval.faiss_index.build_index`` plus the Phase 7 dashboard.

Usage::

    python scripts/fit_hmm_regimes.py
    python scripts/fit_hmm_regimes.py --pairs BTCUSDT --interval 1h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.config import load_config  # noqa: E402
from src.retrieval.hmm_regimes import HMMConfig, fit_regimes, regime_summary  # noqa: E402


if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    p.add_argument("--interval", default="1h")
    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--vol-window", type=int, default=24)
    p.add_argument("--train-frac", type=float, default=0.5)
    args = p.parse_args()

    for pair in args.pairs:
        try:
            logger.info(f"--- HMM regimes for {pair} {args.interval} ---")
            out = fit_regimes(
                pair=pair, interval=args.interval,
                cfg=HMMConfig(
                    n_states=args.n_states,
                    vol_window=args.vol_window,
                    train_frac=args.train_frac,
                ),
            )
            summary = regime_summary(out)
            logger.success(f"wrote regimes for {pair} ({len(out):,} bars)")
            print(summary.to_string(index=False))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"FAILED {pair}: {e}")
