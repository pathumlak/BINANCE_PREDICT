"""Build a FAISS cosine-similarity index over Phase 3 CNN embeddings.

Pre-conditions:
  * ``experiments/embeddings/<PAIR>/<INTERVAL>/<encoder>/embeddings.parquet``
    (Run ``scripts/extract_chart_embeddings.py`` first.)
  * ``data/regimes/<PAIR>/<INTERVAL>.parquet``
    (Run ``scripts/fit_hmm_regimes.py`` first.)

Output goes to:
  ``experiments/retrieval/<PAIR>/<INTERVAL>/{index.faiss, meta.parquet}``

Usage::

    python scripts/build_faiss_index.py
    python scripts/build_faiss_index.py --pairs BTCUSDT --interval 1h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.config import load_config  # noqa: E402
from src.retrieval.faiss_index import build_index  # noqa: E402


if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", default="candle", choices=["candle", "gaf"])
    args = p.parse_args()

    for pair in args.pairs:
        try:
            handle = build_index(pair, args.interval, encoder=args.encoder)
            logger.success(
                f"{pair}: indexed {len(handle.meta):,} embeddings "
                f"(dim={handle.dim})"
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"FAILED {pair}: {e}")
