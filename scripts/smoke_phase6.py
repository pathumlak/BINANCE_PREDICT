"""Phase 6 smoke — verifies the pattern engine wires up end-to-end.

What it checks:
  1. Regime labels file exists (or fail with the exact fix command).
  2. FAISS index loads and reports the expected dimensionality (128).
  3. Querying with a random embedding returns K matches.
  4. With ``regime_filter=q_regime`` set, every match shares the query's
     regime.
  5. The chronological self-exclusion logic drops the query row itself.

Should finish in well under 30 seconds.

Usage::

    python scripts/smoke_phase6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.retrieval.faiss_index import _embeddings_path  # noqa: E402
from src.retrieval.hmm_regimes import REGIME_LABELS, load_regimes  # noqa: E402
from src.retrieval.pattern_engine import PatternEngine  # noqa: E402


def main() -> int:
    pair, interval = "BTCUSDT", "1h"
    logger.info("Phase 6 smoke: pattern engine on BTCUSDT 1h")

    # Pre-flight: regime labels exist?
    try:
        regimes = load_regimes(pair, interval)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2

    # Pre-flight: FAISS index exists?
    try:
        engine = PatternEngine(pair, interval)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 2

    assert engine.handle.dim == 128, f"unexpected dim {engine.handle.dim}"
    logger.info(f"index loaded: {len(engine.handle.meta):,} rows, dim={engine.handle.dim}")
    logger.info(f"regime label distribution:")
    counts = engine.handle.meta["regime"].value_counts().sort_index()
    for rid, c in counts.items():
        name = REGIME_LABELS.get(int(rid), f"state_{int(rid)}")
        logger.info(f"  {int(rid)} ({name:<8}): {int(c):,}")

    # Pick 5 random query rows from the embeddings parquet.
    emb_path = _embeddings_path(pair, interval, "candle")
    emb = pd.read_parquet(emb_path)
    emb["open_time"] = pd.to_datetime(emb["open_time"], utc=True)
    emb = emb.set_index("open_time").sort_index()
    cols = [c for c in emb.columns if c.startswith("e")]

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(emb), size=5, replace=False)

    fail = False
    for idx in sample_idx:
        ts = emb.index[int(idx)]
        q_vec = emb[cols].iloc[int(idx)].to_numpy(dtype=np.float32)

        # What's this bar's regime?
        if ts not in regimes.index:
            continue
        q_regime = int(regimes.loc[ts, "regime"])

        # Top-K with regime filter.
        matches = engine.topk(
            q_vec, k=10, regime_filter=q_regime,
            exclude_self_within_seconds=3600,
            query_open_time=ts,
        )
        if len(matches) < 5:
            logger.error(f"too few matches for {ts}: {len(matches)}")
            fail = True
            continue
        wrong_regime = [m for m in matches if m.regime != q_regime]
        if wrong_regime:
            logger.error(
                f"regime filter leak: {len(wrong_regime)} of "
                f"{len(matches)} matches had wrong regime"
            )
            fail = True
        same_time = [m for m in matches if m.open_time == ts]
        if same_time:
            logger.error(f"self-match leak at {ts}")
            fail = True

        logger.info(
            f"query t={ts}  regime={q_regime} ({REGIME_LABELS[q_regime]}) "
            f"-> top-{len(matches)} matches OK"
        )
        # Print a couple of the matches so you can eyeball them.
        for m in matches[:3]:
            logger.info(
                f"    rank {m.rank:>2}: t={m.open_time}  "
                f"sim={m.similarity:.4f}  regime={m.regime_name}"
            )

    if fail:
        logger.error("smoke phase 6 FAILED")
        return 1
    logger.success("smoke phase 6 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
