"""Regime-aware pattern-matching facade.

Wraps :mod:`src.retrieval.faiss_index` with an optional regime filter so
the dashboard (Phase 7) can ask:

    "Show me the K historical bars most similar to *this* embedding,
    conditional on the market currently being in regime R."

Why filter at all?
------------------
A bull-market chart that visually resembles a bear-market chart is a
trap: the local geometry may match but the macroscopic context doesn't.
Filtering the index by HMM regime restores that context.

Two retrieval modes:
  1. **over_fetch_rerank** (default) — query the full index for
     ``k * 4`` candidates, then drop any whose regime ≠ filter, keep the
     top-K survivors. Cheap, robust to tiny regime imbalances.
  2. **per_regime_index** — pre-build one sub-index per regime and query
     only the matching one. Faster at query time, more memory, only worth
     it for very large indexes (millions of rows).

For Phase 6 we ship mode 1 — at ~38K BTC bars the over-fetch cost is
nanoseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.retrieval.faiss_index import IndexedEmbeddings, load_index, query
from src.retrieval.hmm_regimes import REGIME_LABELS


@dataclass
class Match:
    rank: int
    open_time: pd.Timestamp
    similarity: float            # cosine ∈ [-1, 1]
    regime: int
    regime_name: str


class PatternEngine:
    """Stateful holder for a FAISS index + meta. Cheap to construct."""

    def __init__(self, pair: str, interval: str = "1h") -> None:
        self.pair = pair
        self.interval = interval
        self.handle: IndexedEmbeddings = load_index(pair, interval)

    # ------------------------------------------------------------------
    def topk(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        regime_filter: Optional[int] = None,
        exclude_self_within_seconds: int = 0,
        query_open_time: Optional[pd.Timestamp] = None,
        over_fetch_mult: int = 4,
    ) -> list[Match]:
        """Return the top-K most-similar historical bars.

        Parameters
        ----------
        query_vec
            128-dim chart-CNN embedding.
        k
            Number of matches to return.
        regime_filter
            If not None, restrict matches to bars with that regime id.
        exclude_self_within_seconds
            When > 0 (and ``query_open_time`` is given), drop matches
            within that many seconds of the query. Prevents the query
            from matching itself when it's in the index.
        query_open_time
            The bar the query represents (for self-exclusion).
        over_fetch_mult
            Initial multiplier on ``k`` for the FAISS fetch size. If
            regime filtering / self-exclusion leave us short of ``k``
            matches the fetch is automatically doubled (up to the size
            of the whole index), so a rare regime never returns fewer
            matches than requested.
        """
        meta = self.handle.meta
        index_size = self.handle.index.ntotal
        max_fetch = min(index_size, max(k * 64, 1024))

        n_fetch = min(max(k * over_fetch_mult, k + 8), index_size)
        last_n_fetch = -1
        results: list[Match] = []

        # Grow the candidate pool until we have ``k`` survivors after
        # filtering, or we've already fetched the whole index.
        while True:
            sims, ids = query(self.handle, query_vec, k=n_fetch)
            sims, ids = sims[0], ids[0]

            results = []
            for sim, row_id in zip(sims, ids):
                if row_id < 0:
                    continue
                row = meta.iloc[int(row_id)]
                t = pd.Timestamp(row["open_time"])

                if exclude_self_within_seconds > 0 and query_open_time is not None:
                    dt = abs((t - pd.Timestamp(query_open_time)).total_seconds())
                    if dt < exclude_self_within_seconds:
                        continue

                regime = int(row["regime"])
                if regime_filter is not None and regime != regime_filter:
                    continue

                results.append(Match(
                    rank=len(results) + 1,
                    open_time=t,
                    similarity=float(sim),
                    regime=regime,
                    regime_name=REGIME_LABELS.get(regime, f"state_{regime}"),
                ))
                if len(results) == k:
                    break

            if len(results) >= k or n_fetch >= max_fetch or n_fetch == last_n_fetch:
                break
            last_n_fetch = n_fetch
            n_fetch = min(n_fetch * 2, max_fetch)

        return results
