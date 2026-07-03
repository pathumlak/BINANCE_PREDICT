"""FAISS cosine k-NN index over Phase 3 chart-CNN embeddings.

Design choices
--------------
* **Cosine similarity** (= inner product on L2-normalised vectors).
  Embedding magnitudes are not stable across CNN retrainings, but the
  *direction* of the embedding captures the pattern. Normalising removes
  the magnitude axis from the comparison.
* **IndexFlatIP** — exact search. With ~38K BTC bars on the 1h grid,
  exact k-NN is already sub-millisecond per query, so the complexity of
  IVF / HNSW isn't worth the build cost or the recall hit.
* **Persistent meta** — alongside the binary `.faiss` file we write a
  parallel parquet mapping row-id → (open_time, regime). The dashboard
  in Phase 7 will need the open_time to render the matched candles.

API
---
``build_index(pair, interval)`` reads embeddings + regime labels, writes
the index + meta. ``load_index(pair, interval)`` returns ``(index, meta)``
for the pattern engine. ``query(index, query_vec, k)`` does the actual
k-NN.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INDEX_ROOT = PROJECT_ROOT / "experiments" / "retrieval"


@dataclass
class IndexedEmbeddings:
    index: object              # faiss.Index — typed as object to avoid hard dep on import time
    meta: pd.DataFrame         # columns: open_time, regime
    dim: int


def _embeddings_path(pair: str, interval: str, encoder: str = "candle") -> Path:
    return (PROJECT_ROOT / "experiments" / "embeddings"
            / pair / interval / encoder / "embeddings.parquet")


def _index_dir(pair: str, interval: str) -> Path:
    return INDEX_ROOT / pair / interval


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalise. Zero rows stay zero (safe for FAISS IP)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return x / norms


def build_index(
    pair: str,
    interval: str = "1h",
    encoder: str = "candle",
) -> IndexedEmbeddings:
    """Read embeddings + regime labels, build a FAISS index, persist it.

    Pre-conditions
    --------------
    * Phase 3 embeddings exist at the canonical path.
    * Phase 6 HMM regime labels exist (Run ``scripts/fit_hmm_regimes.py``).
    """
    import faiss  # noqa: PLC0415
    from src.retrieval.hmm_regimes import load_regimes  # noqa: PLC0415

    emb_path = _embeddings_path(pair, interval, encoder)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"embeddings missing: {emb_path}. Run:\n"
            f"  python scripts/extract_chart_embeddings.py "
            f"--pairs {pair} --interval {interval} --encoder {encoder}"
        )
    emb = pd.read_parquet(emb_path)
    emb["open_time"] = pd.to_datetime(emb["open_time"], utc=True)
    emb = emb.set_index("open_time").sort_index()

    cols = [c for c in emb.columns if c.startswith("e")]
    vecs = emb[cols].to_numpy(dtype=np.float32)
    dim = vecs.shape[1]

    # Inner-join with regime labels — only bars that have both an
    # embedding and a regime get indexed.
    regimes = load_regimes(pair, interval)
    joined = emb[[]].join(regimes, how="inner")
    if joined.empty:
        raise RuntimeError(
            f"no overlap between embeddings and regimes for {pair} {interval}"
        )
    vecs = vecs[emb.index.isin(joined.index)]
    vecs = _l2_normalise(vecs)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    meta = joined.reset_index()[["open_time", "regime"]]

    # Persist
    out_dir = _index_dir(pair, interval)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    meta.to_parquet(out_dir / "meta.parquet", compression="snappy", index=False)

    return IndexedEmbeddings(index=index, meta=meta, dim=dim)


def load_index(pair: str, interval: str = "1h") -> IndexedEmbeddings:
    """Load a previously persisted FAISS index + its meta parquet."""
    import faiss  # noqa: PLC0415

    out_dir = _index_dir(pair, interval)
    idx_path = out_dir / "index.faiss"
    meta_path = out_dir / "meta.parquet"
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"FAISS index missing under {out_dir}. Run:\n"
            f"  python scripts/build_faiss_index.py "
            f"--pairs {pair} --interval {interval}"
        )
    index = faiss.read_index(str(idx_path))
    meta = pd.read_parquet(meta_path)
    meta["open_time"] = pd.to_datetime(meta["open_time"], utc=True)
    return IndexedEmbeddings(index=index, meta=meta, dim=index.d)


def query(
    handle: IndexedEmbeddings,
    query_vecs: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(distances, row_ids)`` of shape (n_query, k).

    Distances are cosine *similarities* in [-1, 1] (because we use
    IndexFlatIP on L2-normalised vectors). row_ids index into
    ``handle.meta``.
    """
    qv = np.asarray(query_vecs, dtype=np.float32)
    if qv.ndim == 1:
        qv = qv[None, :]
    qv = _l2_normalise(qv)
    sims, ids = handle.index.search(qv, k)
    return sims, ids
