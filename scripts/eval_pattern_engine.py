"""Evaluate the regime-aware pattern engine on held-out bars.

The pattern engine is unsupervised at the retrieval level — k-NN lookup
doesn't produce a probability. So how do we know if the matches are
"good"?

We use **next-bar direction agreement** as a proxy quality metric.
Specifically, for each held-out test bar:

  1. Fetch the top-K most similar historical bars from the FAISS index
     (with regime filter ON or OFF).
  2. Compute the *direction* (up / down) of each retrieved bar's
     **next** candle.
  3. Take the majority vote.
  4. Score it against the held-out bar's actual next-candle direction.

If the index encodes meaningful chart patterns, the majority vote should
have meaningfully > 0.50 hit rate. Comparing **regime-filtered** to
**no-filter** quantifies whether the HMM is adding value.

Outputs land at::

    experiments/retrieval/<PAIR>/eval_pattern_engine.json
    experiments/retrieval/<PAIR>/eval_pattern_engine.csv

Usage::

    python scripts/eval_pattern_engine.py --pair BTCUSDT --k 10 \
        --test-frac 0.2 --max-test-bars 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.features.dataset import load_ohlcv  # noqa: E402
from src.retrieval.pattern_engine import PatternEngine  # noqa: E402
from src.retrieval.faiss_index import _embeddings_path  # noqa: E402


def _load_next_direction(pair: str, interval: str) -> pd.Series:
    """Per-bar binary label: 1 if next close > current close, else 0.

    Aligned to ``open_time`` (the bar whose NEXT bar we're labelling).
    The very last bar gets dropped.
    """
    raw = load_ohlcv(pair, interval)
    raw = raw.sort_values("open_time").drop_duplicates("open_time")
    raw["open_time"] = pd.to_datetime(raw["open_time"], utc=True)
    raw = raw.set_index("open_time")
    nxt = raw["close"].shift(-1)
    y = (nxt > raw["close"]).astype(int)
    return y[:-1]


def _load_embeddings_with_index(pair: str, interval: str, encoder: str) -> pd.DataFrame:
    path = _embeddings_path(pair, interval, encoder)
    if not path.exists():
        raise FileNotFoundError(
            f"embeddings missing at {path}.  Build them with "
            f"scripts/extract_chart_embeddings.py first."
        )
    emb = pd.read_parquet(path)
    emb["open_time"] = pd.to_datetime(emb["open_time"], utc=True)
    return emb.set_index("open_time").sort_index()


# ---------------------------------------------------------------------------
def evaluate(
    *, pair: str, interval: str, encoder: str,
    k: int, test_frac: float, max_test_bars: Optional[int],
    rng_seed: int = 0,
) -> dict:
    logger.info(f"=== eval_pattern_engine  {pair} {interval}  k={k} ===")
    t0 = time.time()

    engine = PatternEngine(pair, interval)
    emb = _load_embeddings_with_index(pair, interval, encoder)
    y_next = _load_next_direction(pair, interval)
    meta = engine.handle.meta.set_index("open_time")

    # The "history" available for retrieval at the start of the test
    # window is everything that comes before. We avoid look-ahead by
    # restricting retrievals to bars chronologically before the query.
    # The current PatternEngine queries the full index — so for this
    # eval we *filter results* after the fact rather than rebuilding a
    # per-bar index. Approximate but fast.
    all_times = meta.index.to_numpy()

    # Hold out the last `test_frac` of overlapping bars as the test set.
    overlap = sorted(set(emb.index) & set(meta.index) & set(y_next.index))
    overlap_arr = np.array(overlap)
    n_test = int(len(overlap_arr) * test_frac)
    test_times = overlap_arr[-n_test:]

    if max_test_bars is not None and len(test_times) > max_test_bars:
        rng = np.random.default_rng(rng_seed)
        test_times = np.sort(rng.choice(test_times, size=max_test_bars, replace=False))

    cols = [c for c in emb.columns if c.startswith("e")]
    embeddings_lookup = emb[cols]

    results: list[dict] = []
    for t in test_times:
        ts = pd.Timestamp(t)
        q_vec = embeddings_lookup.loc[ts].to_numpy(dtype=np.float32)
        q_regime = int(meta.loc[ts, "regime"])
        true_dir = int(y_next.loc[ts])

        # Two retrieval modes: with and without regime filter.
        for filter_on in (False, True):
            matches = engine.topk(
                q_vec, k=k * 6,
                regime_filter=q_regime if filter_on else None,
                exclude_self_within_seconds=0,
                query_open_time=ts,
            )
            # Keep only matches strictly BEFORE the query (no leak).
            matches = [m for m in matches if m.open_time < ts][:k]
            if len(matches) < max(3, k // 2):
                # Insufficient history for this query at the requested K
                results.append({
                    "open_time": ts, "regime_filter": filter_on,
                    "true_dir": true_dir, "n_matches": len(matches),
                    "majority": np.nan, "hit": np.nan,
                })
                continue
            next_dirs = [int(y_next.loc[m.open_time]) for m in matches
                         if m.open_time in y_next.index]
            if not next_dirs:
                results.append({
                    "open_time": ts, "regime_filter": filter_on,
                    "true_dir": true_dir, "n_matches": 0,
                    "majority": np.nan, "hit": np.nan,
                })
                continue
            maj = int(round(np.mean(next_dirs)))
            results.append({
                "open_time": ts, "regime_filter": filter_on,
                "true_dir": true_dir, "n_matches": len(matches),
                "majority": maj, "hit": int(maj == true_dir),
            })

    df = pd.DataFrame(results)

    summary: dict[str, float] = {}
    for filter_on, sub in df.groupby("regime_filter"):
        valid = sub.dropna(subset=["hit"])
        key = "with_regime_filter" if filter_on else "no_filter"
        summary[f"{key}__hit_rate"] = float(valid["hit"].mean()) if len(valid) else float("nan")
        summary[f"{key}__n_queries"] = float(len(valid))
        summary[f"{key}__avg_n_matches"] = float(valid["n_matches"].mean()) if len(valid) else float("nan")

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["k"] = k
    summary["test_bars"] = int(len(test_times))

    # Persist
    out_dir = PROJECT_ROOT / "experiments" / "retrieval" / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "eval_pattern_engine.csv", index=False)
    (out_dir / "eval_pattern_engine.json").write_text(json.dumps(summary, indent=2))

    logger.success(
        f"hit-rate  no_filter: {summary.get('no_filter__hit_rate', float('nan')):.4f}   "
        f"regime: {summary.get('with_regime_filter__hit_rate', float('nan')):.4f}"
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", default="candle", choices=["candle", "gaf"])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--max-test-bars", type=int, default=2000,
                   help="Subsample to keep runtime sane (set 0 for all).")
    args = p.parse_args()

    summary = evaluate(
        pair=args.pair, interval=args.interval, encoder=args.encoder,
        k=args.k, test_frac=args.test_frac,
        max_test_bars=None if args.max_test_bars == 0 else args.max_test_bars,
    )
    print("\n=== Pattern-engine retrieval summary ===")
    for kk, vv in summary.items():
        print(f"  {kk}: {vv}")
