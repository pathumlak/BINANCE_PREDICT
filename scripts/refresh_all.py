"""One-command refresh driver — brings every artefact on disk up to today.

Chains the phase-by-phase entry points in the correct order:

    Phase 1  ohlcv         → scripts/fetch_historical.py
    Phase 4  news_score    → scripts/score_news.py
    Phase 4  news_features → scripts/build_sentiment_features.py
    Phase 3  embeddings    → scripts/extract_chart_embeddings.py   [SLOW]
    Phase 6  regimes       → scripts/fit_hmm_regimes.py
    Phase 6  faiss         → scripts/build_faiss_index.py

Skip / focus flags::

    python scripts/refresh_all.py                     # run everything
    python scripts/refresh_all.py --skip-cnn          # skip CNN retrain (~20 min saved)
    python scripts/refresh_all.py --skip-news         # skip sentiment scoring
    python scripts/refresh_all.py --only ohlcv        # just fill the OHLCV gap
    python scripts/refresh_all.py --only ohlcv regimes faiss

The driver invokes each entry point as a **subprocess** so a failure in
one step doesn't crash the driver — you get a clear per-step
success/failure summary at the end, and the exit code is non-zero if
anything failed.

Why subprocess (rather than importing)?
  * Each phase's script does its own argparse — subprocess keeps CLI
    behaviour identical to calling the scripts by hand.
  * Torch, transformers, hmmlearn, and faiss all mutate global state
    (CUDA context, huggingface caches, BLAS threads). Running them in
    fresh processes avoids the "second time it crashes" cache issues.
  * The Phase 3 CNN retraining allocates large tensors; a subprocess
    frees them cleanly on exit.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


@dataclass(frozen=True)
class Step:
    key: str                     # short slug used by --only / --skip
    description: str             # human-readable label for logs
    argv: Sequence[str]          # command tail (after python)
    slow: bool = False           # long-running steps get flagged in the summary


DEFAULT_PAIRS = ["BTCUSDT"]
DEFAULT_INTERVAL = "1h"


def _steps(pairs: list[str], interval: str) -> list[Step]:
    pairs_flag = ["--pairs", *pairs]
    interval_flag = ["--interval", interval]
    return [
        Step(
            key="ohlcv",
            description="Phase 1 — fetch/resume historical OHLCV",
            argv=["scripts/fetch_historical.py", *pairs_flag],
        ),
        Step(
            key="news_score",
            description="Phase 4 — score any unscored news articles",
            argv=["scripts/score_news.py"],
        ),
        Step(
            key="news_features",
            description="Phase 4 — build per-bar sentiment features",
            argv=["scripts/build_sentiment_features.py",
                  *pairs_flag, *interval_flag],
        ),
        Step(
            key="embeddings",
            description="Phase 3 — retrain CNN + dump embeddings [SLOW]",
            argv=["scripts/extract_chart_embeddings.py",
                  *pairs_flag, *interval_flag, "--encoder", "candle"],
            slow=True,
        ),
        Step(
            key="regimes",
            description="Phase 6 — fit HMM regime labels",
            argv=["scripts/fit_hmm_regimes.py",
                  *pairs_flag, *interval_flag],
        ),
        Step(
            key="faiss",
            description="Phase 6 — rebuild FAISS pattern index",
            argv=["scripts/build_faiss_index.py",
                  *pairs_flag, *interval_flag],
        ),
    ]


def _select(steps: list[Step], only: list[str] | None,
            skip: list[str], skip_cnn: bool, skip_news: bool) -> list[Step]:
    if only:
        wanted = set(only)
        unknown = wanted - {s.key for s in steps}
        if unknown:
            raise SystemExit(f"unknown --only step(s): {sorted(unknown)}")
        return [s for s in steps if s.key in wanted]

    dropped = set(skip)
    if skip_cnn:
        dropped.add("embeddings")
    if skip_news:
        dropped.update({"news_score", "news_features"})
    return [s for s in steps if s.key not in dropped]


def _run(step: Step) -> tuple[bool, float]:
    print(f"\n=== {step.description} ===", flush=True)
    print(f"    → {PYTHON} {' '.join(step.argv)}", flush=True)
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, *step.argv],
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0
    verdict = "OK" if ok else f"FAILED (exit={result.returncode})"
    print(f"    {verdict}  in {elapsed:0.1f}s", flush=True)
    return ok, elapsed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    p.add_argument("--interval", default=DEFAULT_INTERVAL)
    p.add_argument("--only", nargs="+",
                   help="Run only these steps (by key).")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Skip these steps (by key).")
    p.add_argument("--skip-cnn", action="store_true",
                   help="Shortcut: skip the slow Phase 3 CNN retraining.")
    p.add_argument("--skip-news", action="store_true",
                   help="Shortcut: skip Phase 4 news scoring + features.")
    p.add_argument("--list", action="store_true",
                   help="List available steps and exit.")
    args = p.parse_args()

    all_steps = _steps(args.pairs, args.interval)

    if args.list:
        print("available steps:")
        for s in all_steps:
            tag = " [slow]" if s.slow else ""
            print(f"  {s.key:<15} {s.description}{tag}")
        sys.exit(0)

    plan = _select(all_steps, args.only, args.skip,
                   args.skip_cnn, args.skip_news)
    if not plan:
        print("no steps selected — nothing to do")
        sys.exit(0)

    print("Refresh plan:")
    for s in plan:
        tag = " [SLOW]" if s.slow else ""
        print(f"  · {s.key:<15} {s.description}{tag}")
    print()

    results: list[tuple[Step, bool, float]] = []
    t_total = time.time()
    for step in plan:
        ok, elapsed = _run(step)
        results.append((step, ok, elapsed))
    total = time.time() - t_total

    print("\n" + "=" * 60)
    print(f"REFRESH SUMMARY — {total:0.1f}s total")
    print("-" * 60)
    n_ok = sum(1 for _, ok, _ in results if ok)
    for step, ok, elapsed in results:
        symbol = "OK  " if ok else "FAIL"
        print(f"  [{symbol}] {step.key:<15} {elapsed:>7.1f}s   {step.description}")
    print("-" * 60)
    print(f"{n_ok}/{len(results)} steps succeeded")
    print("=" * 60)

    if n_ok != len(results):
        sys.exit(1)
    print(
        "\nAll steps OK. Restart the dashboard to pick up the fresh artefacts:\n"
        "  python scripts/run_dashboard.py"
    )
