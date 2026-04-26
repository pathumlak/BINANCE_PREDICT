"""Aggregate Phase 2 results into a comparison table + DM-significance matrix.

Reads everything under ``experiments/baselines/`` and produces:

  * Console table: rows = (pair × model), columns = key metrics.
  * Per-pair Diebold-Mariano matrix (best model vs the rest).
  * ``experiments/baselines/summary.csv`` for downstream plotting.

Usage:
    python scripts/summarize_baselines.py
    python scripts/summarize_baselines.py --pair BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.eval.dm_test import diebold_mariano  # noqa: E402

EXPERIMENTS = PROJECT_ROOT / "experiments" / "baselines"


def collect_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in EXPERIMENTS.glob("*/*/metrics.json"):
        with metrics_path.open() as fh:
            payload = json.load(fh)
        rows.append(payload["summary"])
    return pd.DataFrame(rows)


def dm_matrix_for_pair(pair: str) -> pd.DataFrame:
    """Pairwise DM p-values among every model trained on ``pair``."""
    pred_paths = sorted((EXPERIMENTS / pair).glob("*/predictions.parquet"))
    if len(pred_paths) < 2:
        return pd.DataFrame()
    preds = {p.parent.name: pd.read_parquet(p) for p in pred_paths}

    # Align everything on (open_time, fold) to be safe.
    models = list(preds.keys())
    out = pd.DataFrame(index=models, columns=models, dtype=float)

    for a, b in combinations(models, 2):
        merged = preds[a].merge(
            preds[b], on=["open_time", "fold"],
            suffixes=("_a", "_b"),
        )
        if merged.empty:
            continue
        result = diebold_mariano(
            y_true=merged["y_true_a"].to_numpy(),
            pred_a=merged["y_pred_a"].to_numpy(),
            pred_b=merged["y_pred_b"].to_numpy(),
            h=1,
            loss="zero_one",
        )
        # Negative dm_stat → A beats B. Store signed p so direction is visible.
        signed_p = result["p_value"] if result["dm_stat"] >= 0 else -result["p_value"]
        out.loc[a, b] = signed_p
        out.loc[b, a] = -signed_p
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None,
                        help="Show DM matrix for this pair only. Default: all.")
    args = parser.parse_args()

    df = collect_summary()
    if df.empty:
        print("No experiments found under experiments/baselines/. "
              "Run scripts/train_baselines.py first.")
        sys.exit(1)

    cols = ["pair", "model", "accuracy", "f1", "mcc",
            "sharpe_per_bar", "pnl_log", "n_total", "n_folds", "elapsed_seconds"]
    df = df.sort_values(["pair", "accuracy"], ascending=[True, False])
    print("\n=== Aggregated Metrics ===\n")
    print(df[cols].to_string(index=False))

    csv_out = EXPERIMENTS / "summary.csv"
    df[cols].to_csv(csv_out, index=False)
    print(f"\nWrote {csv_out}")

    pairs = [args.pair] if args.pair else sorted(df["pair"].unique())
    for p in pairs:
        m = dm_matrix_for_pair(p)
        if m.empty:
            continue
        print(f"\n=== Diebold-Mariano signed p-values  ({p}) ===")
        print("(Negative p ⇒ row beats column; |p|<0.05 ⇒ significant)\n")
        print(m.round(3).to_string())
