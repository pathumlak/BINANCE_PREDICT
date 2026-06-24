"""Summarise Phase 5 fusion results and DM-test vs Phase 2 / 3 baselines.

Reads:
  * experiments/fusion/<PAIR>/<variant>/predictions.parquet
  * experiments/baselines/<PAIR>/<model>/predictions.parquet
    (so fusion variants can be DM-tested against the strongest existing
    single-modality baseline — usually xgboost or cnn_candle).

Writes:
  * Console table — fusion variants ranked, conformal coverage + set size
    shown alongside accuracy / Sharpe / PnL.
  * experiments/fusion/summary.csv — flat table for plotting.
  * experiments/fusion/dm_vs_baselines.csv — every fusion variant against
    every Phase 2/3 baseline, signed DM p-value.
  * experiments/fusion/PHASE5_RESULTS.md — short markdown report you can
    paste into the viva.

Usage:
    python scripts/summarize_fusion.py
    python scripts/summarize_fusion.py --pair BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.eval.dm_test import diebold_mariano  # noqa: E402


FUSION_ROOT = PROJECT_ROOT / "experiments" / "fusion"
BASELINES_ROOT = PROJECT_ROOT / "experiments" / "baselines"


def _collect_summary(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in root.glob("*/*/metrics.json"):
        with metrics_path.open() as fh:
            payload = json.load(fh)
        rows.append(payload["summary"])
    return pd.DataFrame(rows)


def _load_predictions(root: Path, pair: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for p in sorted((root / pair).glob("*/predictions.parquet")):
        out[p.parent.name] = pd.read_parquet(p)
    return out


def _dm_cross(
    pair: str,
    fusion_preds: dict[str, pd.DataFrame],
    baseline_preds: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """DM signed p-values: rows = fusion variants, cols = baselines.

    Each pair is joined on (open_time, fold) — only the overlap window
    counts so the test is apples-to-apples.
    """
    rows: list[dict] = []
    for f_name, f_df in fusion_preds.items():
        row = {"variant": f_name, "pair": pair}
        for b_name, b_df in baseline_preds.items():
            merged = f_df.merge(b_df, on=["open_time", "fold"], suffixes=("_f", "_b"))
            if len(merged) < 30:
                row[b_name] = float("nan")
                continue
            r = diebold_mariano(
                y_true=merged["y_true_f"].to_numpy(),
                pred_a=merged["y_pred_f"].to_numpy(),   # fusion
                pred_b=merged["y_pred_b"].to_numpy(),   # baseline
                h=1, loss="zero_one",
            )
            signed_p = r["p_value"] if r["dm_stat"] >= 0 else -r["p_value"]
            row[b_name] = round(signed_p, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table.

    Hand-rolled so we don't pull in `tabulate` just for this.
    """
    if df.empty:
        return "*(no rows)*"
    cols = [str(c) for c in df.columns]
    rows: list[list[str]] = []
    for _, r in df.iterrows():
        row: list[str] = []
        for v in r.tolist():
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append("" if v is None else str(v))
        rows.append(row)

    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = "\n".join(
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |"
        for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def _write_markdown(
    out_path: Path,
    summary: pd.DataFrame,
    dm_tables: dict[str, pd.DataFrame],
) -> None:
    lines: list[str] = ["# Phase 5 — Multimodal Fusion Results\n"]
    lines.append("## Fusion variants — headline metrics\n")
    cols = ["pair", "model", "accuracy", "f1", "mcc",
            "sharpe_per_bar", "pnl_log",
            "cp_coverage", "cp_avg_set_size", "cp_singleton_frac"]
    cols = [c for c in cols if c in summary.columns]
    lines.append(_md_table(summary[cols].round(4)))
    lines.append("")
    lines.append(
        "`cp_coverage` is the empirical fraction of test rows whose true "
        "label fell inside the 90 % conformal prediction set "
        "(nominal target 0.90). `cp_avg_set_size` ∈ [1, 2] is the "
        "mean prediction-set cardinality — closer to 1 means the model "
        "is confident enough to commit to a single class."
    )

    for pair, dm in dm_tables.items():
        if dm.empty:
            continue
        lines.append(f"\n## Diebold-Mariano vs Phase 2 / 3 baselines — {pair}\n")
        lines.append(
            "Signed p-value: negative ⇒ the fusion variant (row) beats "
            "the baseline (column); |p| < 0.05 ⇒ significant.\n"
        )
        lines.append(_md_table(dm))
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default=None)
    args = parser.parse_args()

    summary = _collect_summary(FUSION_ROOT)
    if summary.empty:
        print("No experiments found under experiments/fusion/. "
              "Run scripts/train_fusion.py first.")
        sys.exit(1)

    cols = ["pair", "model", "accuracy", "f1", "mcc",
            "sharpe_per_bar", "pnl_log",
            "cp_coverage", "cp_avg_set_size", "cp_singleton_frac",
            "n_total", "n_folds", "elapsed_seconds"]
    cols = [c for c in cols if c in summary.columns]
    summary = summary.sort_values(["pair", "accuracy"], ascending=[True, False])
    print("\n=== Phase 5 — Fusion summary ===\n")
    print(summary[cols].round(4).to_string(index=False))

    FUSION_ROOT.mkdir(parents=True, exist_ok=True)
    summary[cols].to_csv(FUSION_ROOT / "summary.csv", index=False)
    print(f"\nWrote {FUSION_ROOT / 'summary.csv'}")

    pairs = [args.pair] if args.pair else sorted(summary["pair"].unique())
    dm_tables: dict[str, pd.DataFrame] = {}
    cross_rows: list[pd.DataFrame] = []
    for p in pairs:
        f_preds = _load_predictions(FUSION_ROOT, p)
        b_preds = _load_predictions(BASELINES_ROOT, p)
        if not f_preds or not b_preds:
            continue
        dm = _dm_cross(p, f_preds, b_preds)
        dm_tables[p] = dm
        cross_rows.append(dm)
        print(f"\n=== Fusion vs baselines (signed p)  ({p}) ===")
        print(dm.to_string(index=False))

    if cross_rows:
        all_dm = pd.concat(cross_rows, ignore_index=True)
        all_dm.to_csv(FUSION_ROOT / "dm_vs_baselines.csv", index=False)
        print(f"\nWrote {FUSION_ROOT / 'dm_vs_baselines.csv'}")

    md_path = FUSION_ROOT / "PHASE5_RESULTS.md"
    _write_markdown(md_path, summary[cols].copy(), dm_tables)
    print(f"Wrote {md_path}")
