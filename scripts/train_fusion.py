"""Phase 5 — train + evaluate the multimodal late-fusion classifier.

Walk-forward CV on a single pair (default BTCUSDT 1h). Runs four
ablation variants in one shot so the impact of each modality is directly
measurable:

    fusion_num                — numeric features only (sanity sibling to xgboost)
    fusion_num_cnn            — numeric + chart-CNN embeddings
    fusion_num_sent           — numeric + sentiment features
    fusion_num_cnn_sent       — full fusion (default headline result)

Each variant uses inductive conformal prediction with alpha=0.10 by
default (i.e. nominal 90 % coverage of the true class).

Outputs land in:

    experiments/fusion/<PAIR>/<variant>/predictions.parquet
    experiments/fusion/<PAIR>/<variant>/metrics.json

The predictions parquet carries the conformal set membership per row
(`in_set_0`, `in_set_1`) so coverage and average set size can be audited.

Pre-flight checks
-----------------
Before kicking off the sweep we verify:
  * chart embeddings exist for the requested pair/interval
  * sentiment features exist for the requested pair/interval
Each missing input prints a clear "run this first" command.

Usage
-----
    # Smoke (1 fold)
    python scripts/train_fusion.py --pairs BTCUSDT --folds 1 --variants fusion_num_cnn_sent

    # Full sweep — default
    python scripts/train_fusion.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.eval.metrics import aggregate_folds, evaluate_fold, FoldMetrics  # noqa: E402
from src.eval.walk_forward import auto_config, walk_forward_splits  # noqa: E402
from src.features.dataset import build_dataset  # noqa: E402
from src.models.baseline_fusion import FusionBaseline  # noqa: E402


EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "fusion"


# Ablation registry: name -> factory returning a fresh FusionBaseline.
VARIANT_REGISTRY: dict[str, Callable[[], FusionBaseline]] = {
    "fusion_num":          lambda: FusionBaseline(use_cnn=False, use_sentiment=False),
    "fusion_num_cnn":      lambda: FusionBaseline(use_cnn=True,  use_sentiment=False),
    "fusion_num_sent":     lambda: FusionBaseline(use_cnn=False, use_sentiment=True),
    "fusion_num_cnn_sent": lambda: FusionBaseline(use_cnn=True,  use_sentiment=True),
}


# ---------------------------------------------------------------------------
def _next_log_return(close: pd.Series) -> pd.Series:
    return np.log(close.shift(-1) / close)


def _preflight(pair: str, interval: str, need_cnn: bool, need_sent: bool) -> None:
    """Fail fast if a modality input is missing — with the exact fix command."""
    if need_cnn:
        p = (PROJECT_ROOT / "experiments" / "embeddings" / pair
             / interval / "candle" / "embeddings.parquet")
        if not p.exists():
            logger.error(
                f"CNN embeddings missing for {pair} {interval}.\n"
                f"   Run:  python scripts/extract_chart_embeddings.py "
                f"--pairs {pair} --interval {interval} --encoder candle"
            )
            sys.exit(2)
    if need_sent:
        p = (PROJECT_ROOT / "data" / "features_sentiment"
             / pair / f"{interval}.parquet")
        if not p.exists():
            logger.error(
                f"sentiment features missing for {pair} {interval}.\n"
                f"   Run:  python scripts/build_sentiment_features.py "
                f"--pairs {pair} --interval {interval}"
            )
            sys.exit(2)


# ---------------------------------------------------------------------------
def run_variant(
    variant: str, pair: str, interval: str,
    n_folds: int, train_frac: float,
) -> dict:
    """Train + evaluate one fusion variant on one (pair, interval)."""
    logger.info(f"=== {variant}  /  {pair} {interval} ===")
    t0 = time.time()

    ds = build_dataset(pair, interval)
    next_ret = _next_log_return(ds.close)

    cv_cfg = auto_config(len(ds.X), n_folds=n_folds, train_frac=train_frac)
    logger.info(
        f"data: {len(ds.X):,} rows × {len(ds.feature_names)} numeric feats   "
        f"folds: {n_folds}   initial_train: {cv_cfg.initial_train_size:,}   "
        f"test_size: {cv_cfg.test_size:,}"
    )

    fold_metrics: list[FoldMetrics] = []
    pred_rows: list[pd.DataFrame] = []
    cp_rows: list[dict] = []

    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(len(ds.X), cv_cfg)):
        X_tr, y_tr = ds.X.iloc[tr_idx], ds.y.iloc[tr_idx]
        X_te, y_te = ds.X.iloc[te_idx], ds.y.iloc[te_idx]
        nr_te = next_ret.iloc[te_idx].fillna(0.0).to_numpy()

        model = VARIANT_REGISTRY[variant]()
        model.attach_context(pair, interval)
        model.fit(X_tr, y_tr)

        prob = model.predict_proba(X_te)
        pred = (prob >= 0.5).astype(int)

        # Conformal sets (n, 2): cols are class 0, class 1.
        cp_sets = model.predict_set(X_te)
        in_set_0 = cp_sets[:, 0].astype(int)
        in_set_1 = cp_sets[:, 1].astype(int)
        set_size = in_set_0 + in_set_1
        # Empirical coverage = fraction where the TRUE label was in the set.
        y_arr = y_te.to_numpy().astype(int)
        coverage = cp_sets[np.arange(len(y_arr)), y_arr].mean()

        m = evaluate_fold(y_arr, pred, prob, nr_te)
        fold_metrics.append(m)
        cp_rows.append({
            "fold": k,
            "coverage": float(coverage),
            "avg_set_size": float(set_size.mean()),
            "singletons": float((set_size == 1).mean()),
        })
        logger.info(
            f"  fold {k}: acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
            f"cp_cov={coverage:.3f}  set_size={set_size.mean():.2f}"
        )

        pred_rows.append(pd.DataFrame({
            "open_time": ds.X.index[te_idx],
            "fold":      k,
            "y_true":    y_arr,
            "y_pred":    pred,
            "y_prob":    prob,
            "next_ret":  nr_te,
            "in_set_0":  in_set_0,
            "in_set_1":  in_set_1,
        }))

    if not fold_metrics:
        # walk_forward_splits emitted zero folds — usually means the
        # combination of n_folds + train_frac + the gap pushed the final
        # test window past the end of the dataset. Bail out with a clear
        # message rather than failing later on an empty concat().
        raise RuntimeError(
            f"walk-forward CV produced 0 folds for {pair} {interval} "
            f"(n_rows={len(ds.X):,}, n_folds={n_folds}, "
            f"train_frac={train_frac}). Try a smaller train_frac or "
            f"more folds so each fold leaves headroom for the gap row."
        )

    agg = aggregate_folds(fold_metrics)
    elapsed = time.time() - t0
    agg["elapsed_seconds"] = round(elapsed, 2)
    agg["model"] = variant
    agg["pair"] = pair
    agg["interval"] = interval
    # Average conformal stats across folds, weighted by fold size.
    n_total = sum(m.n for m in fold_metrics) or 1
    agg["cp_coverage"] = float(
        sum(cp_rows[i]["coverage"] * fold_metrics[i].n for i in range(len(cp_rows)))
        / n_total
    )
    agg["cp_avg_set_size"] = float(
        sum(cp_rows[i]["avg_set_size"] * fold_metrics[i].n for i in range(len(cp_rows)))
        / n_total
    )
    agg["cp_singleton_frac"] = float(
        sum(cp_rows[i]["singletons"] * fold_metrics[i].n for i in range(len(cp_rows)))
        / n_total
    )

    out_dir = EXPERIMENTS_ROOT / pair / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows, ignore_index=True).to_parquet(
        out_dir / "predictions.parquet", compression="snappy", index=False,
    )
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump({
            "summary": agg,
            "folds": [m.as_dict() for m in fold_metrics],
            "conformal_folds": cp_rows,
        }, fh, indent=2)
    logger.success(
        f"done in {elapsed:.1f}s — acc={agg.get('accuracy', float('nan')):.3f}  "
        f"cp_cov={agg['cp_coverage']:.3f}  set_size={agg['cp_avg_set_size']:.2f}"
    )
    return agg


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Phase 5: multimodal fusion sweep.")
    p.add_argument("--pairs", nargs="+", default=["BTCUSDT"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--variants", nargs="+", default=list(VARIANT_REGISTRY.keys()),
                   choices=list(VARIANT_REGISTRY.keys()))
    args = p.parse_args()

    # Pre-flight: check inputs for every (pair × variant) we'll touch.
    for pair in args.pairs:
        need_cnn = any("cnn" in v for v in args.variants)
        need_sent = any("sent" in v for v in args.variants)
        _preflight(pair, args.interval, need_cnn=need_cnn, need_sent=need_sent)

    rows: list[dict] = []
    for pair in args.pairs:
        for variant in args.variants:
            try:
                rows.append(run_variant(
                    variant, pair, args.interval,
                    n_folds=args.folds, train_frac=args.train_frac,
                ))
            except Exception as e:  # noqa: BLE001
                logger.exception(f"FAILED  {variant} / {pair}: {e}")

    if not rows:
        logger.warning("no successful runs")
        sys.exit(1)

    df = pd.DataFrame(rows)
    cols = ["pair", "model", "accuracy", "f1", "mcc",
            "sharpe_per_bar", "pnl_log",
            "cp_coverage", "cp_avg_set_size", "cp_singleton_frac",
            "elapsed_seconds"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + df[cols].to_string(index=False))

    # Per-pair summary CSV (consumed by summarize_fusion.py).
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPERIMENTS_ROOT / "summary.csv", index=False)
    logger.success(f"wrote {EXPERIMENTS_ROOT / 'summary.csv'}")
