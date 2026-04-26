"""Phase 2 orchestrator.

For each (model, pair) combination:
  1. Load OHLCV → engineered features + label.
  2. Generate walk-forward CV folds.
  3. Train on each train slice, predict on test slice, compute metrics.
  4. Persist predictions + per-fold + aggregate metrics.

The output directory layout matches what summarise_baselines.py expects:
  experiments/baselines/<pair>/<model>/predictions.parquet
  experiments/baselines/<pair>/<model>/metrics.json
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT
from src.eval.metrics import aggregate_folds, evaluate_fold, FoldMetrics
from src.eval.walk_forward import auto_config, walk_forward_splits
from src.features.dataset import Dataset, build_dataset
from src.models.base import Baseline
from src.models.baseline_lstm import LSTMBaseline
from src.models.baseline_naive import MajorityClass, Persistence, RandomCoin
from src.models.baseline_patchtst import PatchTSTBaseline
from src.models.baseline_xgboost import XGBoostBaseline


# Registry: name -> factory callable. Add new baselines here.
MODEL_REGISTRY: dict[str, Callable[[], Baseline]] = {
    "naive_majority":   lambda: MajorityClass(),
    "naive_persistence": lambda: Persistence(),
    "naive_random":     lambda: RandomCoin(),
    "xgboost":          lambda: XGBoostBaseline(),
    "lstm":             lambda: LSTMBaseline(),
    "patchtst":         lambda: PatchTSTBaseline(),
}


EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments" / "baselines"


def _next_log_return(close: pd.Series) -> pd.Series:
    """log(close_{t+1}/close_t) — used for trading-style metrics."""
    return np.log(close.shift(-1) / close)


def run_one(model_name: str, pair: str, interval: str,
            n_folds: int = 6, train_frac: float = 0.5) -> dict:
    """Train + evaluate one (model, pair, interval). Persist outputs."""
    logger.info(f"=== {model_name}  /  {pair} {interval} ===")
    t0 = time.time()

    ds: Dataset = build_dataset(pair, interval)
    next_ret = _next_log_return(ds.close)

    cv_cfg = auto_config(len(ds.X), n_folds=n_folds, train_frac=train_frac)
    logger.info(
        f"data: {len(ds.X):,} rows × {len(ds.feature_names)} feats   "
        f"folds: {n_folds}   initial_train: {cv_cfg.initial_train_size:,}   "
        f"test_size: {cv_cfg.test_size:,}"
    )

    fold_metrics: list[FoldMetrics] = []
    pred_rows: list[pd.DataFrame] = []

    for k, (tr_idx, te_idx) in enumerate(walk_forward_splits(len(ds.X), cv_cfg)):
        X_tr, y_tr = ds.X.iloc[tr_idx], ds.y.iloc[tr_idx]
        X_te, y_te = ds.X.iloc[te_idx], ds.y.iloc[te_idx]
        nr_te = next_ret.iloc[te_idx].fillna(0.0).to_numpy()

        model = MODEL_REGISTRY[model_name]()
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)
        pred = (prob >= 0.5).astype(int)

        m = evaluate_fold(y_te.to_numpy(), pred, prob, nr_te)
        fold_metrics.append(m)
        logger.info(
            f"  fold {k}: acc={m.accuracy:.3f}  f1={m.f1:.3f}  "
            f"sharpe/bar={m.sharpe_per_bar:.3f}"
        )

        pred_rows.append(pd.DataFrame({
            "open_time":  ds.X.index[te_idx],
            "fold":       k,
            "y_true":     y_te.to_numpy(),
            "y_pred":     pred,
            "y_prob":     prob,
            "next_ret":   nr_te,
        }))

    agg = aggregate_folds(fold_metrics)
    elapsed = time.time() - t0
    agg["elapsed_seconds"] = round(elapsed, 2)
    agg["model"] = model_name
    agg["pair"] = pair
    agg["interval"] = interval

    # Persist
    out_dir = EXPERIMENTS_ROOT / pair / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows, ignore_index=True).to_parquet(
        out_dir / "predictions.parquet", compression="snappy", index=False,
    )
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump({
            "summary": agg,
            "folds": [m.as_dict() for m in fold_metrics],
        }, fh, indent=2)
    logger.success(
        f"done in {elapsed:.1f}s — overall acc={agg.get('accuracy', float('nan')):.3f}  "
        f"sharpe/bar={agg.get('sharpe_per_bar', float('nan')):.3f}"
    )
    return agg


def run_all(models: list[str], pairs: list[str], interval: str,
            n_folds: int = 6, train_frac: float = 0.5) -> pd.DataFrame:
    rows: list[dict] = []
    for pair in pairs:
        for m in models:
            try:
                rows.append(run_one(m, pair, interval, n_folds=n_folds, train_frac=train_frac))
            except Exception as e:  # noqa: BLE001
                logger.exception(f"FAILED  {m} / {pair}: {e}")
    return pd.DataFrame(rows)
