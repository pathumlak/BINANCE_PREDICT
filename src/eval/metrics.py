"""Evaluation metrics for binary direction prediction.

Two families:

1. **Pure classification** — accuracy, F1, MCC, log-loss. Standard.
2. **Trading-style** — sign-aware Sharpe-like ratio and hit rate, computed
   by treating the model's prediction (1 = long, 0 = short) as a position
   on the *next* candle's log-return.

The trading metrics are NOT a backtest — there's no fees, slippage, or
position sizing. They're a sanity-check that high accuracy translates to
positive PnL. A model with 51% accuracy that's right on big moves can
beat a 55% model that's right only on small ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass
class FoldMetrics:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    log_loss: float
    hit_rate: float          # fraction of trades whose return sign matched the position
    pnl_log: float           # cumulative log-return of the position-following strategy
    sharpe_per_bar: float    # mean / std of position * return (per bar)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": int(self.n),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "mcc": float(self.mcc),
            "log_loss": float(self.log_loss),
            "hit_rate": float(self.hit_rate),
            "pnl_log": float(self.pnl_log),
            "sharpe_per_bar": float(self.sharpe_per_bar),
        }


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    next_return_log: np.ndarray,
) -> FoldMetrics:
    """Compute one fold's metrics.

    ``next_return_log`` should be the log-return between candle t and t+1
    aligned with each prediction. Used only by the trading metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)

    # Map prediction {0, 1} to position {-1, +1} so the math reads naturally.
    position = np.where(y_pred == 1, 1, -1)
    actual = np.where(y_true == 1, 1, -1)
    hits = (position == actual).mean()

    bar_pnl = position * next_return_log
    cum_log = float(np.nansum(bar_pnl))
    sharpe = (
        float(np.nanmean(bar_pnl) / np.nanstd(bar_pnl))
        if np.nanstd(bar_pnl) > 0 else 0.0
    )

    return FoldMetrics(
        n=len(y_true),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        log_loss=log_loss(y_true, y_prob, labels=[0, 1]),
        hit_rate=float(hits),
        pnl_log=cum_log,
        sharpe_per_bar=sharpe,
    )


def aggregate_folds(folds: list[FoldMetrics]) -> dict[str, float]:
    """Sample-size weighted mean across folds (fairer than plain mean)."""
    if not folds:
        return {}
    total_n = sum(f.n for f in folds)
    keys = [k for k in folds[0].as_dict() if k != "n"]
    out: dict[str, float] = {"n_total": float(total_n), "n_folds": len(folds)}
    for k in keys:
        out[k] = sum(f.n * getattr(f, k) for f in folds) / total_n
    # PnL is additive across folds, so re-sum it (don't weight-average).
    out["pnl_log"] = float(sum(f.pnl_log for f in folds))
    return out
