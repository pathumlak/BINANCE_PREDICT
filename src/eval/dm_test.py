"""Diebold-Mariano test for forecast comparison.

Tests whether the difference in forecast loss between two models is
statistically significant. Standard reference: Diebold & Mariano (1995).
The HLN small-sample correction (Harvey, Leybourne, Newbold 1997) is
applied, which is now the conventional default.

For our binary-direction baselines we use **0/1 loss** (= 1 - accuracy on
each prediction). For the regression head added later we'd switch to
squared-error loss.

Returns a dict with ``dm_stat``, ``p_value``, and a verdict string.
A negative ``dm_stat`` with small p means model A beats model B.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy import stats


LossKind = Literal["zero_one", "squared", "absolute"]


def _loss(y_true: np.ndarray, y_pred: np.ndarray, kind: LossKind) -> np.ndarray:
    if kind == "zero_one":
        return (y_true != y_pred).astype(float)
    if kind == "squared":
        return (y_true - y_pred) ** 2
    if kind == "absolute":
        return np.abs(y_true - y_pred)
    raise ValueError(kind)


def _autocov(d: np.ndarray, lag: int) -> float:
    n = len(d)
    if lag == 0:
        return float(np.var(d, ddof=0))
    d_centred = d - d.mean()
    return float(np.sum(d_centred[lag:] * d_centred[:-lag]) / n)


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int = 1,
    loss: LossKind = "zero_one",
) -> dict:
    """DM test (HLN-corrected) for predictions A vs B against ``y_true``.

    Parameters
    ----------
    y_true : array of true labels
    pred_a, pred_b : array of point predictions
    h : forecast horizon (used to choose number of autocov lags)
    loss : loss function — ``zero_one`` for classification, ``squared`` for regression
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a, pred_b must be same length")

    d = _loss(y_true, pred_a, loss) - _loss(y_true, pred_b, loss)
    n = len(d)
    if n < 8:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "verdict": "n too small for DM",
        }

    # Newey-West-style long-run variance estimate using h-1 lags.
    gamma0 = _autocov(d, 0)
    var_d = gamma0 + 2 * sum(_autocov(d, k) for k in range(1, h))
    var_d = max(var_d, 1e-12)
    dm = d.mean() / math.sqrt(var_d / n)

    # Harvey-Leybourne-Newbold small-sample correction.
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction

    # Two-sided p-value using Student-t with n-1 df (HLN recommendation).
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))

    if p < 0.05:
        verdict = "A beats B" if dm_hln < 0 else "B beats A"
    else:
        verdict = "no significant difference"

    return {
        "dm_stat": float(dm_hln),
        "p_value": float(p),
        "verdict": verdict,
        "n": int(n),
    }
