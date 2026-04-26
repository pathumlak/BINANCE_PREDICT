"""Target / label construction for Phase 2.

Phase 2 framing (per scope decision):
  * Binary direction over the *next* candle.
  * y_t = 1 if close_{t+1} > close_t, else 0.

This file is the *single* source of truth for what the label means. Every
model (naive, XGB, LSTM, PatchTST) consumes labels from here, so there's
zero risk of mismatched targets across baselines.
"""
from __future__ import annotations

import pandas as pd


def make_direction_label(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary direction over the next ``horizon`` candles.

    Parameters
    ----------
    close : pd.Series
        Close-price series indexed by ``open_time``.
    horizon : int
        How many candles ahead to look. Default 1 = next candle.

    Returns
    -------
    pd.Series of int64 in {0, 1}
        1 if close[t + horizon] > close[t], else 0.
        The last ``horizon`` rows are NaN-then-dropped: their future is unknown.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future = close.shift(-horizon)
    label = (future > close).astype("Int64")
    # Where future is NaN (the tail), label is also NaN.
    label[future.isna()] = pd.NA
    return label


def make_return_label(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Continuous log-return label, kept around for later phases.

    Not used directly in Phase 2 baselines but useful for the regression
    head we may add in Phase 5.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    import numpy as np
    return pd.Series(np.log(close.shift(-horizon) / close), index=close.index)
