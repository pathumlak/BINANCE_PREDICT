"""Walk-forward cross-validation splits.

Time-series CV is *not* k-fold. We must never train on a future bar and
test on a past bar — that's a leak that artificially inflates accuracy.

This module yields ``(train_idx, test_idx)`` tuples following an
*expanding-window* scheme:

    fold 0 :  train [0 .. T0]    test [T0+1 .. T0+H]
    fold 1 :  train [0 .. T0+H]  test [T0+H+1 .. T0+2H]
    ...

The training window grows; the test window slides forward by ``test_size``
each fold. We always leave a ``gap`` between the last training row and the
first test row (default 1) so the label of the last train row — which
peeks one step into the future by definition — can't leak into the test
window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class WalkForwardConfig:
    initial_train_size: int      # rows in the very first training window
    test_size: int               # rows per test fold
    step_size: int | None = None # how far the train window grows per fold
                                 #   (None → step = test_size, i.e. non-overlapping tests)
    gap: int = 1                 # buffer rows between train end and test start
    max_folds: int | None = None # cap for fast iteration


def walk_forward_splits(
    n_samples: int, cfg: WalkForwardConfig
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) numpy arrays in chronological order."""
    step = cfg.step_size or cfg.test_size

    train_end = cfg.initial_train_size
    folds_yielded = 0
    while True:
        test_start = train_end + cfg.gap
        test_end = test_start + cfg.test_size
        if test_end > n_samples:
            break

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx

        folds_yielded += 1
        if cfg.max_folds is not None and folds_yielded >= cfg.max_folds:
            break
        train_end += step


def auto_config(n_samples: int, n_folds: int = 6, train_frac: float = 0.5) -> WalkForwardConfig:
    """Pick reasonable defaults given dataset size.

    Reserves the first ``train_frac`` of the data for the initial training
    window, then splits the remainder into ``n_folds`` non-overlapping
    test windows.
    """
    initial = int(n_samples * train_frac)
    remainder = n_samples - initial
    test_size = max(1, remainder // n_folds)
    return WalkForwardConfig(
        initial_train_size=initial,
        test_size=test_size,
        step_size=test_size,
        gap=1,
        max_folds=n_folds,
    )
