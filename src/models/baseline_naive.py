"""Trivial baselines.

Three flavours so we can claim every model "beats <something dumb>":

* ``MajorityClass`` — predict whichever class was most frequent in training.
* ``Persistence``  — predict the same direction as the *previous* candle.
* ``RandomCoin``   — predict 1 with 50% probability (sanity floor).

If a fancy model can't beat ``MajorityClass`` and ``Persistence`` you do
not have a model. You have noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Baseline


class MajorityClass(Baseline):
    name = "naive_majority"

    def __init__(self) -> None:
        self.p1: float = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.p1 = float(y_train.mean())

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return np.full(len(X_test), self.p1, dtype=float)


class Persistence(Baseline):
    """Predict next direction = sign of the immediately previous return.

    Uses the ``ret_1`` feature directly so we don't accidentally peek at close.
    """
    name = "naive_persistence"

    def __init__(self, feature: str = "ret_1") -> None:
        self.feature = feature

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.feature not in X_train.columns:
            raise KeyError(
                f"Persistence needs feature '{self.feature}' which is missing"
            )

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        # Probability is hard 0/1 — we map negative-return rows to 0, positive to 1.
        return (X_test[self.feature].to_numpy() > 0).astype(float)


class RandomCoin(Baseline):
    name = "naive_random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        pass

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.rng.uniform(size=len(X_test))
