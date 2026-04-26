"""Common interface every Phase 2 baseline implements.

Models are tiny stateful objects with two required methods:

  ``fit(X_train, y_train)``      — train on a chronological slice
  ``predict_proba(X_test) -> p`` — return P(y=1) per row in [0, 1]

We derive ``predict(X)`` automatically by thresholding at 0.5 in the
runner. This keeps every baseline interchangeable from the orchestrator's
perspective.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Baseline(ABC):
    name: str = "baseline"

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return P(y=1) for every row, shape (len(X_test),)."""
        ...

    def predict(self, X_test: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X_test) >= threshold).astype(int)
