"""XGBoost classifier on engineered features.

In practice this is the toughest baseline to beat in tabular crypto-ML —
gradient-boosted trees handle non-linearity, missing-value tolerance, and
feature interactions for free, and they're robust to the noisy / heavy-
tailed return distributions we have here.

Hyperparameters here are deliberately *modest*. We're producing a strong
honest baseline, not chasing a leaderboard. Tuning belongs in Phase 5
once we know which signals matter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base import Baseline


class XGBoostBaseline(Baseline):
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",        # fast on CPU, no GPU required
            random_state=random_state,
            verbosity=0,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model.fit(X_train.values, y_train.values)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X_test.values)[:, 1]
