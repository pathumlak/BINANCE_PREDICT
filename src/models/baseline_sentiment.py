"""Sentiment-only toy baseline.

Predicts direction from per-bar sentiment features alone. Intended as a
sanity check to answer: "Does sentiment, on its own, carry any signal
on the news-overlap window?". The real value of sentiment shows up in
Phase 5's late-fusion classifier — this baseline just verifies the
pipeline produces non-trivial features.

Important constraint: sentiment data only exists from "now-forward"
because RSS / CryptoPanic don't backfill. The runner therefore filters
both train and test indices down to bars that have at least one article
in their lookback window. If the overlap is too short the model falls
back to majority-class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.base import Baseline

_FEATURES = ["sent_count", "sent_mean", "sent_std", "sent_min",
             "sent_max", "sent_last", "sent_conf_mean"]


class SentimentBaseline(Baseline):
    name = "sentiment_logreg"

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        self._cls = LogisticRegression(
            penalty="l2", C=1.0, max_iter=200, class_weight="balanced",
        )
        self._fallback_p1 = 0.5
        self._scaler_mean = None
        self._scaler_std = None
        self._features: pd.DataFrame | None = None
        self._pair: str | None = None
        self._interval: str | None = None

    # ------------------------------------------------------------------
    def attach_context(self, pair: str, interval: str) -> None:
        self._pair = pair
        self._interval = interval

    def _load_features(self) -> pd.DataFrame:
        if self._features is not None:
            return self._features
        if not self._pair:
            raise RuntimeError("attach_context not called")
        cfg = load_config()
        path = cfg.storage.root_path / "features_sentiment" / self._pair / f"{self._interval}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"sentiment features missing: {path}. "
                "Run scripts/build_sentiment_features.py first."
            )
        df = pd.read_parquet(path)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()
        self._features = df
        return df

    def _slice(self, X: pd.DataFrame) -> pd.DataFrame | None:
        feats = self._load_features()
        # Index of X is open_time. Inner-join to feats then keep only rows
        # with at least one article (the rest carry no signal).
        joined = feats.reindex(X.index)
        joined = joined[joined["sent_count"] > 0]
        return joined if len(joined) else None

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        sub = self._slice(X_train)
        if sub is None or len(sub) < 32:
            self._fallback_p1 = float(y_train.mean())
            return
        y = y_train.reindex(sub.index).astype(int).to_numpy()
        Xn = sub[_FEATURES].to_numpy(dtype=np.float64)
        self._scaler_mean = Xn.mean(axis=0)
        self._scaler_std = Xn.std(axis=0) + 1e-8
        Xn = (Xn - self._scaler_mean) / self._scaler_std
        self._cls.fit(Xn, y)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        out = np.full(len(X_test), self._fallback_p1, dtype=float)
        if self._scaler_mean is None:
            return out
        sub = self._slice(X_test)
        if sub is None:
            return out
        Xn = sub[_FEATURES].to_numpy(dtype=np.float64)
        Xn = (Xn - self._scaler_mean) / self._scaler_std
        probs = self._cls.predict_proba(Xn)[:, 1]
        # Map back to test rows
        ts_to_pos = {pd.Timestamp(t): i for i, t in enumerate(X_test.index)}
        for ts, p in zip(sub.index, probs):
            pos = ts_to_pos.get(pd.Timestamp(ts))
            if pos is not None:
                out[pos] = float(p)
        return out
