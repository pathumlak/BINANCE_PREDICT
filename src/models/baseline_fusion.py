"""Phase 5 — Multimodal late-fusion baseline.

Concatenates three modality feature blocks into a single per-bar vector
and fits a regularised logistic regression on top. Optionally wraps the
classifier in MAPIE's inductive (split) conformal classifier so each
prediction carries a calibrated 90 % prediction set.

The three modality blocks:

* **numeric**  – Phase 2 engineered features (`src.features.numerical`).
* **cnn**      – 128-dim chart-CNN embeddings produced by
  ``scripts/extract_chart_embeddings.py`` (Phase 3 model).
* **sentiment**– 7 per-bar sentiment features built by
  ``scripts/build_sentiment_features.py`` (Phase 4).

Bars with no news in their 24-hour lookback simply get the
zero-sentiment vector (matching what the Phase 4 aggregator already
emits) — we do NOT drop them, because doing so would shrink the test
window dramatically and make fusion incomparable to Phases 2/3.

Ablation flags
--------------
Pass ``use_cnn=False`` to evaluate (numeric + sentiment) only.
Pass ``use_sentiment=False`` to evaluate (numeric + cnn) only.
Pass both ``False`` to recover a numeric-only logistic regression
(useful as a sanity sibling to xgboost).

Why this fusion architecture
----------------------------
Late-fusion logistic regression is the textbook starting point for
multimodal classification (Baltrušaitis et al. 2018): cheap to fit,
trivially interpretable via per-feature coefficients, and easy to
ablate. It also keeps the conformal coverage analysis tractable —
MAPIE's split conformal classifier is well-defined for any
``sklearn``-compatible probabilistic classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.models.base import Baseline

# Per-bar sentiment columns (kept identical to baseline_sentiment.py so
# the same Phase 4 parquet drops in).
_SENT_FEATURES = [
    "sent_count", "sent_mean", "sent_std", "sent_min",
    "sent_max", "sent_last", "sent_conf_mean",
]
_EMB_DIM = 128  # ChartCNN embedding dimension (see src/vision/cnn.py)


class FusionBaseline(Baseline):
    """Concat-LR late-fusion classifier with optional conformal calibration."""

    name = "fusion_concat_lr"

    def __init__(
        self,
        *,
        use_cnn: bool = True,
        use_sentiment: bool = True,
        encoder: str = "candle",
        C: float = 1.0,
        conformal: bool = True,
        conformal_alpha: float = 0.10,
        calib_frac: float = 0.20,
        random_state: int = 0,
    ) -> None:
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

        self.use_cnn = use_cnn
        self.use_sentiment = use_sentiment
        self.encoder = encoder
        self.conformal = conformal
        self.conformal_alpha = conformal_alpha
        self.calib_frac = calib_frac

        # Underlying point-classifier. ``class_weight=balanced`` mirrors
        # the sentiment baseline so the classes are comparable.
        self._cls = LogisticRegression(
            penalty="l2", C=C, max_iter=500,
            class_weight="balanced", random_state=random_state,
        )

        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._fallback_p1: float = 0.5

        # Inductive-conformal threshold on calibration nonconformity.
        # 1 means "set always contains both classes"; 0 means singleton.
        self._cp_threshold: Optional[float] = None

        # Cached modality features (loaded lazily, keyed on attach_context).
        self._embeddings: Optional[pd.DataFrame] = None
        self._sentiment: Optional[pd.DataFrame] = None
        self._pair: Optional[str] = None
        self._interval: Optional[str] = None

        # Dynamic name reflecting active modalities — surfaced by the runner.
        parts = ["num"]
        if use_cnn:
            parts.append("cnn")
        if use_sentiment:
            parts.append("sent")
        self.name = "fusion_" + "_".join(parts)
        if conformal:
            self.name += "_cp"

    # ------------------------------------------------------------------
    # Context wiring (called by the Phase-5 runner before fit)
    # ------------------------------------------------------------------
    def attach_context(self, pair: str, interval: str) -> None:
        self._pair = pair
        self._interval = interval

    # ------------------------------------------------------------------
    # Modality loaders
    # ------------------------------------------------------------------
    def _load_embeddings(self) -> pd.DataFrame:
        if self._embeddings is not None:
            return self._embeddings
        if not self._pair:
            raise RuntimeError("attach_context not called")
        path = (PROJECT_ROOT / "experiments" / "embeddings"
                / self._pair / self._interval / self.encoder
                / "embeddings.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"CNN embeddings missing: {path}.\n"
                f"Run: python scripts/extract_chart_embeddings.py "
                f"--pairs {self._pair} --interval {self._interval} "
                f"--encoder {self.encoder}"
            )
        emb = pd.read_parquet(path)
        emb["open_time"] = pd.to_datetime(emb["open_time"], utc=True)
        emb = emb.set_index("open_time").sort_index()
        self._embeddings = emb
        return emb

    def _load_sentiment(self) -> pd.DataFrame:
        if self._sentiment is not None:
            return self._sentiment
        if not self._pair:
            raise RuntimeError("attach_context not called")
        cfg = load_config()
        path = (cfg.storage.root_path / "features_sentiment"
                / self._pair / f"{self._interval}.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"sentiment features missing: {path}.\n"
                f"Run: python scripts/build_sentiment_features.py "
                f"--pairs {self._pair} --interval {self._interval}"
            )
        s = pd.read_parquet(path)
        s["open_time"] = pd.to_datetime(s["open_time"], utc=True)
        s = s.set_index("open_time").sort_index()
        self._sentiment = s
        return s

    # ------------------------------------------------------------------
    # Assemble the fused feature matrix aligned to X.index (open_time)
    # ------------------------------------------------------------------
    def _build_matrix(self, X: pd.DataFrame) -> np.ndarray:
        idx = X.index
        blocks: list[np.ndarray] = [X.to_numpy(dtype=np.float64)]

        if self.use_cnn:
            emb = self._load_embeddings()
            cols = [c for c in emb.columns if c.startswith("e")]
            if len(cols) != _EMB_DIM:
                # Older extraction runs may have written a different dim.
                raise ValueError(
                    f"expected {_EMB_DIM} embedding cols, got {len(cols)}"
                )
            sub = emb.reindex(idx)[cols].to_numpy(dtype=np.float64)
            # Bars without an embedding (very early in the series) get
            # zero-vectors — same convention as the sentiment block.
            sub = np.nan_to_num(sub, nan=0.0)
            blocks.append(sub)

        if self.use_sentiment:
            sent = self._load_sentiment()
            sub = sent.reindex(idx)[_SENT_FEATURES].to_numpy(dtype=np.float64)
            sub = np.nan_to_num(sub, nan=0.0)
            blocks.append(sub)

        return np.concatenate(blocks, axis=1)

    # ------------------------------------------------------------------
    # Baseline interface
    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        Z = self._build_matrix(X_train)
        y = y_train.to_numpy(dtype=int)

        # Conformal: chronological hold-out tail of the train slice as
        # calibration set (no shuffling — preserves time order).
        if self.conformal:
            n = len(Z)
            n_cal = max(64, int(n * self.calib_frac))
            n_cal = min(n_cal, n // 2)  # leave plenty for fitting
            n_fit = n - n_cal
        else:
            n_fit = len(Z)
            n_cal = 0

        Z_fit, y_fit = Z[:n_fit], y[:n_fit]

        if len(np.unique(y_fit)) < 2 or n_fit < 32:
            # Degenerate fold — fall back to base rate.
            self._fallback_p1 = float(y_train.mean())
            self._scaler_mean = None
            return

        self._scaler_mean = Z_fit.mean(axis=0)
        self._scaler_std = Z_fit.std(axis=0) + 1e-8
        Z_fit_n = (Z_fit - self._scaler_mean) / self._scaler_std
        self._cls.fit(Z_fit_n, y_fit)

        if self.conformal and n_cal >= 32:
            Z_cal = (Z[n_fit:] - self._scaler_mean) / self._scaler_std
            y_cal = y[n_fit:]
            proba_cal = self._cls.predict_proba(Z_cal)  # (n_cal, 2)
            # Nonconformity = 1 - p_true. Threshold at (1-alpha) quantile
            # with a finite-sample correction (Vovk-style).
            true_probs = proba_cal[np.arange(len(y_cal)), y_cal]
            ncf = 1.0 - true_probs
            q_level = np.ceil((n_cal + 1) * (1 - self.conformal_alpha)) / n_cal
            q_level = min(q_level, 1.0)
            self._cp_threshold = float(np.quantile(ncf, q_level))

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self._scaler_mean is None:
            return np.full(len(X_test), self._fallback_p1, dtype=float)
        Z = self._build_matrix(X_test)
        Zn = (Z - self._scaler_mean) / self._scaler_std
        return self._cls.predict_proba(Zn)[:, 1]

    # ------------------------------------------------------------------
    # Conformal prediction sets (only meaningful if conformal=True)
    # ------------------------------------------------------------------
    def predict_set(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return a (n, 2) boolean array — which class labels are in the set.

        Inductive-conformal classifier (Vovk 2005, Romano et al. 2020).
        ``set[i, k] = True`` iff `1 - p_k(x_i) <= cp_threshold`.
        Coverage guarantee (under exchangeability of calib + test):
            P(y in set) >= 1 - alpha
        """
        if self._cp_threshold is None or self._scaler_mean is None:
            # Without conformal, fall back to singleton at the argmax.
            p1 = self.predict_proba(X_test)
            sets = np.zeros((len(X_test), 2), dtype=bool)
            sets[np.arange(len(X_test)), (p1 >= 0.5).astype(int)] = True
            return sets

        Z = self._build_matrix(X_test)
        Zn = (Z - self._scaler_mean) / self._scaler_std
        proba = self._cls.predict_proba(Zn)  # (n, 2)
        # In-set if its nonconformity is small enough.
        return (1.0 - proba) <= self._cp_threshold
