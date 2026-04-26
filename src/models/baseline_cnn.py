"""CNN baseline wrapper — plugs the chart-CNN into the existing runner.

The runner expects a ``Baseline`` (fit + predict_proba) that consumes
the same X / y the tabular models receive. Chart-CNN doesn't *use* X
directly; instead it's keyed off the underlying OHLCV (re-fetched here).
We accept X just to derive the row alignment with Phase 2, then build a
``ChartImageDataset`` for the actual modelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.dataset import load_ohlcv
from src.models.base import Baseline
from src.vision.dataset import ChartImageDataset
from src.vision.trainer import predict_probabilities, train_cnn_on_indices


class CNNBaseline(Baseline):
    name = "cnn_candle"     # subclassed below for GAF variant

    encoder: str = "candle"
    window: int = 64

    def __init__(
        self,
        pair: str | None = None,
        interval: str = "1h",
        window: int = 64,
        epochs: int = 12,
        batch_size: int = 128,
        lr: float = 1e-3,
        num_workers: int = 0,
    ) -> None:
        # ``pair`` is set by the runner via attach_context() before fit().
        self.pair = pair
        self.interval = interval
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

        self._dataset: ChartImageDataset | None = None
        self._model = None
        self._norm = None
        self._index_to_dataset: dict[pd.Timestamp, int] = {}

    # ------------------------------------------------------------------
    def attach_context(self, pair: str, interval: str) -> None:
        """The runner calls this so we know which (pair, interval) we're on."""
        self.pair = pair
        self.interval = interval

    def _ensure_dataset(self, X_index_sample: pd.DatetimeIndex) -> ChartImageDataset:
        if self._dataset is None:
            assert self.pair is not None, "CNNBaseline.attach_context() not called"
            df = load_ohlcv(self.pair, self.interval)
            self._dataset = ChartImageDataset(
                df, window=self.window, encoder=self.encoder, horizon=1,
            )
            # Map open_time -> dataset row index for alignment with X / y.
            self._index_to_dataset = {
                pd.Timestamp(t): i for i, t in enumerate(self._dataset.open_time)
            }
        return self._dataset

    def _align(self, X: pd.DataFrame) -> np.ndarray:
        """Translate X's open_time index → dataset row indices, dropping any misses."""
        ds_idx = []
        for ts in X.index:
            i = self._index_to_dataset.get(pd.Timestamp(ts))
            if i is not None:
                ds_idx.append(i)
        return np.asarray(ds_idx, dtype=np.int64)

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        ds = self._ensure_dataset(X_train.index)
        train_idx = self._align(X_train)
        if len(train_idx) < self.batch_size:
            self._model = None
            self._fallback_p1 = float(y_train.mean())
            return
        self._model, self._norm = train_cnn_on_indices(
            full_dataset=ds,
            train_idx=train_idx,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            num_workers=self.num_workers,
        )

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            return np.full(len(X_test), getattr(self, "_fallback_p1", 0.5))
        ds = self._dataset
        test_idx = self._align(X_test)
        # Some test rows might not have a matching window (early in dataset).
        if len(test_idx) == 0:
            return np.full(len(X_test), getattr(self, "_fallback_p1", 0.5))
        prob_aligned = predict_probabilities(
            model=self._model, norm=self._norm,
            full_dataset=ds, indices=test_idx,
            batch_size=self.batch_size, num_workers=self.num_workers,
        )
        # Pad rows that didn't have a window with the prior probability.
        out = np.full(len(X_test), getattr(self, "_fallback_p1", 0.5))
        # Map prob_aligned back to positions in X_test.index where alignment succeeded.
        ts_to_pos = {pd.Timestamp(t): i for i, t in enumerate(X_test.index)}
        ds_open_times = ds.open_time
        for i, ds_i in enumerate(test_idx):
            ts = pd.Timestamp(ds_open_times[ds_i])
            pos = ts_to_pos.get(ts)
            if pos is not None:
                out[pos] = prob_aligned[i]
        return out


class CNNCandlestick(CNNBaseline):
    name = "cnn_candle"
    encoder = "candle"


class CNNGafMtf(CNNBaseline):
    name = "cnn_gaf"
    encoder = "gaf"
