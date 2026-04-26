"""LSTM baseline on raw OHLCV sequence windows.

Standard "is the deep model better than XGB" sanity check. In our
experience XGB usually wins on this kind of tabular-ised time series, but
reviewers expect to see an LSTM in the table.

Important design choices:

* **Input** = the engineered feature matrix (same X as XGBoost) shaped
  into windows of length ``seq_len``. We do NOT feed raw price levels —
  those are non-stationary and an LSTM struggles with them; the engineered
  features are returns / ratios / indicators which are bounded.
* **Standardisation** is fit on the training window only, then applied to
  the test window. Re-fitting per fold is essential — the validation
  distribution must not influence training stats.
* **Early stopping** uses a held-out 10% tail of the training window so
  we never peek at the test fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import Baseline


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]                  # last time-step hidden state
        return self.head(last).squeeze(-1)    # (B,) raw logit


def _windowise(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Slide a length-``seq_len`` window. Label of each window = y at the window's end."""
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    # sliding_window_view returns a read-only view; copy so PyTorch can wrap it
    # without spamming "non-writable tensor" warnings.
    Xw = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    Xw = np.ascontiguousarray(Xw.transpose(0, 2, 1))   # (n, seq_len, F), writable
    yw = np.ascontiguousarray(y[seq_len - 1 :])
    return Xw, yw


class LSTMBaseline(Baseline):
    name = "lstm"

    def __init__(
        self,
        seq_len: int = 64,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 4,
        device: str | None = None,
        random_state: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model: _LSTMNet | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X = X_train.values.astype("float32")
        y = y_train.values.astype("float32")

        # Standardise on TRAINING data only.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X = (X - self.mean_) / self.std_

        # Hold out the last 10% of the training window for early-stopping.
        cut = max(self.seq_len + 1, int(len(X) * 0.9))
        X_tr, X_val = X[:cut], X[cut:]
        y_tr, y_val = y[:cut], y[cut:]

        Xw_tr, yw_tr = _windowise(X_tr, y_tr, self.seq_len)
        Xw_val, yw_val = _windowise(X_val, y_val, self.seq_len)

        if len(Xw_tr) == 0:
            # Window doesn't fit — fall back to a constant predictor.
            self.model = None
            self._fallback_p1 = float(y_train.mean())
            return

        ds_tr = TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = _LSTMNet(
            n_features=X.shape[1], hidden=self.hidden,
            layers=self.layers, dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val = float("inf")
        bad = 0
        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in dl_tr:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            # Validation
            if len(Xw_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    xb = torch.from_numpy(Xw_val).to(self.device)
                    yb = torch.from_numpy(yw_val).to(self.device)
                    val_loss = float(loss_fn(self.model(xb), yb).item())
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

    # ------------------------------------------------------------------
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X_test), getattr(self, "_fallback_p1", 0.5))

        X = X_test.values.astype("float32")
        X = (X - self.mean_) / self.std_

        # The first seq_len-1 test rows can't form a full window. We handle them
        # by left-padding with the last training feature mean (= 0 after standardise).
        # This way we still emit one prediction per test row, aligned 1:1 with y_test.
        pad = np.zeros((self.seq_len - 1, X.shape[1]), dtype="float32")
        X_padded = np.vstack([pad, X])
        Xw, _ = _windowise(X_padded, np.zeros(len(X_padded)), self.seq_len)

        self.model.eval()
        out = np.empty(len(Xw), dtype=float)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(Xw), bs):
                xb = torch.from_numpy(Xw[i:i + bs]).to(self.device)
                logits = self.model(xb).cpu().numpy()
                out[i:i + bs] = 1 / (1 + np.exp(-logits))
        return out
