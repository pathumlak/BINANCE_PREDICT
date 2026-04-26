"""PatchTST baseline (compact in-house implementation).

PatchTST (Nie et al., 2023) is the "patch-based transformer" that
outperformed many bespoke time-series transformers on standard
forecasting benchmarks. The headline trick is to treat consecutive
time-steps as patches (analogous to ViT for images) so attention
operates on much shorter sequences.

We implement a small classification variant rather than pulling in the
full HuggingFace `PatchTST` model (which is forecasting-oriented and
overkill for binary direction). The behaviour is faithful to the paper:
patch the input, linear-project each patch, add positional embeddings,
push through a transformer encoder, then a classification head over the
mean-pooled tokens.

Same training rituals as the LSTM baseline (per-fold standardisation,
early stopping, sequence windowing).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import Baseline


class _PatchEmbed(nn.Module):
    """Slice a (B, T, F) tensor into non-overlapping patches and linear-project them."""

    def __init__(self, n_features: int, patch_len: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_features * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, n_patches, d_model)
        b, t, f = x.shape
        # Trim T to a multiple of patch_len from the LEFT (preserving the most recent rows).
        rem = t % self.patch_len
        if rem != 0:
            x = x[:, rem:, :]
        n_patches = x.shape[1] // self.patch_len
        x = x.reshape(b, n_patches, self.patch_len * f)
        return self.proj(x)


class _PositionalEmbed(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Sinusoidal — no learnable parameters, generalises across fold sizes.
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, L, d) -> (B, L, d)
        return x + self.pe[: x.size(1)].unsqueeze(0)


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = _PatchEmbed(n_features, patch_len, d_model)
        n_patches = seq_len // patch_len
        self.pos = _PositionalEmbed(n_patches, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):  # (B, T, F)
        z = self.embed(x)            # (B, n_patches, d_model)
        z = self.pos(z)
        z = self.encoder(z)
        z = z.mean(dim=1)            # mean-pool tokens
        return self.head(z).squeeze(-1)


def _windowise(X: np.ndarray, y: np.ndarray, seq_len: int):
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    # sliding_window_view returns a read-only view; copy so PyTorch can wrap it
    # without spamming "non-writable tensor" warnings.
    Xw = np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0)
    Xw = np.ascontiguousarray(Xw.transpose(0, 2, 1))
    yw = np.ascontiguousarray(y[seq_len - 1 :])
    return Xw, yw


class PatchTSTBaseline(Baseline):
    name = "patchtst"

    def __init__(
        self,
        seq_len: int = 64,
        patch_len: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        patience: int = 4,
        device: str | None = None,
        random_state: int = 0,
    ) -> None:
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.model: _PatchTSTNet | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X = X_train.values.astype("float32")
        y = y_train.values.astype("float32")

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        X = (X - self.mean_) / self.std_

        cut = max(self.seq_len + 1, int(len(X) * 0.9))
        X_tr, X_val = X[:cut], X[cut:]
        y_tr, y_val = y[:cut], y[cut:]

        Xw_tr, yw_tr = _windowise(X_tr, y_tr, self.seq_len)
        Xw_val, yw_val = _windowise(X_val, y_val, self.seq_len)

        if len(Xw_tr) == 0:
            self.model = None
            self._fallback_p1 = float(y_train.mean())
            return

        ds_tr = TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr))
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = _PatchTSTNet(
            n_features=X.shape[1],
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
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

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X_test), getattr(self, "_fallback_p1", 0.5))

        X = X_test.values.astype("float32")
        X = (X - self.mean_) / self.std_

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
