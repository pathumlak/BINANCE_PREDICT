"""CNN trainer + per-channel input normaliser.

Training rituals match the LSTM / PatchTST baselines so the comparison
in the DM matrix is apples-to-apples:

    * Standardise inputs using TRAINING-window stats only.
    * Hold out the last 10% of training as a validation tail for early stopping.
    * AdamW, weight decay, BCEWithLogits.
    * Predict probability per row (caller threshold = 0.5).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.vision.cnn import ChartCNN


def compute_image_stats(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over a DataLoader (online accumulation)."""
    n = 0
    s = torch.zeros(3, dtype=torch.float64)
    s2 = torch.zeros(3, dtype=torch.float64)
    for img, _ in loader:
        # img : (B, 3, H, W)
        b = img.size(0)
        flat = img.reshape(b, 3, -1).double()
        s += flat.sum(dim=(0, 2))
        s2 += (flat ** 2).sum(dim=(0, 2))
        n += flat.size(0) * flat.size(2)
    mean = (s / n).float()
    var = (s2 / n).float() - mean ** 2
    std = var.clamp_min(1e-6).sqrt()
    return mean, std


class _Normalise(nn.Module):
    """Apply (x - mean) / std per channel. Buffers move with .to(device)."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def train_cnn_on_indices(
    full_dataset,
    train_idx: np.ndarray,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    val_frac: float = 0.1,
    num_workers: int = 2,
    device: str | None = None,
    random_state: int = 0,
) -> tuple[nn.Module, _Normalise]:
    """Train a ChartCNN on a chronological train slice.

    The last ``val_frac`` of ``train_idx`` is reserved for early-stopping
    validation — never the test fold.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Chronological train/val split inside the train window.
    cut = max(1, int(len(train_idx) * (1 - val_frac)))
    tr_idx = train_idx[:cut]
    val_idx = train_idx[cut:]

    tr_ds = Subset(full_dataset, tr_idx.tolist())
    val_ds = Subset(full_dataset, val_idx.tolist())

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=num_workers > 0)

    # Per-channel normalisation fit on training data ONLY.
    stats_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    mean, std = compute_image_stats(stats_loader)
    norm = _Normalise(mean, std).to(device)

    model = ChartCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        for img, y in tr_loader:
            img = img.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(norm(img))
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        # ---- validation ----
        if len(val_idx) > 0:
            model.eval()
            v_loss = 0.0
            v_n = 0
            with torch.no_grad():
                for img, y in val_loader:
                    img = img.to(device, non_blocking=True)
                    y = y.float().to(device, non_blocking=True)
                    logits = model(norm(img))
                    v_loss += float(loss_fn(logits, y).item()) * y.size(0)
                    v_n += y.size(0)
            val = v_loss / max(1, v_n)
            if val < best_val - 1e-4:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm


def predict_probabilities(
    model: nn.Module,
    norm: _Normalise,
    full_dataset,
    indices: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str | None = None,
) -> np.ndarray:
    """Run trained model over the listed indices, return P(y=1)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub = Subset(full_dataset, indices.tolist())
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    out = np.empty(len(indices), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        i = 0
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
            logits = model(norm(img)).cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits))
            out[i:i + len(prob)] = prob
            i += len(prob)
    return out
