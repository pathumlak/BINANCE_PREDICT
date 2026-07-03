"""Elastic Weight Consolidation for the chart-CNN.

EWC (Kirkpatrick et al. 2017) attaches a quadratic anchor to each
parameter after training on task *t*:

    L_total(θ) = L_task(θ) + Σ_t (λ_t / 2) · Σ_i F_t,i · (θ_i − θ*_{t,i})²

where F_t is the diagonal of the Fisher information matrix computed at
θ*_t.  The Fisher coordinates that mattered for previous tasks get
pinned; coordinates with near-zero Fisher are free to move on the new
task.  This prevents catastrophic forgetting under non-stationary data —
exactly what crypto market regime drift looks like.

Why diagonal Fisher?
--------------------
Full Fisher is intractable (≈ 250K params squared); the diagonal
approximation is the canonical EWC simplification and works fine when
parameters are roughly independent.

Usage shape (consumed by ``scripts/train_cnn_ewc.py``)::

    reg = EWC()
    # ... train CNN on epoch 0 (no penalty) ...
    reg.absorb(model, dataset, sample_idx, device, norm)

    # ... train CNN on epoch 1 with penalty ...
    penalty = reg.penalty(model)
    loss = task_loss + lambda_ewc * penalty
    # ... after epoch 1 ...
    reg.absorb(model, dataset, sample_idx, device, norm)

    # ... etc.

After every call to ``absorb`` the Fisher accumulates across tasks
(Kirkpatrick §4: "we add the Fisher information matrices"), so a single
penalty term defends against forgetting on *every* previous task at
once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


@dataclass
class _Anchor:
    """One per-task anchor: frozen parameter snapshot + Fisher diagonal."""
    params: dict[str, torch.Tensor] = field(default_factory=dict)
    fisher: dict[str, torch.Tensor] = field(default_factory=dict)


class EWC:
    """Accumulating EWC regulariser across an arbitrary number of tasks."""

    def __init__(self) -> None:
        self._anchors: list[_Anchor] = []

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _snapshot_params(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters()
                if p.requires_grad}

    # ------------------------------------------------------------------
    def absorb(
        self,
        model: nn.Module,
        dataset,
        sample_idx: Iterable[int],
        device: torch.device | str,
        norm: nn.Module,
        batch_size: int = 64,
        max_batches: int = 64,
    ) -> None:
        """Compute Fisher diag on ``dataset[sample_idx]`` and store an anchor.

        The Fisher diagonal is estimated as

            F_i = E_x[ (∂ log p(y | x; θ) / ∂ θ_i)² ]

        Here we use the **empirical** Fisher (gradients of the actual
        labels) since true Fisher needs sampling y ~ p(y|x;θ) which is
        more expensive and gives almost identical results in practice
        (Pascanu & Bengio 2014).
        """
        device = torch.device(device)
        model.eval()
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

        sample_idx = list(sample_idx)
        loader = DataLoader(
            Subset(dataset, sample_idx),
            batch_size=batch_size, shuffle=True,
        )

        # Accumulate squared gradients across the sample.
        sq_grads: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        n_examples = 0
        for b_idx, (img, y) in enumerate(loader):
            if b_idx >= max_batches:
                break
            img = img.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            logits = model(norm(img))
            loss = loss_fn(logits, y)        # sum-reduction => per-example grads sum
            loss.backward()

            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                sq_grads[n] += p.grad.detach() ** 2
            n_examples += y.size(0)

        # Diagonal Fisher = mean of squared gradients (per-example).
        fisher = {n: (sq / max(n_examples, 1)) for n, sq in sq_grads.items()}
        params = self._snapshot_params(model)
        self._anchors.append(_Anchor(params=params, fisher=fisher))
        model.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return the scalar EWC penalty across all stored anchors."""
        if not self._anchors:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            for anchor in self._anchors:
                if n not in anchor.params:
                    continue
                F = anchor.fisher[n]
                t = anchor.params[n]
                total = total + (F * (p - t) ** 2).sum()
        return 0.5 * total

    # ------------------------------------------------------------------
    @property
    def n_tasks(self) -> int:
        return len(self._anchors)


# ---------------------------------------------------------------------------
def train_one_phase(
    model: nn.Module,
    norm: nn.Module,
    full_dataset,
    train_idx: np.ndarray,
    *,
    ewc: Optional[EWC] = None,
    lambda_ewc: float = 5_000.0,
    epochs: int = 6,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.1,
    patience: int = 3,
    num_workers: int = 2,
    device: str | None = None,
) -> dict[str, float]:
    """Train `model` on a chronological slice with an optional EWC penalty.

    Returns a metrics dict for logging (train_loss, val_loss, n_epochs_run).
    The model is updated **in place** — call ``ewc.absorb(...)`` afterward
    if this phase should be remembered.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)
    model.to(device_t)

    cut = max(1, int(len(train_idx) * (1 - val_frac)))
    tr_idx = train_idx[:cut]
    val_idx = train_idx[cut:]

    tr_loader = DataLoader(
        Subset(full_dataset, tr_idx.tolist()),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val, bad = float("inf"), 0
    best_state = None
    epochs_run = 0
    last_train_loss = float("nan")

    for ep in range(epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for img, y in tr_loader:
            img = img.to(device_t, non_blocking=True)
            y = y.float().to(device_t, non_blocking=True)
            opt.zero_grad()
            logits = model(norm(img))
            loss = loss_fn(logits, y)
            if ewc is not None and ewc.n_tasks > 0:
                loss = loss + lambda_ewc * ewc.penalty(model)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * y.size(0)
            n += y.size(0)
        last_train_loss = loss_sum / max(n, 1)
        epochs_run += 1

        if len(val_idx) > 0:
            model.eval()
            v_loss, vn = 0.0, 0
            with torch.no_grad():
                for img, y in val_loader:
                    img = img.to(device_t, non_blocking=True)
                    y = y.float().to(device_t, non_blocking=True)
                    logits = model(norm(img))
                    v_loss += float(loss_fn(logits, y)) * y.size(0)
                    vn += y.size(0)
            val = v_loss / max(vn, 1)
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

    return {
        "train_loss": last_train_loss,
        "val_loss": float(best_val) if best_val != float("inf") else float("nan"),
        "epochs_run": epochs_run,
    }


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    norm: nn.Module,
    full_dataset,
    indices: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 2,
    device: str | None = None,
) -> float:
    """Direction accuracy of the CNN on a chronological slice."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(device)
    model.eval().to(device_t)
    loader = DataLoader(
        Subset(full_dataset, indices.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    correct, n = 0, 0
    for img, y in loader:
        img = img.to(device_t, non_blocking=True)
        y = y.long().to(device_t, non_blocking=True)
        logits = model(norm(img))
        pred = (logits >= 0).long()
        correct += int((pred == y).sum().item())
        n += y.size(0)
    return correct / max(n, 1)
