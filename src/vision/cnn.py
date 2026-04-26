"""Compact CNN for 64×64 chart images.

Architecture (small enough for CPU training in a few minutes per fold):

    Conv(3 → 32, 3x3) → BN → ReLU → MaxPool 2x2     # 64 → 32
    Conv(32 → 64, 3x3) → BN → ReLU → MaxPool 2x2    # 32 → 16
    Conv(64 → 128, 3x3) → BN → ReLU → MaxPool 2x2   # 16 → 8
    Conv(128 → 128, 3x3) → BN → ReLU → AdaptiveAvgPool(1)  # 128
    └──> embedding head (Linear 128 → emb_dim) — reused in Phase 6
    └──> classification head (Linear emb_dim → 1) — binary direction

Total ~250K trainable params. The same architecture serves both
encodings (candlestick PNG and GAF/MTF stack) — the only difference is
input statistics, handled by ``input_normalise``.
"""
from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class ChartCNN(nn.Module):
    """Tiny CNN with separate embedding and classification heads."""

    def __init__(
        self,
        in_channels: int = 3,
        emb_dim: int = 128,
        n_classes: int = 1,         # binary → 1 logit (BCEWithLogits)
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _ConvBlock(in_channels, 32, pool=True),    # 32×32
            _ConvBlock(32, 64, pool=True),             # 16×16
            _ConvBlock(64, 128, pool=True),            # 8×8
            _ConvBlock(128, 128, pool=False),          # 8×8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)             # → (B, 128, 1, 1)

        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(emb_dim, n_classes)

    # ------------------------------------------------------------------
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the post-GAP feature map (B, 128, 1, 1). Used by Grad-CAM."""
        return self.gap(self.backbone(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 128-dim embedding (B, emb_dim) — used in Phase 6."""
        return self.embed_head(self.features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B,) — caller applies sigmoid + threshold."""
        return self.classifier(self.embed(x)).squeeze(-1)
