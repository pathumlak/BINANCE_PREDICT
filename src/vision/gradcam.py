"""Grad-CAM for the chart-CNN.

We hook the last conv layer of the backbone, capture activations and
gradients during a forward+backward pass for one sample, then build the
class-activation map (CAM):

    CAM(x, y) = ReLU( sum_k alpha_k * A_k(x, y) )
    alpha_k   = global average of d(score) / d(A_k)

The result is upsampled to image resolution and returned as a normalised
[0, 1] heatmap. ``overlay_on_image`` blends it on top of an RGB image
for the dashboard / paper figures.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

from src.vision.cnn import ChartCNN


class GradCAM:
    """Lightweight Grad-CAM attached to the last conv block of ChartCNN."""

    def __init__(self, model: ChartCNN, target_layer: nn.Module | None = None) -> None:
        self.model = model
        # Default = the last _ConvBlock in the backbone (8×8 feature map).
        if target_layer is None:
            target_layer = model.backbone[-1]
        self.target = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = self.target.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        # grad_out is a tuple; first element is dLoss/dout
        self._gradients = grad_out[0].detach()

    # ------------------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Return (heatmap_HxW in [0,1], probability of class 1)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)        # (1,)
        prob = torch.sigmoid(logits).item()
        # Backprop the score for class 1 (= raw logit, before sigmoid).
        logits.sum().backward()

        a = self._activations[0]      # (C, h, w)
        g = self._gradients[0]        # (C, h, w)
        weights = g.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * a).sum(dim=0))  # (h, w)

        # Min-max normalise.
        cam = cam.cpu().numpy().astype(np.float32)
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample to (H, W) of the input.
        H, W = x.shape[-2:]
        # Nearest-neighbour for crispness on tiny 8×8 feature maps.
        cam_full = np.kron(cam, np.ones((H // cam.shape[0], W // cam.shape[1]), dtype=np.float32))
        return cam_full, prob

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_on_image(image_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap onto an RGB image. Inputs both in (H, W, 3) / (H, W)."""
    if image_uint8.shape[0] == 3:                       # (3, H, W) → (H, W, 3)
        image_uint8 = np.transpose(image_uint8, (1, 2, 0))
    # Simple red-tinted heatmap → (H, W, 3) uint8.
    heat = np.zeros_like(image_uint8)
    heat[..., 0] = (cam * 255).astype(np.uint8)         # red channel = strength
    blended = ((1 - alpha) * image_uint8 + alpha * heat).clip(0, 255).astype(np.uint8)
    return blended
