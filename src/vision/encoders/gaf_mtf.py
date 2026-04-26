"""Gramian Angular Field + Markov Transition Field encoder.

Output is a 3-channel float32 image (think RGB):
    R = GASF (Gramian Angular Summation Field)
    G = GADF (Gramian Angular Difference Field)
    B = MTF  (Markov Transition Field)

Math summary (Wang & Oates 2015 / Wang 2015):
    1. Min-max normalise the series to [-1, 1].
    2. Take phi = arccos(x). Time becomes the radial coordinate.
    3. GASF[i, j] = cos(phi_i + phi_j)     →  preserves long-range temporal correlations
       GADF[i, j] = sin(phi_i - phi_j)     →  emphasises directional change
    4. MTF[i, j] = transition prob from quantile bin of x_i to quantile bin of x_j.

For our setting the input series is the **closing price** of the window —
the most price-information-dense single channel. We could stack OHLCV
into a multivariate GAF stack later if helpful (Phase 5 fusion).
"""
from __future__ import annotations

import numpy as np

DEFAULT_SIZE = 64       # output spatial resolution = window length
DEFAULT_QUANTILES = 8   # MTF state count


def _minmax(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    """Scale ``x`` to [lo, hi]. Flat series → all zeros."""
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn) * (hi - lo) + lo


def gaf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (GASF, GADF) for a 1D series."""
    xs = np.clip(_minmax(x, -1.0, 1.0), -1.0, 1.0)
    phi = np.arccos(xs)
    # Outer add / sub for vectorised computation.
    gasf = np.cos(phi[:, None] + phi[None, :])
    gadf = np.sin(phi[:, None] - phi[None, :])
    return gasf.astype(np.float32), gadf.astype(np.float32)


def mtf(x: np.ndarray, n_bins: int = DEFAULT_QUANTILES) -> np.ndarray:
    """Markov Transition Field. Returns (T, T) float32."""
    n = len(x)
    # Quantile-based binning (handles non-Gaussian distributions).
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    # np.digitize edges produce bins 1..n_bins. Subtract 1 → 0..n_bins-1.
    bins = np.clip(np.digitize(x, qs[1:-1]), 0, n_bins - 1)

    # Transition matrix W[i, j] = P(state j | state i).
    W = np.zeros((n_bins, n_bins), dtype=np.float64)
    for a, b in zip(bins[:-1], bins[1:]):
        W[a, b] += 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    # MTF[i, j] = W[bin(x_i), bin(x_j)]. Broadcast a (T, T) lookup.
    out = W[bins[:, None], bins[None, :]]
    return out.astype(np.float32)


def render_gaf_mtf(
    close: np.ndarray,
    size: int = DEFAULT_SIZE,
    n_bins: int = DEFAULT_QUANTILES,
) -> np.ndarray:
    """Encode a close-price window as a (3, size, size) float32 image.

    The series is interpolated/decimated to length ``size`` if needed so
    the output is always square at the requested resolution.
    """
    if close.ndim != 1:
        raise ValueError(f"expected 1D close, got shape {close.shape}")
    if len(close) != size:
        # Linear resample. Crude but fine — windows are usually already
        # exactly ``size`` since we pre-pick window length = image size.
        idx_old = np.linspace(0, 1, len(close))
        idx_new = np.linspace(0, 1, size)
        close = np.interp(idx_new, idx_old, close)

    gasf, gadf = gaf(close)
    mtf_img = mtf(close, n_bins=n_bins)

    # Stack as (3, H, W). All channels in [-1, 1] roughly; the CNN's
    # input normalisation in trainer.py handles the per-channel mean/std.
    out = np.stack([gasf, gadf, mtf_img], axis=0).astype(np.float32)
    return out
