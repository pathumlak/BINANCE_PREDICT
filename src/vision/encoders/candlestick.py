"""Candlestick image encoder.

Input  : OHLC window of shape (T, 4) where columns are [open, high, low, close].
Output : RGB image of shape (3, H, W) with dtype uint8 (PyTorch-friendly).

We render with raw numpy (no matplotlib / mplfinance) because we'll be
generating tens of thousands of images per training run and matplotlib's
overhead is brutal in tight loops.

Visual conventions:
    * Up candle (close >= open): green body, green wick.
    * Down candle              : red   body, red   wick.
    * Background = white.
    * Each candle gets ``W // T`` columns of pixels. Body fills the
      central ~70%; wicks are 1px wide centred in the candle column.

These choices match what a human trader would scan, which keeps Grad-CAM
overlays in Phase 7 interpretable.
"""
from __future__ import annotations

import numpy as np

# Image colours (R, G, B)
COLOR_BG = np.array([255, 255, 255], dtype=np.uint8)
COLOR_UP = np.array([34, 139, 34],   dtype=np.uint8)   # forest green
COLOR_DN = np.array([200, 30, 30],   dtype=np.uint8)   # red

DEFAULT_HW = 64           # 64×64 — plenty of resolution at this image size
DEFAULT_BODY_FRAC = 0.7   # body width as fraction of candle column


def render_candlestick(
    ohlc: np.ndarray,
    hw: int = DEFAULT_HW,
    body_frac: float = DEFAULT_BODY_FRAC,
) -> np.ndarray:
    """Render one OHLC window as an (3, hw, hw) uint8 array.

    Parameters
    ----------
    ohlc : np.ndarray of shape (T, 4)
        Columns = open, high, low, close (any positive scale).
    hw : int
        Image side length. Must satisfy ``hw >= T``.
    body_frac : float
        Fraction of each candle's column taken up by the body.
    """
    if ohlc.ndim != 2 or ohlc.shape[1] != 4:
        raise ValueError(f"expected (T, 4) ohlc, got {ohlc.shape}")
    T = ohlc.shape[0]
    if T > hw:
        raise ValueError(f"window length {T} exceeds image width {hw}")

    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    # ---- vertical scaling: map [global_low, global_high] → [hw-1, 0] ----
    g_lo = float(np.min(l))
    g_hi = float(np.max(h))
    if g_hi <= g_lo:
        g_hi = g_lo + 1e-6  # degenerate flat window — avoid /0

    def y(price: np.ndarray) -> np.ndarray:
        # Higher price → smaller row index (image origin top-left).
        return ((g_hi - price) / (g_hi - g_lo) * (hw - 1)).round().astype(int)

    y_o = y(o)
    y_h = y(h)
    y_l = y(l)
    y_c = y(c)

    # ---- horizontal layout: equal-width candle columns ----
    col_w = hw / T
    body_half = max(1, int(round(col_w * body_frac / 2)))

    img = np.broadcast_to(COLOR_BG, (hw, hw, 3)).copy()

    for i in range(T):
        cx = int((i + 0.5) * col_w)
        body_lo = max(0, cx - body_half)
        body_hi = min(hw, cx + body_half + 1)

        up = c[i] >= o[i]
        col = COLOR_UP if up else COLOR_DN

        # Wick (1 px wide) — full high→low range
        img[y_h[i]:y_l[i] + 1, cx, :] = col

        # Body (open→close range, at least 1 px tall)
        top = min(y_o[i], y_c[i])
        bot = max(y_o[i], y_c[i])
        if bot == top:
            bot = min(hw - 1, top + 1)
        img[top:bot + 1, body_lo:body_hi, :] = col

    # Convert (H, W, 3) → (3, H, W) for PyTorch
    return np.transpose(img, (2, 0, 1)).copy()
