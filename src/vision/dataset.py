"""PyTorch Dataset for chart-CNN training.

Slides a length-``window`` view over the (O, H, L, C) array, encodes each
window as an image via the chosen encoder, and emits ``(image, label)``
tensors. Labels are the *same* binary direction targets used by Phase 2's
baselines so the CNN's metrics line up with theirs in the DM matrix.

Performance notes (CPU training):
    * Windows are computed with stride tricks → no per-call slicing cost.
    * Images are encoded lazily in ``__getitem__``. Pre-caching all of
      them blew up to ~10 GB across 5 pairs × 2 encoders, so we trade
      a bit of CPU per epoch for tractable memory.
    * Use ``num_workers >= 2`` in the DataLoader to pipeline encoding.
"""
from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features.dataset import load_ohlcv
from src.features.targets import make_direction_label
from src.vision.encoders.candlestick import DEFAULT_HW, render_candlestick
from src.vision.encoders.gaf_mtf import DEFAULT_SIZE, render_gaf_mtf

EncoderName = Literal["candle", "gaf"]


def _build_windows(close: np.ndarray, ohlc: np.ndarray, window: int):
    """Return (windows_close[N, T], windows_ohlc[N, T, 4]) views — no copy."""
    n = len(close) - window + 1
    if n <= 0:
        return (
            np.empty((0, window), dtype=close.dtype),
            np.empty((0, window, 4), dtype=ohlc.dtype),
        )
    sw_c = np.lib.stride_tricks.sliding_window_view(close, window)
    sw_ohlc = np.lib.stride_tricks.sliding_window_view(ohlc, window, axis=0)
    sw_ohlc = sw_ohlc.transpose(0, 2, 1)  # (N, T, 4)
    return sw_c, sw_ohlc


class ChartImageDataset(Dataset):
    """One sample = (image_tensor, direction_label). End-of-window indexed."""

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        window: int = 64,
        encoder: EncoderName = "candle",
        horizon: int = 1,
        image_size: int | None = None,
    ) -> None:
        if "close" not in ohlcv_df.columns:
            raise ValueError("ohlcv_df must contain at least open/high/low/close")

        self.window = window
        self.encoder_name = encoder
        self.image_size = image_size or (DEFAULT_HW if encoder == "candle" else DEFAULT_SIZE)

        # Compute label first so we can drop the rows with unknown future.
        df = ohlcv_df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        df["__y__"] = make_direction_label(df["close"], horizon=horizon)
        df = df.dropna(subset=["__y__"]).reset_index(drop=True)

        close = df["close"].to_numpy(dtype=np.float64)
        ohlc = df[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
        labels = df["__y__"].to_numpy(dtype=np.int64)
        open_time = df["open_time"].to_numpy()

        # Slide windows. A window covers indices [i, i+window-1]; its label
        # is labels[i + window - 1] (= y of the last bar inside the window).
        win_close, win_ohlc = _build_windows(close, ohlc, window)
        end_idx = np.arange(window - 1, window - 1 + len(win_close))

        self.win_close = win_close
        self.win_ohlc = win_ohlc
        self.labels = labels[end_idx]
        self.open_time = open_time[end_idx]

        # Pick encoder once.
        if encoder == "candle":
            self._encode: Callable[[int], np.ndarray] = self._encode_candle
        elif encoder == "gaf":
            self._encode = self._encode_gaf
        else:
            raise ValueError(encoder)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self._encode(idx)
        return torch.from_numpy(img), int(self.labels[idx])

    # ------------------------------------------------------------------
    def _encode_candle(self, idx: int) -> np.ndarray:
        img = render_candlestick(self.win_ohlc[idx], hw=self.image_size)
        # uint8 → float32 in [0,1] with channel-first layout already done.
        return (img.astype(np.float32) / 255.0)

    def _encode_gaf(self, idx: int) -> np.ndarray:
        return render_gaf_mtf(self.win_close[idx], size=self.image_size)


def make_datasets_for_pair(
    pair: str,
    interval: str,
    window: int = 64,
    encoder: EncoderName = "candle",
    horizon: int = 1,
):
    """Convenience: load a pair's OHLCV from Phase 1's parquet store and
    wrap it as a single ChartImageDataset.

    Splits / fold logic live in the trainer, not here, so this returns one
    big dataset.
    """
    df = load_ohlcv(pair, interval)
    return ChartImageDataset(
        df,
        window=window,
        encoder=encoder,
        horizon=horizon,
    )
