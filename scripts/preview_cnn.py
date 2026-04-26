"""Render sample chart images + (optional) Grad-CAM overlays.

Useful for two things:
  1. Sanity-checking the encoders produce something sensible.
  2. Generating figures for your thesis showing what the CNN attends to.

    python scripts/preview_cnn.py --pair BTCUSDT --encoder candle
    python scripts/preview_cnn.py --pair ETHUSDT --encoder gaf --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.features.dataset import load_ohlcv  # noqa: E402
from src.vision.dataset import ChartImageDataset  # noqa: E402


def _scale(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo)) if hi > lo else np.zeros_like(x)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", default="candle", choices=["candle", "gaf"])
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    df = load_ohlcv(args.pair, args.interval)
    ds = ChartImageDataset(df, window=64, encoder=args.encoder, horizon=1)
    out_dir = PROJECT_ROOT / "experiments" / "phase3_previews" / \
              f"{args.pair}_{args.interval}_{args.encoder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(ds) == 0:
        logger.error("Empty dataset — run Phase 1 first.")
        raise SystemExit(1)

    idxs = np.linspace(0, len(ds) - 1, args.n, dtype=int)
    for i, k in enumerate(idxs):
        img, label = ds[int(k)]
        arr = img.numpy()
        per_channel = np.stack([_scale(arr[c]) for c in range(arr.shape[0])])
        rgb = (per_channel.transpose(1, 2, 0) * 255).clip(0, 255).astype("uint8")
        ts = ds.open_time[int(k)]
        fname = f"{i:02d}_{str(ts).replace(':', '-')}_y={int(label)}.png"
        Image.fromarray(rgb).save(out_dir / fname)
    logger.success(f"Wrote {args.n} previews → {out_dir}")
