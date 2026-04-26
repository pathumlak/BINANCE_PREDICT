"""Extract chart-CNN embeddings for the entire historical dataset.

Used by Phase 6 (FAISS pattern-matching engine) — every candle gets a
128-dim embedding from the trained CNN, then we build a similarity index.

Usage:
    # Default: candle encoder, last fold's model, all configured pairs.
    python scripts/extract_chart_embeddings.py

    # Use the GAF/MTF variant
    python scripts/extract_chart_embeddings.py --encoder gaf

The trained CNN is *retrained on the full series* here (no held-out test)
because the embeddings are an unsupervised similarity feature — the
classifier head is discarded; only the embedding head matters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.vision.dataset import make_datasets_for_pair  # noqa: E402
from src.vision.trainer import train_cnn_on_indices  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "experiments" / "embeddings"


def extract_for_pair(pair: str, interval: str, encoder: str,
                     epochs: int, batch_size: int, num_workers: int) -> int:
    logger.info(f"=== embeddings  /  {pair} {interval}  /  {encoder} ===")
    ds = make_datasets_for_pair(pair, interval, encoder=encoder)
    if len(ds) == 0:
        logger.warning("empty dataset, skipping")
        return 0

    # Train on the whole series with a 10% validation tail (early-stop only).
    full_idx = np.arange(len(ds))
    model, norm = train_cnn_on_indices(
        full_dataset=ds,
        train_idx=full_idx,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Inference: embed every sample.
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    embs = np.empty((len(ds), 128), dtype=np.float32)
    model.eval()
    pos = 0
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(device, non_blocking=True)
            e = model.embed(norm(img)).cpu().numpy().astype(np.float32)
            embs[pos:pos + len(e)] = e
            pos += len(e)

    out_dir = OUT_ROOT / pair / interval / encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(embs, columns=[f"e{i}" for i in range(128)])
    df.insert(0, "open_time", ds.open_time)
    df.to_parquet(out_dir / "embeddings.parquet", compression="snappy", index=False)
    logger.success(f"wrote {out_dir / 'embeddings.parquet'}  ({len(df):,} rows)")
    return len(df)


if __name__ == "__main__":
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=cfg.binance.pairs)
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", choices=["candle", "gaf"], default="candle")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    total = 0
    for pair in args.pairs:
        try:
            total += extract_for_pair(
                pair=pair, interval=args.interval, encoder=args.encoder,
                epochs=args.epochs, batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"FAILED {pair}: {e}")
    logger.success(f"done — {total:,} total embeddings written")
