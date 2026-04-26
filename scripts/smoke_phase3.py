"""Phase 3 smoke test — generate a few images, train CNN briefly, run Grad-CAM.

Should finish in under 2 minutes on CPU. Verifies the whole vision stack
end-to-end before committing to a full multi-fold training run.

    python scripts/smoke_phase3.py
    python scripts/smoke_phase3.py --encoder gaf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.vision.dataset import make_datasets_for_pair  # noqa: E402
from src.vision.encoders.candlestick import render_candlestick  # noqa: E402
from src.vision.encoders.gaf_mtf import render_gaf_mtf  # noqa: E402
from src.vision.gradcam import GradCAM, overlay_on_image  # noqa: E402
from src.vision.trainer import predict_probabilities, train_cnn_on_indices  # noqa: E402

OUT_DIR = PROJECT_ROOT / "experiments" / "smoke_phase3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", choices=["candle", "gaf"], default="candle")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Sample image ----
    logger.info("Building dataset (slicing windows)...")
    ds = make_datasets_for_pair(args.pair, args.interval, encoder=args.encoder)
    logger.info(f"dataset: {len(ds):,} samples")
    img_t, label = ds[0]
    logger.info(f"sample image shape={tuple(img_t.shape)}  label={label}")

    # Save a couple of preview PNGs so you can eyeball them.
    if args.encoder == "candle":
        # uint8 RGB candles
        preview = (img_t.numpy() * 255).astype(np.uint8)
    else:
        # GAF/MTF in [-1, 1] — rescale to [0, 255] for visual sanity
        preview = ((img_t.numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.transpose(preview, (1, 2, 0))).save(
        OUT_DIR / f"sample_{args.encoder}.png"
    )
    logger.success(f"wrote {OUT_DIR / f'sample_{args.encoder}.png'}")

    # ---- 2. Train briefly ----
    n = len(ds)
    train_idx = np.arange(int(n * 0.8))
    test_idx = np.arange(int(n * 0.8), n)
    logger.info(f"training {args.epochs} epoch(s) on {len(train_idx):,} samples...")
    model, norm = train_cnn_on_indices(
        full_dataset=ds, train_idx=train_idx,
        epochs=args.epochs, batch_size=128, num_workers=0,
    )

    # ---- 3. Predict + report accuracy on the held-out tail ----
    prob = predict_probabilities(model, norm, ds, test_idx, batch_size=256, num_workers=0)
    pred = (prob >= 0.5).astype(int)
    truth = np.array([ds.labels[i] for i in test_idx])
    acc = float((pred == truth).mean())
    logger.success(f"smoke acc on held-out 20% : {acc:.3f}")

    # ---- 4. Grad-CAM on one test sample ----
    gcam = GradCAM(model)
    img_one, _ = ds[int(test_idx[len(test_idx) // 2])]
    cam, p_one = gcam(norm(img_one.unsqueeze(0)))
    gcam.close()
    overlay = overlay_on_image(
        ((img_one.numpy() if args.encoder == "candle" else (img_one.numpy() + 1) / 2) * 255)
            .astype(np.uint8),
        cam,
    )
    Image.fromarray(overlay).save(OUT_DIR / f"gradcam_{args.encoder}.png")
    logger.success(f"wrote Grad-CAM overlay → {OUT_DIR / f'gradcam_{args.encoder}.png'}  (P={p_one:.3f})")
    logger.success("smoke OK")


if __name__ == "__main__":
    main()
