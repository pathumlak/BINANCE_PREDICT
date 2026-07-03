"""Train the chart-CNN sequentially with Elastic Weight Consolidation.

We split the chronological series into K = 3 equal-size "regime epochs"
and train the CNN on them in time order — *with* and *without* EWC —
to measure how much catastrophic forgetting EWC prevents.

Each phase reserves the last 10 % of its window as a held-out
test set. After every phase we evaluate the *current* model on every
previously-seen phase's held-out set. The average post-final accuracy
across all three test sets is the headline metric.

Why split chronologically?
--------------------------
Phase 6's deliverable is **regime drift handling**. The simplest
faithful simulation is to walk forward in time, treating each chunk as
a fresh task with a (potentially) different latent distribution. The
HMM regime labels are still consumed by the FAISS index and dashboard;
the CNN continual-learning experiment is a separate, controlled
ablation.

Outputs
-------
``experiments/retrieval/cnn_ewc_<mode>/`` where ``mode`` ∈ {ewc, naive}::

    final_model.pt
    metrics.json     # per-phase train loss + post-final test accuracy
                     #   on every earlier phase's held-out set.

Usage
-----
    python scripts/train_cnn_ewc.py --pair BTCUSDT --interval 1h \\
        --n-phases 3 --epochs-per-phase 6 --lambda-ewc 5000

    # Skip the naive baseline (only EWC):
    python scripts/train_cnn_ewc.py --modes ewc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.retrieval.ewc import EWC, evaluate_accuracy, train_one_phase  # noqa: E402
from src.vision.cnn import ChartCNN  # noqa: E402
from src.vision.dataset import make_datasets_for_pair  # noqa: E402
from src.vision.trainer import _Normalise, compute_image_stats  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "experiments" / "retrieval"


def _phase_splits(n: int, n_phases: int, holdout_frac: float):
    """Yield (phase_idx, train_idx, test_idx) for each phase.

    Train indices for phase k are the first ``(1 - holdout_frac)`` of
    phase k's slice; test indices are the last ``holdout_frac``.
    Holdouts from earlier phases are also returned so we can evaluate
    forgetting after each phase.
    """
    edges = np.linspace(0, n, n_phases + 1, dtype=int)
    phases: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_phases):
        lo, hi = edges[k], edges[k + 1]
        cut = lo + int((hi - lo) * (1 - holdout_frac))
        train_idx = np.arange(lo, cut)
        test_idx = np.arange(cut, hi)
        phases.append((train_idx, test_idx))
    return phases


def _fit_normaliser(dataset, train_idx: np.ndarray, batch_size: int,
                    num_workers: int) -> _Normalise:
    from torch.utils.data import Subset  # noqa: PLC0415
    loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    mean, std = compute_image_stats(loader)
    return _Normalise(mean, std)


def _run_one_mode(
    *, mode: str, pair: str, interval: str, encoder: str,
    n_phases: int, epochs_per_phase: int, lambda_ewc: float,
    batch_size: int, num_workers: int, holdout_frac: float,
) -> dict:
    """Sequential training across phases under one mode (ewc or naive)."""
    assert mode in ("ewc", "naive")
    logger.info(f"=== train_cnn_ewc  mode={mode}  pair={pair}  phases={n_phases} ===")
    t0 = time.time()

    ds = make_datasets_for_pair(pair, interval, encoder=encoder)
    n = len(ds)
    if n < n_phases * 200:
        raise RuntimeError(
            f"dataset too small for {n_phases} phases ({n} rows). "
            f"Pick a finer interval or fewer phases."
        )

    phases = _phase_splits(n, n_phases, holdout_frac)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChartCNN().to(device)

    # Per-phase normaliser is fit ONLY on phase 0's training window so
    # the input statistics stay stable across phases (same trick as the
    # Phase 2 baselines). If you re-fit per phase you mix the
    # distribution-shift with the forgetting signal.
    norm = _fit_normaliser(ds, phases[0][0], batch_size, num_workers).to(device)

    regulariser = EWC() if mode == "ewc" else None
    phase_logs: list[dict] = []

    for k, (tr_idx, te_idx) in enumerate(phases):
        logger.info(
            f"phase {k}:  train={len(tr_idx):,}  holdout={len(te_idx):,}"
        )
        train_metrics = train_one_phase(
            model=model, norm=norm, full_dataset=ds, train_idx=tr_idx,
            ewc=regulariser, lambda_ewc=lambda_ewc,
            epochs=epochs_per_phase, batch_size=batch_size,
            num_workers=num_workers, device=device,
        )

        if regulariser is not None:
            # Absorb the just-finished phase into EWC's memory.
            sample = np.random.RandomState(0).choice(
                tr_idx, size=min(2048, len(tr_idx)), replace=False,
            )
            regulariser.absorb(model, ds, sample.tolist(), device, norm)

        # Evaluate on every previously-seen phase's holdout.
        per_phase_acc: list[float] = []
        for j in range(k + 1):
            acc = evaluate_accuracy(
                model, norm, ds, phases[j][1],
                batch_size=batch_size, num_workers=num_workers, device=device,
            )
            per_phase_acc.append(acc)
            logger.info(f"  after phase {k} → test on phase {j}: acc={acc:.4f}")

        phase_logs.append({
            "phase": k,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            **train_metrics,
            "post_phase_test_accuracies": per_phase_acc,
        })

    elapsed = time.time() - t0

    # Final headline: average accuracy at end of training across all
    # phase holdouts. Bigger = better preservation.
    final_accs = phase_logs[-1]["post_phase_test_accuracies"]
    avg_final = float(np.mean(final_accs))

    out_dir = OUT_ROOT / f"cnn_ewc_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "norm_mean": norm.mean.detach().cpu(),
                "norm_std": norm.std.detach().cpu()},
               out_dir / "final_model.pt")
    metrics = {
        "mode": mode, "pair": pair, "interval": interval,
        "encoder": encoder, "n_phases": n_phases,
        "epochs_per_phase": epochs_per_phase,
        "lambda_ewc": lambda_ewc if mode == "ewc" else 0.0,
        "avg_final_accuracy": avg_final,
        "elapsed_seconds": round(elapsed, 2),
        "phases": phase_logs,
    }
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.success(
        f"{mode}: avg_final_acc={avg_final:.4f}   time={elapsed:.1f}s   "
        f"-> {out_dir}"
    )
    return metrics


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pair", default="BTCUSDT")
    p.add_argument("--interval", default="1h")
    p.add_argument("--encoder", default="candle", choices=["candle", "gaf"])
    p.add_argument("--n-phases", type=int, default=3)
    p.add_argument("--epochs-per-phase", type=int, default=6)
    p.add_argument("--lambda-ewc", type=float, default=5000.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--modes", nargs="+", default=["naive", "ewc"],
                   choices=["naive", "ewc"])
    args = p.parse_args()

    results: dict[str, dict] = {}
    for mode in args.modes:
        results[mode] = _run_one_mode(
            mode=mode, pair=args.pair, interval=args.interval,
            encoder=args.encoder, n_phases=args.n_phases,
            epochs_per_phase=args.epochs_per_phase,
            lambda_ewc=args.lambda_ewc, batch_size=args.batch_size,
            num_workers=args.num_workers, holdout_frac=args.holdout_frac,
        )

    if len(results) >= 2:
        a = results.get("naive", {}).get("avg_final_accuracy")
        b = results.get("ewc", {}).get("avg_final_accuracy")
        if a is not None and b is not None:
            print(
                f"\nForgetting comparison:\n"
                f"  naive sequential: avg final acc = {a:.4f}\n"
                f"  EWC sequential  : avg final acc = {b:.4f}\n"
                f"  Δ (EWC − naive) : {b - a:+.4f}"
            )
            (OUT_ROOT / "cnn_ewc_comparison.json").write_text(
                json.dumps({"naive": a, "ewc": b, "delta": b - a}, indent=2)
            )
