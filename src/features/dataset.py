"""Dataset assembly: load OHLCV from Parquet, build features + labels.

This is the *only* code path models should use to obtain training data.
Centralising it ensures every baseline trains on the same X / y / index,
so any difference in metrics is attributable to the model — not to a
sneaky feature variation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.features.numerical import build_features, feature_columns
from src.features.targets import make_direction_label


@dataclass
class Dataset:
    """Aligned features (X), label (y), and original close (for trading metrics)."""
    pair: str
    interval: str
    X: pd.DataFrame             # engineered features only
    y: pd.Series                # binary direction label
    close: pd.Series            # close at time t (for return calc downstream)
    feature_names: list[str]


def load_ohlcv(pair: str, interval: str, root: Path | None = None) -> pd.DataFrame:
    """Read every monthly Parquet partition for a (pair, interval)."""
    if root is None:
        root = load_config().storage.root_path
    folder = root / "ohlcv" / pair / interval
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No OHLCV partitions under {folder}. Run Phase 1 first."
        )
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


def build_dataset(pair: str, interval: str, horizon: int = 1,
                  root: Path | None = None) -> Dataset:
    """Build the (X, y) tensor for one pair/interval.

    Steps:
      1. Load OHLCV.
      2. Engineer features (lookahead-safe, see ``features.numerical``).
      3. Add direction label for ``horizon`` candles ahead.
      4. Drop the trailing rows where the label is unknown.
      5. Return aligned X, y, and close (for trading-metric back-substitution).
    """
    raw = load_ohlcv(pair, interval, root)
    feats = build_features(raw)

    label = make_direction_label(feats["close"], horizon=horizon)
    feats["__y__"] = label
    feats = feats.dropna(subset=["__y__"])

    feat_names = feature_columns()
    X = feats[feat_names].astype("float64")
    y = feats["__y__"].astype("int64")
    close = feats["close"].astype("float64")

    return Dataset(
        pair=pair,
        interval=interval,
        X=X,
        y=y,
        close=close,
        feature_names=feat_names,
    )
