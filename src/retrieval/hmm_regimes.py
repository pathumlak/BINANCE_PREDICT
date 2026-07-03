"""Gaussian-HMM market-regime classifier.

Fits a 3-state Gaussian HMM on a small feature vector per bar
(log-return + rolling realised volatility) and labels every bar with a
**permanent integer regime** ∈ {0 = bear, 1 = sideways, 2 = bull}.

Why an HMM?
-----------
* Markov-switching models are a 30-year-old standard in finance for
  characterising regimes (Hamilton 1989, Ang & Bekaert 2002).
* Gaussian emissions on (mean, vol) features are well behaved — the
  three latent states are reliably identified as
  "negative-return / high-vol", "near-zero return / low-vol",
  "positive-return / mid-vol" once you sort them by mean return.
* Inference is a single Viterbi pass — fast enough to score the whole
  history in milliseconds.

Leakage control
---------------
HMM fitting uses the **training window only** (configurable). After
fitting, the *entire* time series is labelled via Viterbi on the
already-seen-past-only features, so the labels themselves never embed
future information at any timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config


# Canonical integer encoding of the three states after sorting by mean
# return. Persisted alongside the labels so downstream code never has to
# guess which state is which.
REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}
REGIME_INV = {v: k for k, v in REGIME_LABELS.items()}


@dataclass
class HMMConfig:
    n_states: int = 3
    vol_window: int = 24            # 1h × 24 = 1 day realised vol
    n_iter: int = 200
    tol: float = 1e-4
    random_state: int = 0
    train_frac: float = 0.5         # fit HMM on the first train_frac of data only


# ---------------------------------------------------------------------------
def _bar_features(close: pd.Series, vol_window: int) -> pd.DataFrame:
    """Two-column feature matrix per bar: log_return, rolling realised vol."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(vol_window, min_periods=vol_window).std()
    feats = pd.concat({"log_return": log_ret, "vol": vol}, axis=1)
    return feats.dropna()


def _sorted_state_map(hmm) -> np.ndarray:
    """Return a permutation that re-labels states by ascending mean return.

    hmmlearn assigns state ids arbitrarily; we sort them so the integer
    label has a stable meaning (0 = bear, n-1 = bull).
    """
    # means_ shape: (n_states, n_features). First feature is log_return.
    means_ret = hmm.means_[:, 0]
    order = np.argsort(means_ret)              # ascending: bear → bull
    perm = np.empty_like(order)
    perm[order] = np.arange(len(order))
    return perm


# ---------------------------------------------------------------------------
def fit_regimes(
    pair: str,
    interval: str = "1h",
    cfg: Optional[HMMConfig] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Fit an HMM on a pair's close series and label every bar.

    Returns a DataFrame indexed by ``open_time`` with one column ``regime``
    (int ∈ {0, 1, 2}). Also persists it to
    ``data/regimes/<pair>/<interval>.parquet`` unless ``out_path`` is set.
    """
    from hmmlearn import hmm  # noqa: PLC0415

    cfg = cfg or HMMConfig()

    # Load OHLCV (use the same loader as Phase 2 so we get a consistent
    # chronological index).
    from src.features.dataset import load_ohlcv  # noqa: PLC0415
    raw = load_ohlcv(pair, interval)
    raw = raw.set_index(pd.DatetimeIndex(raw["open_time"]))

    feats = _bar_features(raw["close"], cfg.vol_window)
    if feats.empty:
        raise RuntimeError(f"no usable rows after warmup for {pair} {interval}")

    # Fit on the chronological training window only.
    n_fit = max(cfg.vol_window * 8, int(len(feats) * cfg.train_frac))
    n_fit = min(n_fit, len(feats))
    X_fit = feats.iloc[:n_fit].to_numpy(dtype=np.float64)

    model = hmm.GaussianHMM(
        n_components=cfg.n_states,
        covariance_type="diag",
        n_iter=cfg.n_iter,
        tol=cfg.tol,
        random_state=cfg.random_state,
    )
    model.fit(X_fit)

    # Decode the whole series.
    states = model.predict(feats.to_numpy(dtype=np.float64))
    perm = _sorted_state_map(model)
    canonical = perm[states]

    out = pd.DataFrame({
        "open_time": feats.index,
        "regime": canonical.astype(np.int16),
    })

    if out_path is None:
        cfg_store = load_config()
        out_path = cfg_store.storage.root_path / "regimes" / pair / f"{interval}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, compression="snappy", index=False)

    return out


# ---------------------------------------------------------------------------
def load_regimes(pair: str, interval: str = "1h") -> pd.DataFrame:
    """Load the persisted regime labels for a (pair, interval)."""
    cfg = load_config()
    path = cfg.storage.root_path / "regimes" / pair / f"{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"regime labels missing: {path}. "
            f"Run: python scripts/fit_hmm_regimes.py "
            f"--pairs {pair} --interval {interval}"
        )
    df = pd.read_parquet(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time").sort_index()


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-state summary: count, share, name."""
    counts = df["regime"].value_counts().sort_index()
    return pd.DataFrame({
        "state": counts.index,
        "name": [REGIME_LABELS.get(int(s), f"state_{s}") for s in counts.index],
        "n_bars": counts.values,
        "share": counts.values / counts.sum(),
    })
