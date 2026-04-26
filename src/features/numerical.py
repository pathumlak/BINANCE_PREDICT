"""Numerical feature engineering on OHLCV candles.

Every feature here uses ONLY information available at or before time ``t``.
That property is the whole reason this module exists as a separate file —
if you ever add a feature that calls ``shift(-N)`` here, the experiments are
silently broken. Treat that as a tripwire.

Returns a DataFrame indexed by ``open_time`` with dtype float64 throughout.
The first ``warmup_rows`` rows will contain NaNs from rolling windows; the
caller is expected to drop them before training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# How many rows to drop from the head of the feature frame to guarantee no
# NaNs from rolling-window warmup. Conservative — covers all defaults below.
DEFAULT_WARMUP_ROWS = 50


# ---------------------------------------------------------------------------
#  Returns
# ---------------------------------------------------------------------------
def _log_return(s: pd.Series, lag: int = 1) -> pd.Series:
    return np.log(s / s.shift(lag))


def add_return_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)) -> pd.DataFrame:
    """Past log-returns at multiple lags. ``ret_1`` is the return into ``t``."""
    for k in lags:
        df[f"ret_{k}"] = _log_return(df["close"], k)
    return df


# ---------------------------------------------------------------------------
#  Rolling stats
# ---------------------------------------------------------------------------
def add_rolling_stats(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20, 50)) -> pd.DataFrame:
    for w in windows:
        roll = df["close"].rolling(w, min_periods=w)
        df[f"sma_{w}"] = roll.mean()
        df[f"std_{w}"] = roll.std(ddof=0)
        # Position of current close inside the rolling [min, max] band.
        # 0 = at min, 1 = at max. Often more informative than raw min/max.
        rmin = roll.min()
        rmax = roll.max()
        denom = (rmax - rmin).replace(0, np.nan)
        df[f"band_pos_{w}"] = (df["close"] - rmin) / denom
    return df


# ---------------------------------------------------------------------------
#  Technical indicators
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder's smoothing = EMA with alpha = 1/period
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = _rsi(df["close"], 14)
    macd, sig, hist = _macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    df["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
    # Bollinger %B (where close sits relative to 20-period bands).
    sma20 = df["close"].rolling(20, min_periods=20).mean()
    std20 = df["close"].rolling(20, min_periods=20).std(ddof=0)
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["bb_pct_b"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
#  Volume + microstructure
# ---------------------------------------------------------------------------
def add_volume_features(df: pd.DataFrame, windows: tuple[int, ...] = (10, 50)) -> pd.DataFrame:
    for w in windows:
        vol_sma = df["volume"].rolling(w, min_periods=w).mean()
        df[f"vol_ratio_{w}"] = df["volume"] / vol_sma.replace(0, np.nan)
    # Aggressive-buy share — directly available in Binance OHLCV payload.
    df["taker_buy_share"] = df["taker_buy_base"] / df["volume"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
#  Cyclic time features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day and day-of-week as sin/cos pairs (cyclic-safe)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected df indexed by open_time as DatetimeIndex")
    h = df.index.hour
    dow = df.index.dayofweek
    df["hod_sin"] = np.sin(2 * np.pi * h / 24)
    df["hod_cos"] = np.cos(2 * np.pi * h / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = []  # populated lazily on first call


def build_features(ohlcv: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Build the full feature matrix for one (pair, interval) frame.

    ``ohlcv`` should be the canonical OHLCV schema from utils.storage:
    columns include ``open_time, open, high, low, close, volume, ...``.
    """
    df = ohlcv.copy()
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df = df.set_index(pd.DatetimeIndex(df["open_time"]))

    # Keep only the columns we actually need downstream — keeps memory predictable.
    base_cols = ["open", "high", "low", "close", "volume", "taker_buy_base"]
    df = df[base_cols].astype("float64")

    df = add_return_features(df)
    df = add_rolling_stats(df)
    df = add_indicators(df)
    df = add_volume_features(df)
    df = add_time_features(df)

    if drop_warmup:
        df = df.iloc[DEFAULT_WARMUP_ROWS:]

    # Replace any inf created by zero-division with NaN, then drop residual NaNs.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Cache the feature column list (everything that isn't raw OHLCV).
    if not FEATURE_COLUMNS:
        FEATURE_COLUMNS.extend([c for c in df.columns if c not in base_cols])

    return df


def feature_columns() -> list[str]:
    """Return the names of *engineered* columns (excludes raw OHLC)."""
    if not FEATURE_COLUMNS:
        # Build a tiny dummy frame so the cache is populated even before
        # the first real call (some callers need column names up front).
        dummy = pd.DataFrame({
            "open_time": pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC"),
            "open": np.linspace(1, 2, 200),
            "high": np.linspace(1, 2, 200) + 0.1,
            "low": np.linspace(1, 2, 200) - 0.1,
            "close": np.linspace(1, 2, 200),
            "volume": np.linspace(10, 20, 200),
            "taker_buy_base": np.linspace(5, 10, 200),
        })
        build_features(dummy)
    return list(FEATURE_COLUMNS)
