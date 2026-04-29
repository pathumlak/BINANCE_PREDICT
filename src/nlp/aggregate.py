"""Aggregate scored news articles into per-1h-bar sentiment features.

Inputs
------
* ``news`` — concatenated DataFrame of every scored article. Must carry
  ``published_at``, ``tickers`` (list[str]), and ``sentiment`` (JSON
  string produced by :mod:`src.nlp.scoring`).
* ``bars`` — DataFrame indexed by ``open_time`` (UTC), one row per OHLCV
  bar to label.
* ``pair`` — e.g. ``"BTCUSDT"``. Used to filter news to articles that
  mention the underlying ticker (BTC, ETH, …). Pass ``"ALL"`` to include
  every article regardless of ticker (useful for macro signal).

Output
------
DataFrame indexed identically to ``bars``, with columns:

    sent_count      — articles in the lookback window
    sent_mean       — mean ensemble score
    sent_std        — std of ensemble scores (volatility of opinion)
    sent_min        — min ensemble score (most-bearish article)
    sent_max        — max ensemble score (most-bullish article)
    sent_last       — score of the most-recent article in window
    sent_conf_mean  — average per-article confidence

All features at bar ``t`` use ONLY articles with
``published_at`` ∈ ``[t - LOOKBACK, t)`` — no future leak.
Bars with no articles in the window get zeros for the mean/min/max/last
and zero count, which is then trivially handled by the downstream model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd

# Map a Binance pair symbol to the ticker codes that should "count" for
# news matching.
_PAIR_TICKERS: dict[str, set[str]] = {
    "BTCUSDT": {"BTC"},
    "ETHUSDT": {"ETH"},
    "BNBUSDT": {"BNB"},
    "SOLUSDT": {"SOL"},
    "XRPUSDT": {"XRP"},
}


@dataclass
class AggregateConfig:
    lookback: timedelta = timedelta(hours=24)   # rolling news window per bar
    include_macro: bool = True                  # also include pair="ALL" articles


# ---------------------------------------------------------------------------
def _decode_sentiment(s) -> dict | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _filter_for_pair(news: pd.DataFrame, pair: str, include_macro: bool) -> pd.DataFrame:
    if pair == "ALL":
        return news
    tickers = _PAIR_TICKERS.get(pair, set())
    if not tickers:
        return news.iloc[0:0]

    def keep(row_tickers) -> bool:
        if not isinstance(row_tickers, (list, tuple)):
            return include_macro and (row_tickers is None or len(row_tickers) == 0)
        if not row_tickers:
            return include_macro                 # macro article (no ticker)
        return any(t in tickers for t in row_tickers)

    mask = news["tickers"].apply(keep)
    return news.loc[mask]


def aggregate_for_bars(
    news: pd.DataFrame,
    bars: pd.DataFrame,
    pair: str,
    cfg: AggregateConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or AggregateConfig()
    if "open_time" not in bars.columns:
        raise ValueError("bars must contain an `open_time` column")

    # 1. filter & decode sentiment
    n = _filter_for_pair(news, pair, include_macro=cfg.include_macro).copy()
    if n.empty:
        return _empty(bars)

    n["s_dict"] = n["sentiment"].apply(_decode_sentiment)
    n = n.dropna(subset=["s_dict"])
    if n.empty:
        return _empty(bars)

    n["score"] = n["s_dict"].apply(lambda d: float(d.get("ensemble", 0.0)))
    n["conf"] = n["s_dict"].apply(lambda d: float(d.get("confidence", 0.0)))
    # Normalise to UTC-naive numpy datetime64 so np.searchsorted has a real
    # numpy time array (tz-aware Series .to_numpy() returns an OBJECT array
    # of pd.Timestamp, which then breaks tz comparisons).
    n["published_at"] = (
        pd.to_datetime(n["published_at"], utc=True)
          .dt.tz_convert("UTC").dt.tz_localize(None)
    )
    n = n.sort_values("published_at")

    # Restrict to the time-range we'll need.
    bars = bars.copy()
    bars["open_time"] = (
        pd.to_datetime(bars["open_time"], utc=True)
          .dt.tz_convert("UTC").dt.tz_localize(None)
    )
    bars = bars.sort_values("open_time")
    earliest_needed = bars["open_time"].min() - cfg.lookback
    n = n[n["published_at"] >= earliest_needed]
    if n.empty:
        return _empty(bars)

    # 2. for every bar, slice the news frame by [t - lookback, t)
    # Both arrays are now naive datetime64[ns] so searchsorted is happy.
    pub_ts = n["published_at"].values.astype("datetime64[ns]")
    scores = n["score"].to_numpy()
    confs = n["conf"].to_numpy()

    out = {"sent_count": [], "sent_mean": [], "sent_std": [],
           "sent_min": [], "sent_max": [], "sent_last": [],
           "sent_conf_mean": []}

    for t in bars["open_time"].values.astype("datetime64[ns]"):
        lo = pd.Timestamp(t) - cfg.lookback
        # Strict-less-than for upper bound = no future leak.
        i_lo = int(np.searchsorted(pub_ts, np.datetime64(lo.to_datetime64())))
        i_hi = int(np.searchsorted(pub_ts, t))
        if i_hi <= i_lo:
            out["sent_count"].append(0)
            for k in ("sent_mean", "sent_std", "sent_min",
                      "sent_max", "sent_last", "sent_conf_mean"):
                out[k].append(0.0)
            continue
        s = scores[i_lo:i_hi]
        c = confs[i_lo:i_hi]
        out["sent_count"].append(int(len(s)))
        out["sent_mean"].append(float(np.mean(s)))
        out["sent_std"].append(float(np.std(s)) if len(s) > 1 else 0.0)
        out["sent_min"].append(float(np.min(s)))
        out["sent_max"].append(float(np.max(s)))
        out["sent_last"].append(float(s[-1]))
        out["sent_conf_mean"].append(float(np.mean(c)))

    feats = pd.DataFrame(out, index=bars.index)
    feats["open_time"] = bars["open_time"].values
    return feats[["open_time"] + [c for c in feats.columns if c != "open_time"]]


def _empty(bars: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "open_time":      bars["open_time"].values,
        "sent_count":     0,
        "sent_mean":      0.0,
        "sent_std":       0.0,
        "sent_min":       0.0,
        "sent_max":       0.0,
        "sent_last":      0.0,
        "sent_conf_mean": 0.0,
    })
    return out


# ---------------------------------------------------------------------------
def load_all_scored_news(news_root) -> pd.DataFrame:
    """Concatenate every news Parquet under ``news_root``."""
    from pathlib import Path
    news_root = Path(news_root)
    files = sorted(news_root.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)
