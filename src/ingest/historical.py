"""Historical OHLCV downloader for Binance.

Downloads candles for every (pair, interval) in ``config.yaml`` from the
pair's listing date forward, saving monthly Parquet partitions under
``data/ohlcv/<PAIR>/<INTERVAL>/<YYYY-MM>.parquet``.

Design notes
------------
* **Resumable.** Before downloading we check the latest ``open_time`` already
  on disk and start from there + one interval. Crash → just rerun.
* **Rate-limited.** The Binance public REST API allows ~1200 weight/min and
  one /klines request costs 1 weight. We sleep to stay well under.
* **Retried.** Network blips are retried with exponential backoff via
  ``tenacity``. Persistent 4xx errors fail loudly.
* **Schema-pinned.** All partitions follow ``OHLCV_SCHEMA`` from
  ``utils.storage`` so downstream tooling can treat the dataset as one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.config import Config, load_config
from src.utils.storage import (
    ohlcv_partition_path,
    latest_open_time,
    utc_now_ms,
    write_ohlcv_partition,
)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

# Number of milliseconds in each Binance interval. Used to advance the cursor
# and to chunk requests into monthly partitions.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


@dataclass
class FetchPlan:
    """A pair × interval download job."""

    pair: str
    interval: str
    start_ms: int          # inclusive
    end_ms: int            # exclusive
    interval_ms: int


# ---------------------------------------------------------------------------
#  Binance HTTP helpers
# ---------------------------------------------------------------------------

class BinanceRESTError(RuntimeError):
    """Raised for non-retryable Binance API errors (4xx other than 429)."""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((requests.RequestException,)),
)
def _http_get(url: str, params: dict) -> list | dict:
    resp = requests.get(url, params=params, timeout=20)
    # 429 = rate limited, 418 = banned. Both are retryable; the backoff above
    # gives the server a breather before we try again.
    if resp.status_code in (429, 418):
        retry_after = int(resp.headers.get("Retry-After", "5"))
        logger.warning(f"Binance rate-limited us — sleeping {retry_after}s")
        time.sleep(retry_after)
        raise requests.RequestException(f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        raise BinanceRESTError(f"{resp.status_code}: {resp.text[:200]}")
    return resp.json()


def fetch_listing_time_ms(pair: str) -> int:
    """Return the earliest available 1m candle's open_time for ``pair``.

    Binance's /exchangeInfo doesn't expose a listing date directly, so we
    request 1 candle starting at unix-epoch zero and read its open_time.
    """
    data = _http_get(
        BINANCE_KLINES_URL,
        {"symbol": pair, "interval": "1m", "startTime": 0, "limit": 1},
    )
    if not data:
        raise BinanceRESTError(f"No klines returned for {pair} — symbol valid?")
    return int(data[0][0])


# ---------------------------------------------------------------------------
#  Download loop
# ---------------------------------------------------------------------------

def _kline_rows_to_df(rows: list[list], ingested_at: pd.Timestamp) -> pd.DataFrame:
    """Convert raw Binance kline payload into a typed DataFrame matching schema."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ]
    df = pd.DataFrame(rows, columns=cols).drop(columns="_ignore")

    # ms -> UTC timestamp
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ("open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"):
        df[col] = df[col].astype("float64")
    df["trades"] = df["trades"].astype("int64")

    df["ingested_at"] = ingested_at
    return df


def _iter_pages(plan: FetchPlan, page_size: int, sleep_s: float) -> Iterator[pd.DataFrame]:
    """Yield one page of candles at a time, advancing the cursor each call."""
    cursor = plan.start_ms
    while cursor < plan.end_ms:
        ingested_at = utc_now_ms()
        page = _http_get(
            BINANCE_KLINES_URL,
            {
                "symbol": plan.pair,
                "interval": plan.interval,
                "startTime": cursor,
                "endTime": plan.end_ms - 1,
                "limit": page_size,
            },
        )
        if not page:
            return
        df = _kline_rows_to_df(page, ingested_at)
        yield df
        # Advance cursor to one ms past the last candle to avoid duplicates.
        cursor = int(page[-1][0]) + plan.interval_ms
        time.sleep(sleep_s)


def _flush_by_month(
    df: pd.DataFrame, root: Path, pair: str, interval: str, compression: str
) -> int:
    """Split a multi-month dataframe across monthly partitions and write each."""
    if df.empty:
        return 0
    df = df.copy()
    df["_month"] = df["open_time"].dt.strftime("%Y-%m")
    written = 0
    for month, sub in df.groupby("_month", sort=True):
        sub = sub.drop(columns="_month")
        ts = pd.Timestamp(sub["open_time"].iloc[0])
        path = ohlcv_partition_path(root, pair, interval, ts)
        write_ohlcv_partition(sub, path, compression=compression)
        written += len(sub)
    return written


def build_plans(cfg: Config) -> list[FetchPlan]:
    """Compute the (pair, interval) jobs we need to run, accounting for resume."""
    plans: list[FetchPlan] = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for pair in cfg.binance.pairs:
        # Find this pair's listing date once, reuse across intervals.
        if cfg.binance.history_start == "max":
            listing_ms = fetch_listing_time_ms(pair)
            logger.info(
                f"{pair} listing time: "
                f"{datetime.fromtimestamp(listing_ms/1000, tz=timezone.utc).isoformat()}"
            )
        else:
            listing_ms = int(
                pd.Timestamp(cfg.binance.history_start, tz="UTC").timestamp() * 1000
            )

        for interval in cfg.binance.intervals:
            if interval not in INTERVAL_MS:
                logger.warning(f"Skipping unknown interval: {interval}")
                continue
            interval_ms = INTERVAL_MS[interval]

            # Resume: start from one interval past the latest candle on disk.
            on_disk = latest_open_time(cfg.storage.root_path, pair, interval)
            if on_disk is not None:
                start_ms = int(on_disk.timestamp() * 1000) + interval_ms
            else:
                start_ms = listing_ms

            if start_ms >= end_ms:
                logger.info(f"{pair} {interval}: already up-to-date")
                continue

            plans.append(
                FetchPlan(
                    pair=pair,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    interval_ms=interval_ms,
                )
            )
    return plans


def run_plan(plan: FetchPlan, cfg: Config) -> int:
    """Execute one (pair, interval) download. Returns rows written."""
    sleep_s = 1.0 / max(cfg.binance.rest.requests_per_second, 1)
    page_size = cfg.binance.rest.page_size

    total_candles_est = max(1, (plan.end_ms - plan.start_ms) // plan.interval_ms)
    total_pages = (total_candles_est + page_size - 1) // page_size

    written = 0
    buffer: list[pd.DataFrame] = []
    BUFFER_LIMIT = 50  # flush after ~50 pages so we don't hold huge memory

    pbar = tqdm(
        total=total_pages,
        desc=f"{plan.pair} {plan.interval}",
        unit="page",
        leave=False,
    )
    for page_df in _iter_pages(plan, page_size=page_size, sleep_s=sleep_s):
        buffer.append(page_df)
        pbar.update(1)
        if len(buffer) >= BUFFER_LIMIT:
            written += _flush_by_month(
                pd.concat(buffer, ignore_index=True),
                cfg.storage.root_path,
                plan.pair,
                plan.interval,
                cfg.storage.compression,
            )
            buffer.clear()
    pbar.close()

    if buffer:
        written += _flush_by_month(
            pd.concat(buffer, ignore_index=True),
            cfg.storage.root_path,
            plan.pair,
            plan.interval,
            cfg.storage.compression,
        )
    return written


def main(config_path: str | None = None,
         pairs: list[str] | None = None,
         intervals: list[str] | None = None) -> None:
    """Fetch missing OHLCV.

    Optional overrides of ``pairs`` / ``intervals`` narrow the sweep at
    the CLI without editing ``config.yaml`` — useful for a one-off
    backfill (e.g. ``--pairs BTCUSDT --interval 1h``).
    """
    cfg = load_config(config_path)
    if pairs:
        cfg.binance.pairs = list(pairs)
    if intervals:
        cfg.binance.intervals = list(intervals)

    logger.info(f"Storage root: {cfg.storage.root_path}")
    logger.info(f"Pairs: {cfg.binance.pairs}   Intervals: {cfg.binance.intervals}")
    cfg.storage.root_path.mkdir(parents=True, exist_ok=True)

    plans = build_plans(cfg)
    if not plans:
        logger.success("Nothing to do — every (pair, interval) is up-to-date.")
        return

    logger.info(f"Planned {len(plans)} download jobs.")
    grand_total = 0
    for plan in plans:
        rows = run_plan(plan, cfg)
        grand_total += rows
        logger.success(f"{plan.pair} {plan.interval}: +{rows:,} rows")
    logger.success(f"Done. {grand_total:,} candles written across all jobs.")


if __name__ == "__main__":
    main()
