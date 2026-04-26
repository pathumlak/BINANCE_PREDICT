"""News ingestion (RSS + CryptoPanic).

Two pipelines:

1. **RSS feeds** (CoinDesk, CoinTelegraph, Decrypt, Bitcoin Magazine).
   One-shot fetch — call repeatedly via cron / scheduler.

2. **CryptoPanic** posts API.
   Polls every ``poll_interval_seconds`` and persists new posts. Can run
   stand-alone in a long-lived loop if you call ``run_cryptopanic_loop``.

Both pipelines write to ``data/news/<source>/<YYYY-MM-DD>.parquet``,
deduped on URL. Each row carries:

  - ``published_at``: when the article was published (per the source)
  - ``ingested_at``:  when WE saw it (used to detect lookahead bugs)

If ``published_at > ingested_at`` ever appears at training time, you've
got a leak — *every* model build should assert against that.
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Config, RssFeed, load_config
from src.utils.storage import news_partition_path, utc_now_ms, write_news_partition

# Map free-form ticker mentions inside text to canonical symbols.
TICKER_REGEX = re.compile(r"\b(BTC|BITCOIN|ETH|ETHEREUM|BNB|SOL|SOLANA|XRP|RIPPLE)\b", re.I)
TICKER_NORMALISE = {
    "btc": "BTC", "bitcoin": "BTC",
    "eth": "ETH", "ethereum": "ETH",
    "bnb": "BNB",
    "sol": "SOL", "solana": "SOL",
    "xrp": "XRP", "ripple": "XRP",
}


def _hash_id(url: str) -> str:
    """Stable id derived from URL — survives across runs."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def _strip_html(text: str | None) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def _extract_tickers(*texts: str | None) -> list[str]:
    found: set[str] = set()
    for t in texts:
        if not t:
            continue
        for m in TICKER_REGEX.findall(t):
            found.add(TICKER_NORMALISE[m.lower()])
    return sorted(found)


# ---------------------------------------------------------------------------
#  RSS
# ---------------------------------------------------------------------------

def fetch_rss(feed: RssFeed) -> pd.DataFrame:
    """Pull one RSS feed, return a DataFrame of articles."""
    logger.info(f"RSS: {feed.name} → {feed.url}")
    parsed = feedparser.parse(feed.url)
    if parsed.bozo and not parsed.entries:
        logger.warning(f"  bozo/empty feed: {parsed.bozo_exception}")
        return pd.DataFrame()

    rows = []
    ingested_at = utc_now_ms()
    for e in parsed.entries:
        url = e.get("link")
        if not url:
            continue
        title = e.get("title", "")
        summary = _strip_html(e.get("summary"))
        # feedparser exposes ``published_parsed`` as a struct_time tuple.
        if e.get("published_parsed"):
            published_at = pd.Timestamp(
                datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            )
        elif e.get("updated_parsed"):
            published_at = pd.Timestamp(
                datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
            )
        else:
            # No timestamp from source — fall back to now (last resort).
            published_at = ingested_at

        rows.append({
            "id": _hash_id(url),
            "source": feed.name,
            "title": title,
            "url": url,
            "summary": summary,
            "published_at": published_at,
            "ingested_at": ingested_at,
            "tickers": _extract_tickers(title, summary),
            "raw_sentiment": None,         # populated in Phase 4
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  CryptoPanic
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20), reraise=True)
def fetch_cryptopanic_page(cfg: Config, page: int = 1) -> dict:
    params = {
        "public": "true",
        "currencies": ",".join(cfg.news.cryptopanic.currencies),
        "page": page,
    }
    if cfg.cryptopanic_api_key:
        params["auth_token"] = cfg.cryptopanic_api_key
    resp = requests.get(cfg.news.cryptopanic.base_url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_cryptopanic(cfg: Config, max_pages: int = 1) -> pd.DataFrame:
    """One-shot CryptoPanic pull (default: just the first page).

    The free public API rate-limits hard, so we keep ``max_pages`` low and
    rely on frequent polling rather than deep paginate-back.
    """
    if not cfg.news.cryptopanic.enabled:
        return pd.DataFrame()

    rows = []
    ingested_at = utc_now_ms()
    for page in range(1, max_pages + 1):
        try:
            payload = fetch_cryptopanic_page(cfg, page=page)
        except requests.RequestException as e:
            logger.warning(f"CryptoPanic page {page} failed: {e}")
            break

        for post in payload.get("results", []):
            url = post.get("url") or post.get("source", {}).get("domain")
            if not url:
                continue
            title = post.get("title", "")
            published_at = pd.Timestamp(post.get("published_at"))
            if published_at.tzinfo is None:
                published_at = published_at.tz_localize("UTC")
            tickers = [c.get("code") for c in post.get("currencies") or [] if c.get("code")]
            rows.append({
                "id": _hash_id(url),
                "source": "CryptoPanic",
                "title": title,
                "url": url,
                "summary": "",
                "published_at": published_at,
                "ingested_at": ingested_at,
                "tickers": sorted(set(tickers)),
                "raw_sentiment": post.get("votes"),  # upvote/downvote dict
            })
        if not payload.get("next"):
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

def _persist(df: pd.DataFrame, root: Path, compression: str) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df["_date"] = df["published_at"].dt.strftime("%Y-%m-%d")
    written = 0
    for (source, date), sub in df.groupby(["source", "_date"], sort=True):
        sub = sub.drop(columns="_date")
        ts = pd.Timestamp(sub["published_at"].iloc[0])
        path = news_partition_path(root, source, ts)
        write_news_partition(sub, path, compression=compression)
        written += len(sub)
    return written


def run_once(cfg: Config) -> int:
    """Pull every configured RSS feed + one CryptoPanic page, then persist."""
    frames: list[pd.DataFrame] = []
    for feed in cfg.news.rss_feeds:
        try:
            frames.append(fetch_rss(feed))
        except Exception as e:  # noqa: BLE001 — never let one feed kill the rest
            logger.exception(f"RSS feed {feed.name} blew up: {e}")

    try:
        frames.append(fetch_cryptopanic(cfg))
    except Exception as e:  # noqa: BLE001
        logger.exception(f"CryptoPanic fetch failed: {e}")

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    written = _persist(df, cfg.storage.root_path, cfg.storage.compression)
    logger.success(f"news: persisted {written} articles")
    return written


def run_cryptopanic_loop(cfg: Config) -> None:
    """Long-lived poller for CryptoPanic, useful as a sidecar process."""
    interval = cfg.news.cryptopanic.poll_interval_seconds
    logger.info(f"CryptoPanic loop: polling every {interval}s")
    while True:
        try:
            df = fetch_cryptopanic(cfg)
            _persist(df, cfg.storage.root_path, cfg.storage.compression)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"loop iteration failed: {e}")
        time.sleep(interval)


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    run_once(cfg)


if __name__ == "__main__":
    main()
