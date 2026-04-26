"""Central config loader.

Reads ``config.yaml`` from the project root and exposes a typed accessor.
Also pulls API keys from ``.env`` (via python-dotenv) without crashing if
keys are missing — Phase 1 should run with zero keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = parent of the `src/` directory this file lives in.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

# Load .env once at import time. Silently no-op if the file is absent.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class BinanceRestCfg:
    requests_per_second: int = 8
    page_size: int = 1000
    max_retries: int = 5


@dataclass
class BinanceWsCfg:
    only_closed: bool = True
    reconnect_initial: int = 1
    reconnect_max: int = 60


@dataclass
class BinanceCfg:
    pairs: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=list)
    history_start: str = "max"
    rest: BinanceRestCfg = field(default_factory=BinanceRestCfg)
    websocket: BinanceWsCfg = field(default_factory=BinanceWsCfg)


@dataclass
class RssFeed:
    name: str
    url: str


@dataclass
class CryptoPanicCfg:
    enabled: bool = True
    base_url: str = "https://cryptopanic.com/api/v1/posts/"
    poll_interval_seconds: int = 120
    currencies: list[str] = field(default_factory=list)


@dataclass
class NewsCfg:
    rss_feeds: list[RssFeed] = field(default_factory=list)
    cryptopanic: CryptoPanicCfg = field(default_factory=CryptoPanicCfg)


@dataclass
class StorageCfg:
    root: str = "data"
    ohlcv_partition: str = "monthly"
    news_partition: str = "daily"
    compression: str = "snappy"

    @property
    def root_path(self) -> Path:
        """Absolute path to the data root."""
        p = Path(self.root)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p


@dataclass
class Config:
    binance: BinanceCfg
    news: NewsCfg
    storage: StorageCfg

    # API keys (resolved from environment, not config.yaml)
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    cryptopanic_api_key: str | None = None
    newsapi_key: str | None = None


def _coerce(raw: dict[str, Any]) -> Config:
    """Convert raw YAML into typed dataclasses (handles missing keys)."""
    b = raw.get("binance", {}) or {}
    binance = BinanceCfg(
        pairs=b.get("pairs", []),
        intervals=b.get("intervals", []),
        history_start=str(b.get("history_start", "max")),
        rest=BinanceRestCfg(**(b.get("rest") or {})),
        websocket=BinanceWsCfg(**(b.get("websocket") or {})),
    )

    n = raw.get("news", {}) or {}
    news = NewsCfg(
        rss_feeds=[RssFeed(**f) for f in (n.get("rss_feeds") or [])],
        cryptopanic=CryptoPanicCfg(**(n.get("cryptopanic") or {})),
    )

    storage = StorageCfg(**(raw.get("storage") or {}))

    return Config(
        binance=binance,
        news=news,
        storage=storage,
        binance_api_key=os.getenv("BINANCE_API_KEY") or None,
        binance_api_secret=os.getenv("BINANCE_API_SECRET") or None,
        cryptopanic_api_key=os.getenv("CRYPTOPANIC_API_KEY") or None,
        newsapi_key=os.getenv("NEWSAPI_KEY") or None,
    )


def load_config(path: Path | str | None = None) -> Config:
    """Load and parse the project config. Cached after first call."""
    global _CACHED
    if _CACHED is not None and path is None:
        return _CACHED
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _coerce(raw)
    if path is None:
        _CACHED = cfg
    return cfg


_CACHED: Config | None = None
