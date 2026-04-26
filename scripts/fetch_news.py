"""Entry point: pull all configured news feeds once.

Run via cron / Task Scheduler every few minutes for steady accumulation.

Usage:
    python scripts/fetch_news.py
    python scripts/fetch_news.py --loop      # CryptoPanic continuous polling
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.ingest.news import run_cryptopanic_loop, run_once  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch crypto news (RSS + CryptoPanic).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--loop", action="store_true",
                        help="Long-lived CryptoPanic poller instead of one-shot.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.loop:
        run_cryptopanic_loop(cfg)
    else:
        run_once(cfg)
