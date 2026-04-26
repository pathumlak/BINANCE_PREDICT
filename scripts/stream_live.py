"""Entry point: start the live WebSocket streamer.

Usage:
    python scripts/stream_live.py
    Ctrl-C to stop (final flush will run before exit).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.live import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Binance live OHLCV via WebSocket.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
