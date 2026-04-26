"""Entry point: download all configured historical OHLCV.

Usage:
    python scripts/fetch_historical.py
    python scripts/fetch_historical.py --config path/to/other.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `from src...` imports work when invoked
# directly (i.e. without `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.historical import main  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance historical OHLCV.")
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml (defaults to project-root config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
