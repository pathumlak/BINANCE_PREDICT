"""Entry point: score every news article on disk with the sentiment ensemble.

Usage:
    python scripts/score_news.py
    python scripts/score_news.py --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlp.scoring import score_all  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    score_all(batch_size=args.batch_size)
