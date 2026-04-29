"""Batch-score every news article on disk and persist the result.

Walks every Parquet under ``data/news/<source>/<date>.parquet`` produced
by Phase 1, and adds a ``sentiment`` column populated by the
``CryptoBERT + FinBERT`` ensemble defined in :mod:`src.nlp.sentiment`.

Resumable: rows whose ``sentiment`` is non-null are skipped, so re-runs
only score newly-arrived articles.

The text fed to the model is ``title + ". " + summary`` (summary is
present for RSS items, blank for CryptoPanic). 256-token truncation in
the pipeline keeps things fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import Config, load_config
from src.nlp.sentiment import score_texts


def _join_text(row: pd.Series) -> str:
    title = (row.get("title") or "").strip()
    summary = (row.get("summary") or "").strip()
    return f"{title}. {summary}".strip(" .") if summary else title


def _has_sentiment(df: pd.DataFrame) -> pd.Series:
    """True per row if the row already has a non-null sentiment dict."""
    if "sentiment" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["sentiment"].apply(lambda v: isinstance(v, (dict, str)) and bool(v))


def score_partition(path: Path, batch_size: int = 32) -> int:
    """Score one news Parquet partition. Returns rows newly scored."""
    df = pd.read_parquet(path)
    if df.empty:
        return 0

    todo_mask = ~_has_sentiment(df)
    todo_idx = df.index[todo_mask]
    if len(todo_idx) == 0:
        return 0

    texts = df.loc[todo_idx].apply(_join_text, axis=1).tolist()
    scores = score_texts(texts, batch_size=batch_size)

    # Store as JSON strings so the dtype stays well-defined on disk.
    payload = [json.dumps(s.as_dict()) for s in scores]

    if "sentiment" not in df.columns:
        df["sentiment"] = None
    df.loc[todo_idx, "sentiment"] = payload

    df.to_parquet(path, compression="snappy", index=False)
    return len(todo_idx)


def score_all(cfg: Config | None = None, batch_size: int = 32) -> int:
    cfg = cfg or load_config()
    root = cfg.storage.root_path / "news"
    if not root.exists():
        logger.warning(f"No news directory yet — run scripts/fetch_news.py first ({root})")
        return 0

    files = sorted(root.glob("*/*.parquet"))
    logger.info(f"scoring {len(files)} partitions…")
    grand_total = 0
    for f in tqdm(files, desc="news partitions", unit="file"):
        try:
            n = score_partition(f, batch_size=batch_size)
            if n:
                logger.info(f"  +{n:>4} → {f.parent.name}/{f.name}")
            grand_total += n
        except Exception as e:                   # noqa: BLE001
            logger.exception(f"failed to score {f}: {e}")
    logger.success(f"total newly-scored articles: {grand_total}")
    return grand_total
