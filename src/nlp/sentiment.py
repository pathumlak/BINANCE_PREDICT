"""CryptoBERT + FinBERT sentiment ensemble.

Scores a list of texts (headlines + summaries) and returns a calibrated
score per text in ``[-1.0, +1.0]``:

  * ``-1.0`` = strongly bearish / negative
  * ``+0.0`` = neutral / no signal
  * ``+1.0`` = strongly bullish / positive

Two backbones are used:

  * **CryptoBERT** (`ElKulako/cryptobert`) — crypto-native vocabulary,
    classes ``Bearish / Neutral / Bullish``.
  * **FinBERT**     (`ProsusAI/finbert`) — finance-domain BERT,
    classes ``positive / negative / neutral``.

Each model returns class probabilities; we project to a signed score by
``score = P(bull/positive) - P(bear/negative)`` and average the two
backbones to form the ensemble score. Returning the per-model and the
ensemble values keeps every downstream consumer honest about provenance.

Models load lazily on first call; until then the module is import-cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

# Lazy / optional import of transformers: keeping the top-level import
# clean means tests that don't touch sentiment aren't blocked by a heavy
# torch+transformers stack.
_PIPELINES: dict[str, "Pipeline"] = {}        # populated by _get_pipeline


@dataclass
class SentimentScore:
    """One row of ensemble output."""

    text: str
    cryptobert: float
    finbert: float
    ensemble: float
    confidence: float                 # mean per-model |score|, in [0, 1]

    def as_dict(self) -> dict:
        return {
            "cryptobert": float(self.cryptobert),
            "finbert":    float(self.finbert),
            "ensemble":   float(self.ensemble),
            "confidence": float(self.confidence),
        }


# ---------------------------------------------------------------------------
#  Lazy pipeline construction
# ---------------------------------------------------------------------------
_MODEL_IDS = {
    "cryptobert": "ElKulako/cryptobert",
    "finbert":    "ProsusAI/finbert",
}

# Map raw class label -> signed contribution.
# CryptoBERT exposes "Bearish/Neutral/Bullish" (capitalised).
# FinBERT exposes "positive/negative/neutral" (lowercase).
_LABEL_SIGN = {
    "bearish":  -1.0, "Bearish":  -1.0,
    "negative": -1.0, "Negative": -1.0,
    "neutral":   0.0, "Neutral":   0.0,
    "bullish":   1.0, "Bullish":   1.0,
    "positive":  1.0, "Positive":  1.0,
}


def _get_pipeline(name: str):
    """Return a HuggingFace text-classification pipeline, building once."""
    if name in _PIPELINES:
        return _PIPELINES[name]

    # Local imports so the module can be imported in pure-Phase-1 envs.
    from transformers import (        # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    model_id = _MODEL_IDS[name]
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=device,
        truncation=True,
        max_length=256,
        top_k=None,                  # return ALL class scores per item
    )
    _PIPELINES[name] = pipe
    return pipe


def _signed_score(class_dist: list[dict]) -> float:
    """Convert a HuggingFace ``[{label, score}, …]`` to one signed scalar."""
    score = 0.0
    for entry in class_dist:
        sign = _LABEL_SIGN.get(entry["label"], 0.0)
        score += sign * float(entry["score"])
    return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
def score_texts(texts: Iterable[str], batch_size: int = 32) -> list[SentimentScore]:
    """Run both backbones over ``texts`` and return ``SentimentScore`` per row.

    Empty / whitespace-only strings short-circuit to a zero score so
    callers don't need to pre-filter.
    """
    texts = [t if t and t.strip() else "" for t in texts]
    n = len(texts)
    if n == 0:
        return []

    # Score with both models.
    cb_pipe = _get_pipeline("cryptobert")
    fb_pipe = _get_pipeline("finbert")

    cb_out = cb_pipe(texts, batch_size=batch_size)
    fb_out = fb_pipe(texts, batch_size=batch_size)

    results: list[SentimentScore] = []
    for t, cb, fb in zip(texts, cb_out, fb_out):
        if not t:
            results.append(SentimentScore(text=t, cryptobert=0.0, finbert=0.0,
                                          ensemble=0.0, confidence=0.0))
            continue
        cb_s = _signed_score(cb)
        fb_s = _signed_score(fb)
        ens = 0.5 * (cb_s + fb_s)
        conf = 0.5 * (abs(cb_s) + abs(fb_s))
        results.append(SentimentScore(text=t, cryptobert=cb_s, finbert=fb_s,
                                      ensemble=ens, confidence=conf))
    return results


def score_text(text: str) -> SentimentScore:
    """Convenience wrapper for one-shot scoring."""
    return score_texts([text])[0]
