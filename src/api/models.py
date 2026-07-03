"""Pydantic request/response schemas for the Phase 7 dashboard API.

Every datetime ships out as an ISO-8601 UTC string so the frontend can
hand it straight to TradingView Lightweight Charts (which accepts a UNIX
seconds timestamp, but ISO is the safest interchange format).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    pair: str
    interval: str
    total_bars: int
    n_features: int
    time_range_start: str
    time_range_end: str
    anchor_cut_row: int


class CandleResponse(BaseModel):
    open_time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class RegimeResponse(BaseModel):
    open_time: str
    regime: int
    regime_name: str


class ConformalSet(BaseModel):
    down: bool
    up: bool


class PredictionResponse(BaseModel):
    open_time: str
    p_up: float = Field(..., description="P(next bar close > current close)")
    label: int = Field(..., description="argmax: 0 = down, 1 = up")
    conformal_set: ConformalSet
    regime: Optional[int] = None
    regime_name: Optional[str] = None
    is_in_demo_window: bool = Field(
        ..., description="True when the bar is in the held-out window "
                        "the anchor fusion model has never seen."
    )


class SimilarMatchResponse(BaseModel):
    open_time: str
    similarity: float = Field(..., description="Cosine similarity in [-1, 1]")
    regime: int
    regime_name: str


class SimilarResponse(BaseModel):
    query_open_time: str
    k: int
    regime_filter: bool
    matches: list[SimilarMatchResponse]
