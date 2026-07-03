"""Phase 7 inference service.

Loads every artefact the dashboard needs **once at process startup** so
each HTTP request only does cheap per-query work:

* Phase 1 OHLCV → in-memory DataFrame.
* Phase 2 numerical features → DataFrame indexed by ``open_time``.
* Phase 3 CNN embeddings parquet → DataFrame indexed by ``open_time``.
* Phase 3 trained CNN state-dict (taken from
  ``experiments/retrieval/cnn_ewc_naive/final_model.pt`` if present) so
  Grad-CAM can run on any bar's chart window.
* Phase 4 per-bar sentiment features.
* Phase 5 FusionBaseline, *fit on the first 90 % of the overlap window*
  with inductive-conformal calibration; queries on later bars get
  unbiased predictions with calibrated 90 % prediction sets.
* Phase 6 HMM regime labels + FAISS index via :class:`PatternEngine`.

Two design choices worth pointing out:

1. **Single anchor model, not walk-forward.** The dashboard is for live
   exploration; we fit one fusion model on the first 90 % of bars and
   serve predictions on the last 10 %. This keeps the API responses
   trivially deterministic.
2. **Window length matches Phase 3** (64 bars). When the UI clicks a
   candle at ``open_time = t``, we render the window ending at ``t``,
   embed it, and look up its top-K — exactly what the CNN saw during
   training, so the embedding is in-distribution.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.features.dataset import build_dataset, load_ohlcv
from src.features.numerical import build_features
from src.features.targets import make_direction_label
from src.models.baseline_fusion import FusionBaseline
from src.retrieval.hmm_regimes import REGIME_LABELS, load_regimes
from src.retrieval.pattern_engine import PatternEngine

log = logging.getLogger("phase7.inference")

# Phase 3 chart-image window length. Hard-coded because the persisted
# CNN was trained on this; changing it requires retraining.
CHART_WINDOW = 64
CHART_HW = 64
EMB_DIM = 128

# When fitting the anchor fusion model, reserve this much of the front
# of the overlap as the training+calibration window. Everything after
# is the "demo predictable" tail.
ANCHOR_FIT_FRAC = 0.90


@dataclass(frozen=True)
class CandleRow:
    open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class SimilarRow:
    open_time: pd.Timestamp
    similarity: float
    regime: int
    regime_name: str


# ---------------------------------------------------------------------------
class InferenceService:
    """Process-wide singleton, constructed once in the FastAPI lifespan."""

    def __init__(self, pair: str = "BTCUSDT", interval: str = "1h",
                 encoder: str = "candle") -> None:
        self.pair = pair
        self.interval = interval
        self.encoder = encoder

        log.info(f"[InferenceService] loading artefacts for {pair} {interval}...")
        self._load_ohlcv()
        self._load_features_and_labels()
        self._load_embeddings()
        self._load_regimes()
        self._load_pattern_engine()
        self._load_cnn()
        self._fit_anchor_fusion()
        log.info("[InferenceService] ready")

    # ------------------------------------------------------------------
    def _load_ohlcv(self) -> None:
        df = load_ohlcv(self.pair, self.interval)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time").sort_index()
        self._ohlcv = df

    def _load_features_and_labels(self) -> None:
        ds = build_dataset(self.pair, self.interval)
        self._features = ds.X            # DatetimeIndex of open_time
        self._labels = ds.y
        self._close = ds.close
        self._feature_names = ds.feature_names

    def _load_embeddings(self) -> None:
        path = (PROJECT_ROOT / "experiments" / "embeddings"
                / self.pair / self.interval / self.encoder / "embeddings.parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"missing CNN embeddings at {path}.  Run "
                f"scripts/extract_chart_embeddings.py first."
            )
        emb = pd.read_parquet(path)
        emb["open_time"] = pd.to_datetime(emb["open_time"], utc=True)
        emb = emb.set_index("open_time").sort_index()
        self._embeddings = emb

    def _load_regimes(self) -> None:
        self._regimes = load_regimes(self.pair, self.interval)

    def _load_pattern_engine(self) -> None:
        self._engine = PatternEngine(self.pair, self.interval)

    def _load_cnn(self) -> None:
        """Load a CNN checkpoint for Grad-CAM. Phase-6 EWC training
        produces ``experiments/retrieval/cnn_ewc_naive/final_model.pt`` —
        we use that if available. If not, Grad-CAM is disabled and the
        relevant endpoint returns a 503.
        """
        import torch  # noqa: PLC0415
        from src.vision.cnn import ChartCNN  # noqa: PLC0415
        from src.vision.trainer import _Normalise  # noqa: PLC0415

        ckpt_path = (PROJECT_ROOT / "experiments" / "retrieval"
                     / "cnn_ewc_naive" / "final_model.pt")
        self._cnn: Optional[ChartCNN] = None
        self._norm: Optional[_Normalise] = None
        if not ckpt_path.exists():
            log.warning(
                f"[InferenceService] no CNN checkpoint at {ckpt_path}; "
                f"Grad-CAM will be disabled. Run scripts/train_cnn_ewc.py "
                f"to enable it."
            )
            return
        # ``weights_only=False`` is the historical default and lets us load
        # the dict that ``train_cnn_ewc.py`` wrote (state_dict + tensors).
        # We *own* this checkpoint, so the pickle-trust warning is noise.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChartCNN()
        model.load_state_dict(ckpt["model"])
        model.eval()
        norm = _Normalise(ckpt["norm_mean"], ckpt["norm_std"])
        self._cnn = model
        self._norm = norm
        log.info(f"[InferenceService] loaded CNN from {ckpt_path}")

    # ------------------------------------------------------------------
    def _fit_anchor_fusion(self) -> None:
        n = len(self._features)
        cut = int(n * ANCHOR_FIT_FRAC)
        X_fit = self._features.iloc[:cut]
        y_fit = self._labels.iloc[:cut]

        model = FusionBaseline(
            use_cnn=True, use_sentiment=True,
            encoder=self.encoder,
            conformal=True, conformal_alpha=0.10,
        )
        model.attach_context(self.pair, self.interval)
        model.fit(X_fit, y_fit)
        self._fusion = model
        self._anchor_cut = cut
        log.info(
            f"[InferenceService] anchor fusion fit on {cut:,} rows; "
            f"demo-predictable window = last {n - cut:,} rows"
        )

    # ------------------------------------------------------------------
    # Public read-side API consumed by routes.py
    # ------------------------------------------------------------------
    def list_pairs(self) -> list[str]:
        return [self.pair]

    def get_candles(self, start: Optional[pd.Timestamp] = None,
                    end: Optional[pd.Timestamp] = None,
                    limit: int = 500) -> list[CandleRow]:
        df = self._ohlcv
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if limit and len(df) > limit:
            df = df.iloc[-limit:]              # most recent ``limit`` bars
        cols = ["open", "high", "low", "close", "volume"]
        out: list[CandleRow] = []
        for ts, row in df[cols].iterrows():
            out.append(CandleRow(
                open_time=pd.Timestamp(ts),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]),
            ))
        return out

    def get_regimes(self, start: Optional[pd.Timestamp] = None,
                    end: Optional[pd.Timestamp] = None,
                    limit: int = 500) -> list[dict]:
        df = self._regimes
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        if limit and len(df) > limit:
            df = df.iloc[-limit:]
        out: list[dict] = []
        for ts, row in df.iterrows():
            rid = int(row["regime"])
            out.append({
                "open_time": pd.Timestamp(ts),
                "regime": rid,
                "regime_name": REGIME_LABELS.get(rid, f"state_{rid}"),
            })
        return out

    # ------------------------------------------------------------------
    def predict(self, open_time: pd.Timestamp) -> dict:
        """Return fused prediction + conformal set for one query bar."""
        ts = pd.Timestamp(open_time, tz="UTC") if pd.Timestamp(open_time).tzinfo is None \
            else pd.Timestamp(open_time).tz_convert("UTC")

        if ts not in self._features.index:
            raise KeyError(f"no engineered features for {ts}")

        # Build a one-row DataFrame so FusionBaseline can reindex its
        # CNN + sentiment blocks against it.
        X_one = self._features.loc[[ts]]
        prob = float(self._fusion.predict_proba(X_one)[0])
        cp = self._fusion.predict_set(X_one)[0]   # (2,) bool array

        regime = None
        if ts in self._regimes.index:
            regime = int(self._regimes.loc[ts, "regime"])

        return {
            "open_time": ts,
            "p_up": prob,
            "label": int(prob >= 0.5),
            "conformal_set": {
                "down": bool(cp[0]),
                "up": bool(cp[1]),
            },
            "regime": regime,
            "regime_name": (REGIME_LABELS.get(regime, f"state_{regime}")
                            if regime is not None else None),
            "is_in_demo_window": bool(
                self._features.index.get_loc(ts) >= self._anchor_cut
            ),
        }

    # ------------------------------------------------------------------
    def similar(self, open_time: pd.Timestamp, k: int = 10,
                regime_filter: bool = True) -> list[SimilarRow]:
        ts = pd.Timestamp(open_time, tz="UTC") if pd.Timestamp(open_time).tzinfo is None \
            else pd.Timestamp(open_time).tz_convert("UTC")

        if ts not in self._embeddings.index:
            raise KeyError(f"no CNN embedding for {ts}")

        cols = [c for c in self._embeddings.columns if c.startswith("e")]
        q_vec = self._embeddings.loc[ts, cols].to_numpy(dtype=np.float32)

        q_regime = None
        if ts in self._regimes.index:
            q_regime = int(self._regimes.loc[ts, "regime"])

        # Exclude any match within ±1 bar of the query (1h = 3600s).
        bar_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
            "4h": 14_400, "1d": 86_400,
        }.get(self.interval, 3600)

        matches = self._engine.topk(
            q_vec, k=k,
            regime_filter=q_regime if (regime_filter and q_regime is not None) else None,
            exclude_self_within_seconds=bar_seconds,
            query_open_time=ts,
        )
        return [
            SimilarRow(
                open_time=m.open_time,
                similarity=m.similarity,
                regime=m.regime,
                regime_name=m.regime_name,
            )
            for m in matches
        ]

    # ------------------------------------------------------------------
    def gradcam_png(self, open_time: pd.Timestamp) -> bytes:
        """Render the chart window ending at ``open_time``, run Grad-CAM,
        and return a PNG blending the heatmap on top of the candle image.
        """
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from src.vision.encoders.candlestick import render_candlestick  # noqa: PLC0415
        from src.vision.gradcam import GradCAM, overlay_on_image  # noqa: PLC0415

        if self._cnn is None or self._norm is None:
            raise RuntimeError(
                "Grad-CAM unavailable — no CNN checkpoint loaded. "
                "Run scripts/train_cnn_ewc.py to produce one."
            )

        ts = pd.Timestamp(open_time, tz="UTC") if pd.Timestamp(open_time).tzinfo is None \
            else pd.Timestamp(open_time).tz_convert("UTC")

        # Pull the OHLC window ending at ts.
        end_pos = self._ohlcv.index.get_indexer([ts])[0]
        if end_pos < CHART_WINDOW - 1:
            raise KeyError(
                f"not enough history before {ts} to render a "
                f"{CHART_WINDOW}-bar window"
            )
        start_pos = end_pos - CHART_WINDOW + 1
        win = self._ohlcv.iloc[start_pos:end_pos + 1][
            ["open", "high", "low", "close"]
        ].to_numpy(dtype=np.float64)

        img_chw_u8 = render_candlestick(win, hw=CHART_HW)            # (3, H, W) uint8
        img_chw_f32 = img_chw_u8.astype(np.float32) / 255.0
        x = torch.from_numpy(img_chw_f32).unsqueeze(0)                # (1, 3, H, W)

        # Apply the persisted per-channel normaliser, then GradCAM.
        x_norm = self._norm(x)
        cam_helper = GradCAM(self._cnn)
        try:
            heat, prob = cam_helper(x_norm)
        finally:
            cam_helper.close()

        overlay = overlay_on_image(img_chw_u8, heat, alpha=0.45)      # (H, W, 3) uint8

        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Phase 8 — live-bar ingestion + on-the-fly prediction
    # ------------------------------------------------------------------
    def append_live_bar(self, candle: dict) -> dict:
        """Ingest a closed candle, recompute features + embedding, predict.

        ``candle`` must carry at least
        ``{open_time, open, high, low, close, volume}``. Idempotent: if
        we already have that ``open_time`` on disk we just re-predict on
        the existing row.

        Returns the same dict shape as :meth:`predict` plus the close
        price (used by the paper trader to settle the previous position
        and open the new one) and a rolling volatility metric.
        """
        ts = pd.Timestamp(candle["open_time"]).tz_convert("UTC") \
            if pd.Timestamp(candle["open_time"]).tzinfo is not None \
            else pd.Timestamp(candle["open_time"], tz="UTC")

        if ts not in self._ohlcv.index:
            # Build the new row with explicit defaults for every column
            # build_features depends on. ``taker_buy_base`` defaults to
            # half the volume (neutral split) when not provided — which
            # matters for the smoke and for any consumer that doesn't
            # plumb the full kline through.
            volume = float(candle["volume"])
            row_dict: dict = {
                "open_time":   ts,
                "open":        float(candle["open"]),
                "high":        float(candle["high"]),
                "low":         float(candle["low"]),
                "close":       float(candle["close"]),
                "volume":      volume,
                "close_time":  pd.Timestamp(candle.get("close_time", ts)),
                "quote_volume":   float(candle.get("quote_volume", 0.0)),
                "trades":         int(candle.get("trades", 0)),
                "taker_buy_base": float(candle.get("taker_buy_base", volume * 0.5)),
                "taker_buy_quote": float(candle.get("taker_buy_quote", 0.0)),
                "ingested_at": pd.Timestamp.utcnow().tz_localize(None)
                               if pd.Timestamp.utcnow().tzinfo is None
                               else pd.Timestamp.utcnow(),
            }
            row = pd.DataFrame([row_dict]).set_index("open_time")
            # Align to the historical cache's column set so concat keeps
            # a stable schema (no all-NaN columns => silences pandas'
            # empty-column concat FutureWarning).
            for c in self._ohlcv.columns:
                if c not in row.columns:
                    row[c] = self._ohlcv[c].iloc[-1]   # carry last value forward
            row = row[self._ohlcv.columns]
            self._ohlcv = pd.concat([self._ohlcv, row]).sort_index()

        # Recompute features on a tail slice (cheap).
        feats_tail = build_features(
            self._ohlcv.iloc[-200:].reset_index()
        )
        # build_features re-indexes by open_time. Pick out the new row.
        if ts not in feats_tail.index:
            raise RuntimeError(
                f"live bar {ts} dropped during feature engineering "
                f"(probably warmup) — supply more historical context"
            )

        # Update labels: the previous bar can now be labelled.
        if len(self._ohlcv) >= 2:
            prev_ts = self._ohlcv.index[-2]
            prev_close = float(self._ohlcv.loc[prev_ts, "close"])
            this_close = float(self._ohlcv.loc[ts, "close"])
            if prev_ts in self._features.index:
                self._labels.loc[prev_ts] = int(this_close > prev_close)

        new_feat_row = feats_tail.loc[[ts], self._feature_names].astype("float64")
        if ts not in self._features.index:
            self._features = pd.concat([self._features, new_feat_row]).sort_index()
            # Current bar's label is unknown until the NEXT bar closes.
            self._labels = pd.concat([
                self._labels,
                pd.Series([0], index=[ts], name=self._labels.name, dtype=self._labels.dtype),
            ])
            self._close = pd.concat([
                self._close,
                pd.Series([float(candle["close"])], index=[ts], name=self._close.name),
            ])

        # Embed the chart window ending at ts via the loaded CNN, so the
        # fusion model's CNN-block reindex resolves cleanly.
        emb_vec = self._embed_window_ending_at(ts)
        if emb_vec is not None:
            cols = [c for c in self._embeddings.columns if c.startswith("e")]
            if ts not in self._embeddings.index:
                emb_row = pd.DataFrame([emb_vec], index=[ts], columns=cols)
                self._embeddings = pd.concat([self._embeddings, emb_row]).sort_index()
                # Force the fusion baseline to re-load embeddings next call.
                self._fusion._embeddings = self._embeddings

        # Run prediction on the new bar via the shared `predict` path.
        out = self.predict(ts)
        # Decorate with extras needed by the paper trader.
        out["close"] = float(self._ohlcv.loc[ts, "close"])
        out["realised_vol_24h"] = self._realised_vol_at(ts, lookback=24)
        out["realised_vol_168h"] = self._realised_vol_at(ts, lookback=168)
        return out

    # ------------------------------------------------------------------
    def _embed_window_ending_at(self, ts: pd.Timestamp) -> Optional[np.ndarray]:
        """Render the chart window ending at ``ts`` and embed via the CNN.

        Returns ``None`` if the CNN checkpoint isn't loaded (Grad-CAM
        disabled) or there's not enough history yet.
        """
        if self._cnn is None or self._norm is None:
            return None
        import torch                                              # noqa: PLC0415
        from src.vision.encoders.candlestick import render_candlestick  # noqa: PLC0415

        end_pos = self._ohlcv.index.get_indexer([ts])[0]
        if end_pos < CHART_WINDOW - 1:
            return None
        start_pos = end_pos - CHART_WINDOW + 1
        win = self._ohlcv.iloc[start_pos:end_pos + 1][
            ["open", "high", "low", "close"]
        ].to_numpy(dtype=np.float64)
        img = render_candlestick(win, hw=CHART_HW).astype(np.float32) / 255.0
        x = torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            emb = self._cnn.embed(self._norm(x)).cpu().numpy().astype(np.float32)
        return emb[0]                                              # (128,)

    def _realised_vol_at(self, ts: pd.Timestamp, lookback: int) -> float:
        """Std of log-returns over the last ``lookback`` bars ending at ts."""
        idx = self._ohlcv.index.get_indexer([ts])[0]
        if idx < lookback:
            return float("nan")
        closes = self._ohlcv["close"].iloc[idx - lookback:idx + 1].to_numpy(dtype=np.float64)
        rets = np.log(closes[1:] / closes[:-1])
        return float(np.std(rets, ddof=0))

    # ------------------------------------------------------------------
    @property
    def total_bars(self) -> int:
        return len(self._ohlcv)

    @property
    def anchor_cut(self) -> int:
        return self._anchor_cut

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self._ohlcv.index.min(), self._ohlcv.index.max())
