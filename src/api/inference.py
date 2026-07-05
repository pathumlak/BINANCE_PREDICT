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
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
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
        self._load_news()
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
    def _load_news(self) -> None:
        """Concatenate every scored news parquet into one in-memory frame.

        The dashboard shows the most recent N articles alongside their
        CryptoBERT+FinBERT ensemble score. Missing / malformed rows are
        gracefully skipped.
        """
        import json  # noqa: PLC0415
        cfg = load_config()
        root = cfg.storage.root_path / "news"
        self._news: Optional[pd.DataFrame] = None
        if not root.exists():
            log.warning(f"[InferenceService] no news root at {root}")
            return
        files = sorted(root.glob("*/*.parquet"))
        if not files:
            log.warning(f"[InferenceService] no news files under {root}")
            return
        try:
            frames = [pd.read_parquet(f) for f in files]
            news = pd.concat(frames, ignore_index=True)
        except Exception as e:                             # noqa: BLE001
            log.exception(f"news load failed: {e}")
            return

        # Decode sentiment JSON into columns.
        def _decode(s):
            if s is None or (isinstance(s, float) and np.isnan(s)):
                return None
            try:
                return json.loads(s) if isinstance(s, str) else s
            except Exception:                              # noqa: BLE001
                return None

        if "sentiment" in news.columns:
            decoded = news["sentiment"].map(_decode)
            news["sent_score"] = decoded.map(
                lambda d: (d.get("ensemble")
                           if isinstance(d, dict) else None)
            )
            news["sent_conf"] = decoded.map(
                lambda d: (d.get("confidence")
                           if isinstance(d, dict) else None)
            )
        else:
            news["sent_score"] = None
            news["sent_conf"] = None

        # Timestamps → tz-aware UTC.
        news["published_at"] = pd.to_datetime(news["published_at"], utc=True,
                                              errors="coerce")
        news = news.dropna(subset=["published_at"])
        news = news.sort_values("published_at").reset_index(drop=True)
        self._news = news
        log.info(f"[InferenceService] loaded {len(news):,} news articles")

    def get_news(self, limit: int = 25, min_ticker: Optional[str] = None) -> list[dict]:
        """Return the most-recent ``limit`` articles.

        ``min_ticker`` (e.g. ``"BTC"``) filters to articles whose
        ``tickers`` list contains that symbol; ``None`` returns all.
        """
        if self._news is None or self._news.empty:
            return []
        df = self._news
        if min_ticker:
            def _has(t):
                if isinstance(t, (list, tuple, np.ndarray)):
                    return min_ticker in t
                return False
            df = df[df["tickers"].map(_has)]
        df = df.tail(limit)

        out: list[dict] = []
        for _, r in df.iterrows():
            tickers = r.get("tickers")
            if not isinstance(tickers, (list, tuple, np.ndarray)):
                tickers = []
            score = r.get("sent_score")
            out.append({
                "published_at": pd.Timestamp(r["published_at"]).isoformat(),
                "source": str(r.get("source", "")),
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "tickers": list(tickers),
                "sent_score": (float(score) if score is not None and not
                                (isinstance(score, float) and np.isnan(score))
                                else None),
            })
        # Newest first.
        out.reverse()
        return out

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
    # V2 — similar-match OHLCV windows (for mini charts)
    # ------------------------------------------------------------------
    def latest_embedded_bar(self) -> Optional[pd.Timestamp]:
        """Newest bar that has both a CNN embedding AND a regime label.

        Used as an auto-fallback query for the top-K similarity strip
        when the current live bar has no embedding yet (because the CNN
        wasn't rerun after the last OHLCV backfill).
        """
        if self._embeddings is None or self._embeddings.empty:
            return None
        emb_idx = self._embeddings.index
        # Intersect with regimes so PatternEngine.topk doesn't itself
        # 404 on missing regime.
        if self._regimes is not None and not self._regimes.empty:
            common = emb_idx.intersection(self._regimes.index)
            if len(common) == 0:
                return None
            return pd.Timestamp(common.max())
        return pd.Timestamp(emb_idx.max())

    def similar_with_windows(self, open_time: pd.Timestamp, k: int = 5,
                             regime_filter: bool = True,
                             window: int = 64) -> list[dict]:
        """Top-K similar bars including their OHLC window for rendering."""
        matches = self.similar(open_time, k=k, regime_filter=regime_filter)
        out: list[dict] = []
        cols = ["open", "high", "low", "close"]
        for m in matches:
            end_pos = self._ohlcv.index.get_indexer([m.open_time])[0]
            if end_pos < 0 or end_pos < window - 1:
                continue
            start_pos = end_pos - window + 1
            win = self._ohlcv.iloc[start_pos:end_pos + 1][cols]
            # Also grab the NEXT bar's close so viewers can see the
            # outcome of the pattern — that's the punchline.
            next_close = None
            if end_pos + 1 < len(self._ohlcv):
                next_close = float(self._ohlcv["close"].iloc[end_pos + 1])
            out.append({
                "open_time": m.open_time,
                "similarity": m.similarity,
                "regime": m.regime,
                "regime_name": m.regime_name,
                "window": [
                    {"o": float(r.open), "h": float(r.high),
                     "l": float(r.low),  "c": float(r.close)}
                    for r in win.itertuples()
                ],
                "next_close": next_close,
                "next_direction": (
                    None if next_close is None
                    else int(next_close > float(win["close"].iloc[-1]))
                ),
            })
        return out

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
    # V2 — range analysis (for the "Analyze visible range" tool)
    # ------------------------------------------------------------------
    def analyze_range(self, start: pd.Timestamp, end: pd.Timestamp,
                      paper_predictions: Optional[list] = None,
                      n_hist_bins: int = 12) -> dict:
        """Rich summary of everything the app knows about ``[start, end]``.

        Returns a dict shaped for :func:`api.routes.analyze_range` to
        serialise directly to JSON.
        """
        if end < start:
            start, end = end, start

        ohlc = self._ohlcv.loc[start:end]
        n_bars = len(ohlc)

        # --- OHLC summary --------------------------------------------
        if n_bars > 0:
            first_close = float(ohlc["close"].iloc[0])
            last_close  = float(ohlc["close"].iloc[-1])
            high = float(ohlc["high"].max())
            low  = float(ohlc["low"].min())
            ret_pct = (last_close - first_close) / first_close if first_close else 0.0
            log_returns = np.log(ohlc["close"].to_numpy(dtype=np.float64) /
                                 np.roll(ohlc["close"].to_numpy(dtype=np.float64), 1))
            log_returns = log_returns[1:] if len(log_returns) > 1 else log_returns
            realised_vol = float(np.std(log_returns, ddof=0)) if len(log_returns) else 0.0
        else:
            first_close = last_close = high = low = ret_pct = realised_vol = 0.0

        # --- Regime histogram ---------------------------------------
        regime_hist = {"bear": 0, "sideways": 0, "bull": 0, "unknown": 0}
        if self._regimes is not None and not self._regimes.empty:
            sub = self._regimes.loc[start:end]
            for rid, cnt in sub["regime"].value_counts().items():
                name = REGIME_LABELS.get(int(rid), "unknown")
                regime_hist[name] = regime_hist.get(name, 0) + int(cnt)
            missing = max(0, n_bars - int(sub.shape[0]))
            regime_hist["unknown"] += missing
        else:
            regime_hist["unknown"] = n_bars

        # --- News in range ------------------------------------------
        news_items: list[dict] = []
        news_agg = {"count": 0, "mean": None, "max": None, "min": None}
        if self._news is not None and not self._news.empty:
            n = self._news
            sub = n[(n["published_at"] >= start) & (n["published_at"] <= end)]
            news_agg["count"] = int(len(sub))
            scores = sub["sent_score"].dropna().astype(float)
            if len(scores):
                news_agg["mean"] = float(scores.mean())
                news_agg["max"]  = float(scores.max())
                news_agg["min"]  = float(scores.min())
            # Top 5 most notable (largest |score|) for the panel.
            if not sub.empty:
                sub2 = sub.copy()
                sub2["abs_score"] = sub2["sent_score"].abs()
                sub2 = sub2.sort_values("abs_score", ascending=False).head(5)
                for _, r in sub2.iterrows():
                    tickers = r.get("tickers")
                    if not isinstance(tickers, (list, tuple, np.ndarray)):
                        tickers = []
                    news_items.append({
                        "published_at": pd.Timestamp(r["published_at"]).isoformat(),
                        "source": str(r.get("source", "")),
                        "title": str(r.get("title", "")),
                        "url": str(r.get("url", "")),
                        "tickers": list(tickers),
                        "sent_score": (float(r["sent_score"])
                                       if pd.notna(r.get("sent_score"))
                                       else None),
                    })

        # --- Predictions in range (from paper log) ------------------
        # `paper_predictions` is passed in from the route because it
        # lives on the orchestrator, not the InferenceService.
        p_up_hist = [0] * n_hist_bins
        n_pred = n_correct = n_traded = n_abstained = 0
        acc_confident = acc_uncertain = None
        conf_correct = conf_total = unc_correct = unc_total = 0
        if paper_predictions:
            for p in paper_predictions:
                ts_str = p.get("open_time")
                if not ts_str:
                    continue
                ts = pd.Timestamp(ts_str)
                if ts < start or ts > end:
                    continue
                n_pred += 1
                p_up = p.get("p_up")
                if isinstance(p_up, (int, float)) and 0.0 <= p_up <= 1.0:
                    b = min(n_hist_bins - 1, int(p_up * n_hist_bins))
                    p_up_hist[b] += 1
                if p.get("cp_singleton"):
                    n_traded += 1
                else:
                    n_abstained += 1
                if p.get("correct") is True:
                    n_correct += 1
                    if p.get("cp_singleton"):
                        conf_correct += 1
                    else:
                        unc_correct += 1
                if p.get("correct") is not None:
                    if p.get("cp_singleton"):
                        conf_total += 1
                    else:
                        unc_total += 1
            if conf_total:
                acc_confident = conf_correct / conf_total
            if unc_total:
                acc_uncertain = unc_correct / unc_total

        overall_acc = (n_correct / n_pred) if n_pred else None

        return {
            "start": pd.Timestamp(start).isoformat(),
            "end":   pd.Timestamp(end).isoformat(),
            "bars": n_bars,
            "ohlc": {
                "first_close": first_close,
                "last_close":  last_close,
                "high": high,
                "low":  low,
                "return_pct": ret_pct,
                "realised_vol": realised_vol,
            },
            "regime_histogram": regime_hist,
            "news": {
                **news_agg,
                "top": news_items,
            },
            "predictions": {
                "n": n_pred,
                "n_traded": n_traded,
                "n_abstained": n_abstained,
                "overall_accuracy": overall_acc,
                "confident_accuracy": acc_confident,
                "uncertain_accuracy": acc_uncertain,
                "p_up_histogram": p_up_hist,
                "p_up_bin_edges": [
                    round(i / n_hist_bins, 3) for i in range(n_hist_bins + 1)
                ],
            },
        }

    # ------------------------------------------------------------------
    def latest_embedding_for(self, open_time: pd.Timestamp) -> Optional[np.ndarray]:
        """Return the 128-d CNN embedding cached for ``open_time`` in
        this process, or None if the bar is missing / CNN unavailable.

        Used by the LivePersistor to snapshot the embedding produced by
        :meth:`append_live_bar` and write it to the embeddings parquet.
        """
        ts = pd.Timestamp(open_time)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if self._embeddings is None or ts not in self._embeddings.index:
            return None
        cols = [c for c in self._embeddings.columns if c.startswith("e")]
        return self._embeddings.loc[ts, cols].to_numpy(dtype=np.float32).copy()

    # ------------------------------------------------------------------
    def refit_fast_models(self) -> dict:
        """Rebuild HMM / FAISS / fusion in place from current on-disk data.

        Called by :class:`RetrainerScheduler` and by the manual "Refit now"
        button. Returns a status dict with per-step timings so the UI can
        show a summary.
        """
        import subprocess  # noqa: PLC0415
        import time         # noqa: PLC0415

        pair, interval = self.pair, self.interval
        timings: dict[str, float] = {}
        errors: list[str] = []

        # Persist any in-memory bars to disk before refitting.
        # (The LivePersistor's own flush is best-effort; the retrainer
        # asks for one via the orchestrator.)

        # HMM regimes.
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, "scripts/fit_hmm_regimes.py",
                 "--pairs", pair, "--interval", interval],
                cwd=str(PROJECT_ROOT), check=True,
            )
        except Exception as e:                                # noqa: BLE001
            errors.append(f"hmm: {e}")
        timings["hmm_s"] = round(time.time() - t0, 2)

        # FAISS index.
        t0 = time.time()
        try:
            subprocess.run(
                [sys.executable, "scripts/build_faiss_index.py",
                 "--pairs", pair, "--interval", interval],
                cwd=str(PROJECT_ROOT), check=True,
            )
        except Exception as e:                                # noqa: BLE001
            errors.append(f"faiss: {e}")
        timings["faiss_s"] = round(time.time() - t0, 2)

        # Reload the freshened artefacts + refit fusion.
        t0 = time.time()
        try:
            self._load_ohlcv()
            self._load_features_and_labels()
            self._load_embeddings()
            self._load_regimes()
            self._load_pattern_engine()
            self._fit_anchor_fusion()
        except Exception as e:                                # noqa: BLE001
            errors.append(f"reload: {e}")
        timings["reload_and_fusion_s"] = round(time.time() - t0, 2)

        return {
            "ok": not errors,
            "errors": errors,
            "timings_seconds": timings,
            "total_bars_now": len(self._ohlcv),
        }

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
