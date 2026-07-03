"""Confidence-gated, vol-scaled paper trading.

Behaviour spec
--------------
At each closed candle ``c_t``:

  1. **Settle** any open position from the *previous* candle ``c_{t-1}``:
     P&L = position_dollars × (close_t − entry_price) / entry_price for
     a long; mirror sign for a short. Update balance, log the closed
     trade, attach a "hit" flag.

  2. **Decide** the next position from the model's prediction on ``c_t``.
     The rule is **confidence-gated**: open a position only when the
     conformal prediction set is a singleton — meaning the model has
     calibrated 90 % confidence in *one* direction. If the set has both
     classes, stay flat.

  3. **Size** the new position by inverse realised volatility:

         size_pct = clip(target_vol / current_vol_24h, 0.10, 1.00)
         dollars  = balance × size_pct

     The intent is constant **dollar-volatility** per trade: when BTC
     is calm, the bot bets more; when BTC is wild, it bets less. The
     0.10–1.00 clip prevents the size from collapsing to a no-op or
     blowing up beyond the available cash.

Validity tracking
-----------------
The state object exposes rolling stats over the *closed* trades:

  * ``win_rate``        — fraction of closed trades with P&L > 0
  * ``avg_return_pct``  — mean per-trade % return
  * ``sharpe_per_trade``— mean / std of per-trade returns
  * ``pnl_total_dollars`` and equity curve

A flat trade (no position taken) counts toward the audit log but not
toward win-rate / Sharpe — that's what the conformal abstention is *for*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal, Optional

import math

import pandas as pd

Direction = Literal["long", "short", "flat"]


@dataclass
class OpenPosition:
    direction: Direction               # "long" / "short"
    entry_time: pd.Timestamp           # candle close at which we entered
    entry_price: float
    dollars: float                     # notional sized at entry
    size_pct: float                    # fraction of balance committed
    conformal_singleton: bool          # True if entered on calibrated confidence
    p_up: float                        # P(up) at entry, for audit
    regime: Optional[int] = None


@dataclass
class ClosedTrade:
    direction: Direction
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    dollars: float
    size_pct: float
    pnl_dollars: float
    pnl_pct: float                     # P&L per dollar committed
    hit: bool                          # pnl > 0
    p_up: float
    regime: Optional[int] = None


@dataclass
class PaperTraderState:
    initial_balance: float = 100.0
    balance: float = 100.0
    is_running: bool = False
    started_at: Optional[pd.Timestamp] = None

    open_position: Optional[OpenPosition] = None
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    last_signal: Optional[dict] = None       # last prediction blob + decision
    last_candle_close: Optional[float] = None
    last_candle_time: Optional[pd.Timestamp] = None
    bars_seen: int = 0
    bars_traded: int = 0
    bars_abstained: int = 0

    # ------------------------------------------------------------------
    def reset(self, initial_balance: float = 100.0) -> None:
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.open_position = None
        self.closed_trades = []
        self.equity_curve = []
        self.last_signal = None
        self.last_candle_close = None
        self.last_candle_time = None
        self.bars_seen = 0
        self.bars_traded = 0
        self.bars_abstained = 0


# ---------------------------------------------------------------------------
# Pure functions over the state. Kept module-level so the orchestrator
# can call them synchronously inside an async task.
# ---------------------------------------------------------------------------
def _settle(state: PaperTraderState, exit_time: pd.Timestamp,
            exit_price: float) -> Optional[ClosedTrade]:
    """If a position is open, mark-to-market against ``exit_price`` and
    move it from open → closed. Returns the closed trade record (or None)."""
    pos = state.open_position
    if pos is None:
        return None

    if pos.direction == "long":
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
    elif pos.direction == "short":
        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
    else:
        return None  # "flat" should never reach here, but guard anyway

    pnl_dollars = pos.dollars * pnl_pct
    state.balance += pnl_dollars

    trade = ClosedTrade(
        direction=pos.direction,
        entry_time=pos.entry_time,
        exit_time=exit_time,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        dollars=pos.dollars,
        size_pct=pos.size_pct,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_pct,
        hit=pnl_dollars > 0,
        p_up=pos.p_up,
        regime=pos.regime,
    )
    state.closed_trades.append(trade)
    state.open_position = None
    return trade


def _decide(prediction: dict) -> Direction:
    """Apply the confidence-gated trading rule.

    A trade is only opened when the model's calibrated 90 % conformal
    set contains exactly one class. Both-classes-in-set ⇒ stay flat.
    """
    cp = prediction.get("conformal_set", {})
    down_in = bool(cp.get("down", False))
    up_in = bool(cp.get("up", False))

    if up_in and not down_in:
        return "long"
    if down_in and not up_in:
        return "short"
    return "flat"


def _vol_scaled_size_pct(
    vol_24h: float, vol_168h: float,
    target_pct: float = 1.00,
    min_pct: float = 0.10,
    max_pct: float = 1.00,
) -> float:
    """Inverse-volatility position size.

    target_vol = 1-week median realised vol (proxy: ``vol_168h``).
    size_pct = target_pct × (target_vol / current_vol_24h), clipped.

    All inputs are bar-level std of log returns; ratios are unit-free.
    NaNs and zeros default to ``target_pct``.
    """
    if not math.isfinite(vol_24h) or vol_24h <= 0:
        return target_pct
    if not math.isfinite(vol_168h) or vol_168h <= 0:
        ratio = 1.0
    else:
        ratio = vol_168h / vol_24h
    return max(min_pct, min(max_pct, target_pct * ratio))


def step(state: PaperTraderState, candle: dict, prediction: dict) -> dict:
    """Process one closed candle through the paper-trading state machine.

    ``candle`` must include ``open_time`` (UTC) and ``close``.
    ``prediction`` is the dict returned by
    :py:meth:`InferenceService.append_live_bar`. Returns a small dict
    describing what happened for logging / UI.
    """
    ts = pd.Timestamp(candle["open_time"]).tz_convert("UTC") \
        if pd.Timestamp(candle["open_time"]).tzinfo is not None \
        else pd.Timestamp(candle["open_time"], tz="UTC")
    px = float(candle["close"])

    state.bars_seen += 1
    state.last_candle_close = px
    state.last_candle_time = ts

    # 1) Settle previous position at this candle's close.
    closed = _settle(state, exit_time=ts, exit_price=px)

    # 2) Decide next direction from this candle's prediction.
    direction = _decide(prediction)

    # 3) If we want a trade, size it and open the position.
    opened: Optional[OpenPosition] = None
    if direction != "flat" and state.balance > 0.01:
        vol_24 = float(prediction.get("realised_vol_24h", float("nan")))
        vol_168 = float(prediction.get("realised_vol_168h", float("nan")))
        size_pct = _vol_scaled_size_pct(vol_24, vol_168)
        opened = OpenPosition(
            direction=direction,
            entry_time=ts, entry_price=px,
            dollars=state.balance * size_pct,
            size_pct=size_pct,
            conformal_singleton=True,
            p_up=float(prediction.get("p_up", 0.5)),
            regime=prediction.get("regime"),
        )
        state.open_position = opened
        state.bars_traded += 1
    else:
        state.open_position = None
        state.bars_abstained += 1

    state.equity_curve.append((ts, state.balance))
    state.last_signal = {
        "open_time": ts.isoformat(),
        "p_up": float(prediction.get("p_up", float("nan"))),
        "decision": direction,
        "regime": prediction.get("regime"),
        "regime_name": prediction.get("regime_name"),
        "conformal_set": prediction.get("conformal_set"),
        "close": px,
        "vol_24h": float(prediction.get("realised_vol_24h", float("nan"))),
        "vol_168h": float(prediction.get("realised_vol_168h", float("nan"))),
    }

    return {
        "closed_trade": closed,
        "opened_position": opened,
        "balance": state.balance,
        "decision": direction,
    }


# ---------------------------------------------------------------------------
def validity_stats(state: PaperTraderState, window: int = 50) -> dict:
    """Roll-up of recent trade quality for the dashboard."""
    trades = state.closed_trades[-window:] if window else state.closed_trades
    if not trades:
        return {
            "n_closed": 0, "win_rate": float("nan"),
            "avg_return_pct": float("nan"),
            "sharpe_per_trade": float("nan"),
            "pnl_total_dollars": float(state.balance - state.initial_balance),
            "verdict": "no trades yet",
        }
    returns = [t.pnl_pct for t in trades]
    hits = sum(1 for r in returns if r > 0)
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
    std = var ** 0.5
    sharpe = mean / std if std > 0 else 0.0
    total_pnl = state.balance - state.initial_balance
    win_rate = hits / n

    if n < 8:
        verdict = "warming up"
    elif win_rate > 0.55 and total_pnl > 0:
        verdict = "model is working"
    elif win_rate < 0.45 and total_pnl < 0:
        verdict = "model is losing"
    else:
        verdict = "inconclusive"

    return {
        "n_closed": n,
        "win_rate": win_rate,
        "avg_return_pct": mean,
        "sharpe_per_trade": sharpe,
        "pnl_total_dollars": float(total_pnl),
        "verdict": verdict,
    }
