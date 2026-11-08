"""
Pure signal-outcome state machine (no I/O). Used by both the live tracker and
the backtester so live statistics and backtest statistics are computed the
same way.

Accounting model (documented in every report):
* The signal is treated as ONE trade with partial exits. With the default
  fractions 50/30/20 %, hitting TP1 closes 50 % at +1R, etc.
* After TP1 the stop moves to breakeven (configurable); after TP2 it moves to TP1.
* Realised R = sum(fraction_i * R_at_exit_i) - costs. A signal is a WIN if
  realised R > 0, a LOSS if < 0. TP1/TP2/TP3 are milestones of the same
  trade, never three separate wins.
* Bars are resolved pessimistically: if a bar touches both the stop and a
  target, the stop is assumed to be hit first.
* Prices: bars are BID prices (MT5 convention). Buys exit at bid; sells exit
  at ask = bid + spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from core.domain import Direction, Outcome, SignalStatus


@dataclass
class TradeEvent:
    kind: str          # TP1_HIT | TP2_HIT | TP3_HIT | SL_HIT | BE_HIT | EXPIRED | CANCELLED
    time: datetime
    price: float
    r: float           # R multiple of this exit price


@dataclass
class TradeState:
    direction: Direction
    entry: float
    stop_loss: float
    take_profits: List[float]
    close_fractions: List[float]
    opened_at: datetime
    max_hold: timedelta
    move_sl_to_be: bool = True
    exit_slippage: float = 0.0      # price units, applied against us on stop exits
    commission: float = 0.0         # round-turn cost in price units
    current_sl: Optional[float] = None
    tp_hits: int = 0
    status: SignalStatus = SignalStatus.ACTIVE
    realized_r: float = 0.0
    remaining: float = 1.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    events: List[TradeEvent] = field(default_factory=list)

    def __post_init__(self):
        if self.current_sl is None:
            self.current_sl = self.stop_loss
        if self.risk <= 0:
            raise ValueError("Stop loss must be on the losing side of entry")

    # ------------------------------------------------------------------ helpers
    @property
    def risk(self) -> float:
        return (self.entry - self.stop_loss) * self.direction.sign

    def r_of(self, price: float) -> float:
        return (price - self.entry) * self.direction.sign / self.risk

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def outcome(self) -> Outcome:
        if self.is_open:
            return Outcome.OPEN
        if self.status is SignalStatus.CANCELLED:
            return Outcome.CANCELLED
        if self.realized_r > 1e-9:
            return Outcome.WIN
        if self.realized_r < -1e-9:
            return Outcome.LOSS
        return Outcome.BREAKEVEN

    # ------------------------------------------------------------------ updates
    def _close_remaining(self, price: float, time: datetime, status: SignalStatus, kind: str) -> TradeEvent:
        r = self.r_of(price)
        self.realized_r += self.remaining * r
        self.realized_r -= self.commission / self.risk
        self.remaining = 0.0
        self.status = status
        self.closed_at = time
        self.exit_price = price
        ev = TradeEvent(kind, time, price, round(r, 3))
        self.events.append(ev)
        return ev

    def update_bar(self, high: float, low: float, close: float, time: datetime, spread: float = 0.0) -> List[TradeEvent]:
        """Process one bar (BID prices). Returns the events triggered."""
        if not self.is_open:
            return []
        out: List[TradeEvent] = []
        buy = self.direction is Direction.BUY
        fav = high if buy else low + spread
        adv = low if buy else high + spread
        self.mfe_r = max(self.mfe_r, self.r_of(fav))
        self.mae_r = min(self.mae_r, self.r_of(adv))

        def sl_crossed() -> bool:
            return adv <= self.current_sl if buy else adv >= self.current_sl

        # 1. stop first (pessimistic)
        if sl_crossed():
            out.append(self._stop(time))
            return out

        # 2. targets in order
        d = self.direction.sign
        while self.tp_hits < len(self.take_profits):
            tp = self.take_profits[self.tp_hits]
            if (fav - tp) * d < 0:
                break
            frac = self.close_fractions[self.tp_hits]
            r = self.r_of(tp)
            self.tp_hits += 1
            kind = f"TP{self.tp_hits}_HIT"
            if self.tp_hits == len(self.take_profits):
                out.append(self._close_remaining(tp, time, SignalStatus.TP3_HIT, kind))
                return out
            self.realized_r += frac * r
            self.remaining -= frac
            self.status = SignalStatus.TP1_HIT if self.tp_hits == 1 else SignalStatus.TP2_HIT
            ev = TradeEvent(kind, time, tp, round(r, 3))
            self.events.append(ev)
            out.append(ev)
            if self.tp_hits == 1 and self.move_sl_to_be:
                self.current_sl = self.entry
            elif self.tp_hits >= 2:
                self.current_sl = self.take_profits[self.tp_hits - 2]

        # 3. same-bar ambiguity after trailing: be conservative
        if out and sl_crossed():
            out.append(self._stop(time))
            return out

        # 4. time stop
        if time - self.opened_at >= self.max_hold:
            px = close if buy else close + spread
            out.append(self._close_remaining(px, time, SignalStatus.EXPIRED, "EXPIRED"))
        return out

    def _stop(self, time: datetime) -> TradeEvent:
        d = self.direction.sign
        px = self.current_sl - d * self.exit_slippage
        kind = "SL_HIT" if self.tp_hits == 0 else "BE_HIT" if abs(self.current_sl - self.entry) < 1e-12 else "SL_HIT"
        return self._close_remaining(px, time, SignalStatus.STOPPED, kind)

    def cancel(self, price: float, time: datetime) -> Optional[TradeEvent]:
        if not self.is_open:
            return None
        return self._close_remaining(price, time, SignalStatus.CANCELLED, "CANCELLED")
