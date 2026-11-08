"""
Persistent signal tracking: stores every signal and advances its TradeState
with new closed M1 bars (or any bar series) as they arrive.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd

from config import RiskConfig
from core.domain import Direction, SignalCandidate, SignalStatus
from core.trade_state import TradeEvent, TradeState
from database.database import Database, _aware
from database.models import SignalRecord

log = logging.getLogger(__name__)
signals_log = logging.getLogger("signals")


def record_from_candidate(c: SignalCandidate, source: str = "live") -> SignalRecord:
    p = c.plan
    tps = list(p.take_profits) + [p.take_profits[-1]] * (3 - len(p.take_profits))
    return SignalRecord(
        dedup_key=f"{source}:{c.dedup_key}", source=source, symbol=c.symbol, direction=c.direction.value,
        setup_type=c.setup_type, entry_timeframe=c.entry_timeframe, confirm_timeframe=c.confirm_timeframe,
        session=c.session, session_bucket=c.context.get("session_bucket", ""), trend_label=c.trend_label,
        score=c.score, grade=c.grade, entry=p.entry, entry_low=p.entry_low, entry_high=p.entry_high,
        stop_loss=p.stop_loss, current_sl=p.stop_loss, tp1=tps[0], tp2=tps[1], tp3=tps[2], risk=p.risk,
        rr=p.primary_rr, atr=c.atr, spread=c.spread, status=SignalStatus.ACTIVE.value, outcome="OPEN",
        created_at=c.created_at, last_checked_at=c.created_at, reasons=list(c.reasons),
        warnings=list(c.warnings), components={x.name: round(x.points, 2) for x in c.components},
        context=c.context, events=[],
    )


class SignalTracker:
    def __init__(self, db: Database, risk_cfg: RiskConfig):
        self.db = db
        self.cfg = risk_cfg

    # ------------------------------------------------------------------ create
    def register(self, cand: SignalCandidate, source: str = "live") -> Optional[SignalRecord]:
        rec = self.db.add_signal(record_from_candidate(cand, source))
        if rec is not None:
            signals_log.info("NEW %s %s %s entry=%s sl=%s tp=%s score=%.1f setup=%s",
                             rec.id, rec.symbol, rec.direction, rec.entry, rec.stop_loss,
                             [rec.tp1, rec.tp2, rec.tp3], rec.score, rec.setup_type)
        return rec

    # ------------------------------------------------------------------ state mapping
    def state_of(self, rec: SignalRecord) -> TradeState:
        st = TradeState(
            direction=Direction(rec.direction), entry=rec.entry, stop_loss=rec.stop_loss,
            take_profits=[rec.tp1, rec.tp2, rec.tp3], close_fractions=list(self.cfg.tp_close_fractions),
            opened_at=_aware(rec.created_at), max_hold=timedelta(hours=self.cfg.max_holding_hours),
            move_sl_to_be=self.cfg.move_sl_to_be_after_tp1,
        )
        st.current_sl = rec.current_sl
        st.tp_hits = rec.tp_hits
        st.status = SignalStatus(rec.status)
        st.realized_r = rec.realized_r
        st.remaining = rec.remaining
        st.mfe_r = rec.mfe_r
        st.mae_r = rec.mae_r
        return st

    @staticmethod
    def apply_state(rec: SignalRecord, st: TradeState, events: List[TradeEvent]) -> None:
        rec.current_sl = st.current_sl
        rec.tp_hits = st.tp_hits
        rec.status = st.status.value
        rec.realized_r = round(st.realized_r, 4)
        rec.remaining = st.remaining
        rec.mfe_r = round(st.mfe_r, 3)
        rec.mae_r = round(st.mae_r, 3)
        rec.outcome = st.outcome.value
        rec.closed_at = st.closed_at
        rec.exit_price = st.exit_price
        rec.events = list(rec.events or []) + [
            {"kind": e.kind, "time": e.time.isoformat(), "price": e.price, "r": e.r} for e in events]

    # ------------------------------------------------------------------ updates
    def update_with_bars(self, rec: SignalRecord, bars: pd.DataFrame, spread: float,
                         bar_minutes: int = 1) -> List[TradeEvent]:
        """
        Feed closed bars (BID OHLC, UTC open-time index) that closed after the
        last check. Bars that opened before the signal time are skipped.
        """
        last = _aware(rec.last_checked_at) or _aware(rec.created_at)
        created = _aware(rec.created_at)
        st = self.state_of(rec)
        events: List[TradeEvent] = []
        step = timedelta(minutes=bar_minutes)
        for ts, row in bars.iterrows():
            t_open = ts.to_pydatetime()
            t_close = t_open + step
            if t_open < created or t_close <= last:
                continue
            sp = float(row["spread_price"]) if "spread_price" in row and not pd.isna(row["spread_price"]) else spread
            evs = st.update_bar(float(row["high"]), float(row["low"]), float(row["close"]), t_close, sp)
            events.extend(evs)
            last = t_close
            if not st.is_open:
                break
        rec.last_checked_at = last
        self.apply_state(rec, st, events)
        self.db.save_signal(rec)
        for e in events:
            signals_log.info("EVENT %s %s %s price=%s r=%.2f total_r=%.2f", rec.id, rec.symbol, e.kind,
                             e.price, e.r, rec.realized_r)
        return events

    def cancel(self, rec: SignalRecord, price: float, now: datetime, reason: str) -> Optional[TradeEvent]:
        """Cancel (only allowed before TP1). Closes at `price`."""
        st = self.state_of(rec)
        if st.tp_hits > 0 or not st.is_open:
            return None
        ev = st.cancel(price, now)
        if ev:
            self.apply_state(rec, st, [ev])
            rec.review = dict(rec.review or {}, cancel_reason=reason)
            self.db.save_signal(rec)
            signals_log.info("CANCEL %s %s reason=%s r=%.2f", rec.id, rec.symbol, reason, ev.r)
        return ev
