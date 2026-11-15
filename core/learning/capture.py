"""
Signal-time capture (L1.1b): collects the inputs, builds the snapshot with the
pure builders in core.learning.snapshot and stores it.

Called from SignalBot.scan_symbol AFTER the signal was stored and sent, inside
an isolated try/except: a failure here never blocks, delays or changes a signal.
Read-only towards the engine: SMC snapshots are taken from the engine's cache
when they are for the same closed bar, otherwise recomputed with the same
analyzer WITHOUT touching the cache. No MT5 call, no Telegram call.
Duplicate-safe: one snapshot per signal id (insert-or-ignore).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Iterable, Optional

from sqlalchemy.exc import IntegrityError

from core.learning.snapshot import NEWS_WINDOW, build_snapshot

log = logging.getLogger(__name__)


def smc_for_capture(engine, symbol: str, frames) -> Dict:
    """{tf: SMCSnapshot or None} for every frame, without modifying the engine's cache."""
    out: Dict = {}
    for tf, df in frames.items():
        if df is None or len(df) == 0:
            out[tf] = None
            continue
        hit = engine._smc_cache.get((symbol, tf))
        if hit is not None and hit[0] == df.index[-1]:
            out[tf] = hit[1]
            continue
        try:
            out[tf] = engine._analyzers[tf].analyze(df)
        except Exception as exc:                                   # recorded as missing, never raised
            log.debug("Learning capture: SMC %s %s unavailable: %s", symbol, tf.value, exc)
            out[tf] = None
    return out


def store_snapshot(db, signal_id: int, fields: dict, rows: dict, fingerprint: Optional[str],
                   capture_ms: Optional[float]) -> bool:
    """Insert the snapshot and its indicator rows in one transaction. False if it already exists."""
    from database.learning_models import SignalIndicatorSnapshot, SignalSnapshot
    try:
        with db.session() as s:
            if s.get(SignalSnapshot, signal_id) is not None:
                return False
            s.add(SignalSnapshot(signal_id=signal_id, settings_fingerprint=fingerprint, capture_ms=capture_ms,
                                 **fields))
            for tf, (bar_time, values) in rows.items():
                s.add(SignalIndicatorSnapshot(signal_id=signal_id, timeframe=tf, bar_time=bar_time, values=values))
        return True
    except IntegrityError:                                          # concurrent duplicate
        return False


def capture_signal(db, engine, calendar, cfg, signal_id: int, candidate, ctx, fingerprint: Optional[str],
                   expected_tfs: Iterable) -> bool:
    t0 = time.perf_counter()
    smc = smc_for_capture(engine, ctx.symbol, ctx.frames)
    if cfg.news.calendar_provider == "none":
        status = "disabled"
        events = []
    else:
        status = "ok" if calendar.available else "unavailable"
        events = calendar.events_between(ctx.now - NEWS_WINDOW, ctx.now + NEWS_WINDOW, ctx.spec.currencies or None)
    fields, rows = build_snapshot(ctx, smc, expected_tfs, events, status, getattr(candidate, "atr", None))
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    stored = store_snapshot(db, signal_id, fields, rows, fingerprint, ms)
    if stored:
        log.info("Learning capture: signal %s snapshot %s (%d timeframes, %.0f ms)%s", signal_id,
                 fields["capture_status"], len(rows), ms,
                 f" missing={fields['missing']}" if fields["missing"] else "")
    return stored
