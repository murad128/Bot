"""
Signal-time snapshot builders (L1.1b). PURE functions: no database, no MT5,
no network, no clock. Input is exactly what the engine evaluated (the
MarketContext frames - closed bars only - and the SMC snapshots), so a
snapshot can never contain information from after the signal. The same
builders can later be used by the backtester.

Nothing here computes a new indicator: indicator values are copied from the
already-prepared frames, SMC values are summarised from SMCSnapshot objects.
Data that is already stored on the signal row (score, components, context,
reasons, session) is not duplicated.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

FEATURE_VERSION = 1
NEWS_WINDOW = timedelta(hours=24)
MAX_ZONES = 3            # nearest order blocks / FVGs per side kept in the summary
MAX_EVENTS = 3           # most recent BOS/CHOCH and sweeps kept


def plain(v: Any) -> Any:
    """JSON-safe scalar: numpy/pandas -> python, NaN/inf -> None, timestamps -> ISO UTC."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if hasattr(v, "item") and not hasattr(v, "isoformat"):     # numpy scalar
        v = v.item()
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if hasattr(v, "isoformat"):
        return iso(v)
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(v)


def iso(ts) -> Optional[str]:
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _tf_order(tfs: Iterable) -> List:
    return sorted(tfs, key=lambda tf: tf.minutes)


# ---------------------------------------------------------------------------- indicators
def indicator_rows(frames: Mapping) -> Dict[str, Tuple[datetime, Dict[str, Any]]]:
    """{tf: (open time of the last closed bar, {column: value})} - OHLC and every existing column."""
    out: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
    for tf in _tf_order(frames):
        df = frames[tf]
        if df is None or len(df) == 0:
            continue
        row = df.iloc[-1]
        bar_time = df.index[-1]
        bar_time = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time
        out[tf.value] = (bar_time, {str(k): plain(v) for k, v in row.items()})
    return out


# ---------------------------------------------------------------------------- SMC
def _bars_ago(snap, index: Optional[int]) -> Optional[int]:
    return None if index is None else int(snap.n - 1 - index)


def _zone(snap, z, price: float, ob: bool) -> Dict[str, Any]:
    d = {"direction": int(z.direction), "top": plain(z.top), "bottom": plain(z.bottom),
         "distance": plain(abs((z.top + z.bottom) / 2.0 - price)),
         "age_bars": _bars_ago(snap, z.created_at)}
    if ob:
        d.update(has_fvg=bool(z.has_fvg), cause=z.cause, touches=int(z.touches),
                 displacement_atr=plain(z.displacement_atr))
    else:
        d.update(filled_frac=plain(z.filled_frac))
    return d


def _nearest(snap, zones, price: float, ob: bool) -> Dict[str, Any]:
    act = [z for z in zones if z.active]
    above = sorted((z for z in act if z.bottom > price), key=lambda z: z.bottom)[:MAX_ZONES]
    below = sorted((z for z in act if z.top < price), key=lambda z: -z.top)[:MAX_ZONES]
    inside = [z for z in act if z.bottom <= price <= z.top][:MAX_ZONES]
    return {"active": len(act), "bull": sum(1 for z in act if z.direction > 0),
            "bear": sum(1 for z in act if z.direction < 0),
            "inside": [_zone(snap, z, price, ob) for z in inside],
            "above": [_zone(snap, z, price, ob) for z in above],
            "below": [_zone(snap, z, price, ob) for z in below]}


def smc_summary(snap, price: float) -> Dict[str, Any]:
    """Compact, JSON-safe summary of one SMCSnapshot relative to the signal price."""
    unswept = [p for p in snap.pools if p.swept_at is None]
    buy = sorted((p for p in unswept if p.side == "buy_side" and p.price > price), key=lambda p: p.price)
    sell = sorted((p for p in unswept if p.side == "sell_side" and p.price < price), key=lambda p: -p.price)

    def pool(p):
        return None if p is None else {"price": plain(p.price), "count": int(p.count), "equal": bool(p.is_equal),
                                       "distance": plain(abs(p.price - price))}

    return {
        "bars": int(snap.n), "trend": int(snap.trend), "atr": plain(snap.atr), "last_close": plain(snap.last_close),
        "zone": snap.zone_label, "range_position": plain(snap.range_position),
        "range_high": plain(snap.range_high), "range_low": plain(snap.range_low),
        "structure": [{"kind": e.kind, "direction": int(e.direction), "level": plain(e.level),
                       "bars_ago": _bars_ago(snap, e.index)} for e in snap.events[-MAX_EVENTS:]],
        "order_blocks": _nearest(snap, snap.order_blocks, price, ob=True),
        "fvgs": _nearest(snap, snap.fvgs, price, ob=False),
        "liquidity": {"above": pool(buy[0] if buy else None), "below": pool(sell[0] if sell else None),
                      "unswept_above": len(buy), "unswept_below": len(sell)},
        "sweeps": [{"kind": e.kind, "direction": e.direction, "level": plain(e.level),
                    "bars_ago": _bars_ago(snap, e.index)} for e in snap.sweeps[-MAX_EVENTS:]],
    }


# ---------------------------------------------------------------------------- market / news
def market_summary(ctx, atr: Optional[float]) -> Dict[str, Any]:
    spec = ctx.spec
    spread = ctx.ask - ctx.bid
    atr_ok = atr is not None and math.isfinite(atr) and atr > 0
    return {
        "bid": plain(ctx.bid), "ask": plain(ctx.ask), "mid": plain((ctx.bid + ctx.ask) / 2.0),
        "spread": plain(spread), "spread_pips": plain(spec.to_pips(spread)),
        "spread_atr": plain(spread / atr) if atr_ok else None, "atr_m15": plain(atr),
        "digits": int(spec.digits), "pip_size": plain(spec.pip_size),
        "bars": {tf.value: int(len(ctx.frames[tf])) for tf in _tf_order(ctx.frames) if ctx.frames[tf] is not None},
        "last_bar_open": {tf.value: iso(ctx.frames[tf].index[-1]) for tf in _tf_order(ctx.frames)
                          if ctx.frames[tf] is not None and len(ctx.frames[tf])},
    }


def news_summary(events: Iterable, now: datetime, status: str) -> Dict[str, Any]:
    """Calendar events within +-24h of the signal. status: ok | unavailable | disabled."""
    rows = []
    for e in sorted(events, key=lambda e: e.time):
        rows.append({"time": iso(e.time), "minutes_from_signal": round((e.time - now).total_seconds() / 60.0, 1),
                     "currency": e.currency, "impact": e.impact, "title": e.title, "forecast": e.forecast,
                     "previous": e.previous, "actual": e.actual, "source": e.source, "key": e.key})
    return {"status": status, "window_hours": NEWS_WINDOW.total_seconds() / 3600.0, "events": rows}


# ---------------------------------------------------------------------------- assembly
def build_snapshot(ctx, smc_snaps: Mapping, expected_tfs: Iterable, events: Iterable, calendar_status: str,
                   atr: Optional[float]) -> Tuple[Dict[str, Any], Dict[str, Tuple[datetime, Dict[str, Any]]]]:
    """
    -> (snapshot fields, indicator rows). `capture_status` is 'complete' when every expected timeframe,
    its SMC summary and the calendar were available, otherwise 'partial' with the gaps in `missing`.
    """
    missing: List[str] = []
    for tf in _tf_order(expected_tfs):
        df = ctx.frames.get(tf)
        if df is None or len(df) == 0:
            missing.append(f"bars:{tf.value}")
        elif smc_snaps.get(tf) is None:
            missing.append(f"smc:{tf.value}")
    if calendar_status == "unavailable":
        missing.append("calendar")
    price = (ctx.bid + ctx.ask) / 2.0
    fields = {
        "symbol": ctx.symbol, "broker_symbol": ctx.spec.broker_symbol, "signal_time": ctx.now,
        "feature_version": FEATURE_VERSION, "capture_status": "partial" if missing else "complete",
        "missing": missing, "market": market_summary(ctx, atr),
        "smc": {tf.value: smc_summary(smc_snaps[tf], price) for tf in _tf_order(smc_snaps) if smc_snaps[tf] is not None},
        "news": news_summary(events, ctx.now, calendar_status),
    }
    return fields, indicator_rows(ctx.frames)
