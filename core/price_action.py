"""
Price-action primitives: swing points, support/resistance, candlestick
patterns, breakout/retest, liquidity sweeps and false breakouts.

Look-ahead note: a swing at bar j with strength s needs s bars to its right,
so it only *exists* from bar j+s onwards. `swing_points` never marks a
pivot whose confirmation bars are not in the frame, so running it on
`df.iloc[:i+1]` is always causal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Swing:
    index: int          # position of the pivot bar
    confirmed_at: int   # position at which it became known
    price: float
    kind: str           # "high" | "low"


@dataclass
class Level:
    price: float
    touches: int
    kind: str           # "support" | "resistance"
    last_index: int


def swing_points(high: np.ndarray, low: np.ndarray, strength: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays of pivot highs / lows (only pivots with full right side)."""
    n = len(high)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    s = strength
    w = 2 * s + 1
    if n < w:
        return sh, sl
    hw = sliding_window_view(high, w)
    lw = sliding_window_view(low, w)
    centre_h = hw[:, s]
    centre_l = lw[:, s]
    left_h = hw[:, :s].max(axis=1)
    right_h = hw[:, s + 1:].max(axis=1)
    left_l = lw[:, :s].min(axis=1)
    right_l = lw[:, s + 1:].min(axis=1)
    sh[s:n - s] = (centre_h > left_h) & (centre_h >= right_h)
    sl[s:n - s] = (centre_l < left_l) & (centre_l <= right_l)
    return sh, sl


def list_swings(df: pd.DataFrame, strength: int) -> List[Swing]:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    sh, sl = swing_points(high, low, strength)
    out: List[Swing] = []
    for j in np.flatnonzero(sh | sl):
        if sh[j]:
            out.append(Swing(int(j), int(j + strength), float(high[j]), "high"))
        if sl[j]:
            out.append(Swing(int(j), int(j + strength), float(low[j]), "low"))
    out.sort(key=lambda x: (x.confirmed_at, x.index))
    return out


# ---------------------------------------------------------------------------
# candlestick patterns (vectorised, causal)
# ---------------------------------------------------------------------------
def candle_patterns(df: pd.DataFrame, atr: Optional[pd.Series] = None) -> pd.DataFrame:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    po, pc = o.shift(1), c.shift(1)
    min_body = 0.3 * atr if atr is not None else 0.0

    bull_engulf = (pc < po) & (c > o) & (c >= po) & (o <= pc) & (body > (pc - po).abs()) & (body >= min_body)
    bear_engulf = (pc > po) & (c < o) & (c <= po) & (o >= pc) & (body > (pc - po).abs()) & (body >= min_body)
    bull_pin = (lower_wick >= 2 * body) & (lower_wick >= 0.6 * rng) & ((c - l) / rng >= 0.6)
    bear_pin = (upper_wick >= 2 * body) & (upper_wick >= 0.6 * rng) & ((h - c) / rng >= 0.6)
    inside = (h <= h.shift(1)) & (l >= l.shift(1))
    strong_bull = (c > o) & (body >= 0.6 * rng)
    strong_bear = (c < o) & (body >= 0.6 * rng)
    return pd.DataFrame({
        "bull_engulf": bull_engulf.fillna(False),
        "bear_engulf": bear_engulf.fillna(False),
        "bull_pin": bull_pin.fillna(False),
        "bear_pin": bear_pin.fillna(False),
        "inside_bar": inside.fillna(False),
        "strong_bull": strong_bull.fillna(False),
        "strong_bear": strong_bear.fillna(False),
    }, index=df.index)


# ---------------------------------------------------------------------------
# support / resistance
# ---------------------------------------------------------------------------
def support_resistance(df: pd.DataFrame, swings: List[Swing], tol: float, max_levels: int = 8) -> List[Level]:
    """Cluster confirmed swing prices into levels; classify relative to last close."""
    if not swings:
        return []
    last_close = float(df["close"].iloc[-1])
    pts = sorted(swings, key=lambda s: s.price)
    clusters: List[List[Swing]] = [[pts[0]]]
    for s in pts[1:]:
        if abs(s.price - np.mean([x.price for x in clusters[-1]])) <= tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    levels = []
    for cl in clusters:
        price = float(np.mean([x.price for x in cl]))
        kind = "resistance" if price > last_close else "support"
        levels.append(Level(price, len(cl), kind, max(x.index for x in cl)))
    levels.sort(key=lambda lv: (-lv.touches, abs(lv.price - last_close)))
    return levels[:max_levels]


def nearest_level(levels: List[Level], price: float, kind: str) -> Optional[Level]:
    cands = [lv for lv in levels if lv.kind == kind]
    if not cands:
        return None
    return min(cands, key=lambda lv: abs(lv.price - price))


# ---------------------------------------------------------------------------
# breakouts / sweeps
# ---------------------------------------------------------------------------
@dataclass
class PriceEvent:
    kind: str           # breakout_retest | liquidity_sweep | false_breakout
    direction: str      # "bull" | "bear"
    level: float
    index: int


def detect_breakout_retest(df: pd.DataFrame, levels: List[Level], tol: float, lookback: int = 12) -> List[PriceEvent]:
    """Close beyond a level within `lookback` bars, then a retest that holds on the last bar."""
    events: List[PriceEvent] = []
    n = len(df)
    if n < 3:
        return events
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    last = n - 1
    start = max(1, n - lookback)
    for lv in levels:
        p = lv.price
        # bullish: crossed up, retest now from above
        for i in range(start, last):
            if c[i - 1] <= p < c[i]:
                if l[last] <= p + tol and c[last] > p and all(c[k] > p - tol for k in range(i, last + 1)):
                    events.append(PriceEvent("breakout_retest", "bull", p, last))
                break
        for i in range(start, last):
            if c[i - 1] >= p > c[i]:
                if h[last] >= p - tol and c[last] < p and all(c[k] < p + tol for k in range(i, last + 1)):
                    events.append(PriceEvent("breakout_retest", "bear", p, last))
                break
    return events


def detect_sweeps(df: pd.DataFrame, swings: List[Swing], lookback: int) -> List[PriceEvent]:
    """
    Liquidity sweep: a bar trades through a prior confirmed swing (stop run)
    and closes back on the original side. Only swings confirmed before the
    sweeping bar are considered.
    """
    events: List[PriceEvent] = []
    n = len(df)
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    for i in range(max(0, n - lookback), n):
        prior = [s for s in swings if s.confirmed_at < i and s.index < i]
        highs = [s for s in prior if s.kind == "high"][-4:]
        lows = [s for s in prior if s.kind == "low"][-4:]
        for s in highs:
            # untouched until bar i
            if np.max(h[s.index + 1:i], initial=-np.inf) > s.price:
                continue
            if h[i] > s.price and c[i] < s.price:
                events.append(PriceEvent("liquidity_sweep", "bear", s.price, i))
        for s in lows:
            if np.min(l[s.index + 1:i], initial=np.inf) < s.price:
                continue
            if l[i] < s.price and c[i] > s.price:
                events.append(PriceEvent("liquidity_sweep", "bull", s.price, i))
    return events


def detect_false_breakouts(df: pd.DataFrame, levels: List[Level], lookback: int = 4) -> List[PriceEvent]:
    """A close beyond a level that is reversed by a close back inside within `lookback` bars."""
    events: List[PriceEvent] = []
    n = len(df)
    c = df["close"].to_numpy()
    for lv in levels:
        p = lv.price
        for i in range(max(1, n - lookback), n):
            # bull trap: closed above, now back below
            if c[i - 1] > p and c[i] < p and np.any(c[max(0, i - lookback):i - 1] <= p):
                events.append(PriceEvent("false_breakout", "bear", p, i))
            if c[i - 1] < p and c[i] > p and np.any(c[max(0, i - lookback):i - 1] >= p):
                events.append(PriceEvent("false_breakout", "bull", p, i))
    return events
