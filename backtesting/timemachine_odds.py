"""
Offline "Time Machine" odds for past signals (pure computation: no files, network, MT5 or Telegram).

For one signal, on cached M5 bars (UTC, BID prices, optional `spread_price`):

  A. actual path    - the signal's own Entry / SL / TP from its entry bar forward for the horizon
  B. history replay - the signal's SL and TP DISTANCES applied to the entry price at the same UTC time of day
                      on each earlier day of the lookback (windows end before the signal: no look-ahead)
  C. simulation     - N random paths for the horizon, each M5 bar drawn from the same time of day (+- window
                      bars) in the lookback, as moves relative to the previous close; fixed seed

Price rules are the bot's own tracker rules (core/trade_state.py): bars are BID; a BUY is stopped by the bid
low and reaches its target at the bid high; a SELL is stopped by the ASK high (high + spread) and reaches its
target at the ask low. A bar that touches both SL and target counts as the STOP (conservative) and is also
counted as "ambiguous". The horizon is clock time from the entry (the bot expires positions by clock time).

Break-even: a target k R away needs a win share TP / (TP + SL) of 1 / (1 + k). Simulated odds are a model
built from past bars, not a prediction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.sessions import is_fx_market_open

BAR = pd.Timedelta(minutes=5)
SLOTS = 288                                    # 5-minute slots in a day


# ---------------------------------------------------------------------------- basics
def levels(direction: str, entry: float, sl_dist: float, tp_dist: float) -> Tuple[float, float]:
    """(stop, target) for an entry and the original distances."""
    if direction == "BUY":
        return entry - sl_dist, entry + tp_dist
    return entry + sl_dist, entry - tp_dist


def break_even(k: float) -> Optional[float]:
    return 1.0 / (1.0 + k) if k and k > 0 else None


def wilson(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n <= 0:
        return None
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def verdict(target: int, stop: int, p_be: Optional[float]) -> Dict:
    """Win share among outcomes that hit a side, compared with break-even."""
    sided = target + stop
    if sided == 0 or p_be is None:
        return {"win_share": None, "break_even": p_be, "diff": None, "label": "no side was hit: no verdict"}
    share = target / sided
    diff = share - p_be
    a = abs(diff)
    side = "target" if diff > 0 else "stop"
    label = ("close to break-even" if a < 0.05 else f"leans toward the {side}" if a < 0.15
             else f"strongly toward the {side}")
    return {"win_share": share, "break_even": p_be, "diff": diff, "label": label}


def prepare_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated M5 bars with a float spread column (NaN when unknown)."""
    out = df[~df.index.duplicated(keep="last")].sort_index()
    if "spread_price" not in out:
        out = out.assign(spread_price=np.nan)
    return out[["open", "high", "low", "close", "spread_price"]].astype(float)


def entry_index(bars: pd.DataFrame, when: datetime) -> Optional[int]:
    """The last bar CLOSED at or before `when` (the bar the bot evaluated)."""
    t = pd.Timestamp(when)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    i = int(bars.index.searchsorted(t - BAR, side="right")) - 1
    return i if i >= 0 else None


# ---------------------------------------------------------------------------- one path on real bars
@dataclass
class Touch:
    result: str                    # target | stop | neither | no_data
    ambiguous: bool = False
    bars: int = 0
    exit_time: Optional[str] = None
    complete: bool = True          # False when the data ended before the horizon


def first_touch(bars: pd.DataFrame, start: int, direction: str, stop: float, target: float, end_time: pd.Timestamp,
                fallback_spread: float) -> Touch:
    """Walk real bars from index `start` until a side is hit or the horizon ends (bar close > end_time)."""
    if start >= len(bars):
        return Touch("no_data", complete=False)
    seg = bars.iloc[start:]
    seg = seg[seg.index + BAR <= end_time]
    complete = bool(bars.index[-1] + BAR >= end_time)      # the data reaches the end of the horizon
    if seg.empty:
        return Touch("no_data", complete=False)
    sp = seg["spread_price"].to_numpy()
    sp = np.where(np.isfinite(sp), sp, fallback_spread)
    hi, lo = seg["high"].to_numpy(), seg["low"].to_numpy()
    if direction == "BUY":
        hit_sl, hit_tp = lo <= stop, hi >= target
    else:
        hit_sl, hit_tp = hi + sp >= stop, lo + sp <= target
    any_hit = hit_sl | hit_tp
    if not any_hit.any():
        return Touch("neither", bars=len(seg), complete=bool(complete))
    k = int(np.argmax(any_hit))
    both = bool(hit_sl[k] and hit_tp[k])
    return Touch("stop" if hit_sl[k] else "target", ambiguous=both, bars=k + 1,
                 exit_time=(seg.index[k] + BAR).isoformat())


# ---------------------------------------------------------------------------- A. actual path
def actual_path(bars: pd.DataFrame, sig: Dict, horizon: timedelta, target_key: str = "tp1") -> Touch:
    i = entry_index(bars, sig["created_at"])
    if i is None:
        return Touch("no_data", complete=False)
    end = bars.index[i] + BAR + pd.Timedelta(horizon)
    return first_touch(bars, i + 1, sig["direction"], sig["stop_loss"], sig[target_key], end,
                       float(sig.get("spread") or 0.0))


# ---------------------------------------------------------------------------- B. history replay
@dataclass
class Counts:
    target: int = 0
    stop: int = 0
    neither: int = 0
    ambiguous: int = 0
    skipped: Dict[str, int] = field(default_factory=dict)
    first_day: Optional[str] = None
    last_day: Optional[str] = None

    @property
    def n(self) -> int:
        return self.target + self.stop + self.neither

    def add(self, t: Touch) -> None:
        setattr(self, t.result, getattr(self, t.result) + 1)
        self.ambiguous += int(t.ambiguous)

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1


def history_replay(bars: pd.DataFrame, sig: Dict, horizon: timedelta, lookback_days: int,
                   target_key: str = "tp1") -> Counts:
    """The signal's SL/TP distances, re-applied to each earlier day's entry at the same UTC time of day."""
    c = Counts()
    i = entry_index(bars, sig["created_at"])
    if i is None:
        c.skip("signal outside the data")
        return c
    entry_open = bars.index[i]
    signal_entry_close = entry_open + BAR
    sl_dist = abs(sig["entry"] - sig["stop_loss"])
    tp_dist = abs(sig[target_key] - sig["entry"])
    fallback = float(sig.get("spread") or 0.0)
    pos = {t: n for n, t in enumerate(bars.index)}
    for d in range(1, lookback_days + 1):
        t_open = entry_open - pd.Timedelta(days=d)
        j = pos.get(t_open)
        if j is None:
            c.skip("no bar at that time (market closed or missing data)")
            continue
        end = t_open + BAR + pd.Timedelta(horizon)
        if end > signal_entry_close:
            c.skip("window would overlap the signal")
            continue
        if bars.index[-1] + BAR < end:
            c.skip("not enough data after the entry")
            continue
        entry = float(bars["close"].iloc[j])
        stop, target = levels(sig["direction"], entry, sl_dist, tp_dist)
        t = first_touch(bars, j + 1, sig["direction"], stop, target, end, fallback)
        if t.result == "no_data":
            c.skip("no bars in the window")
            continue
        c.add(t)
        day = t_open.date().isoformat()
        c.first_day = day if c.first_day is None or day < c.first_day else c.first_day
        c.last_day = day if c.last_day is None or day > c.last_day else c.last_day
    return c


# ---------------------------------------------------------------------------- C. simulation
def build_pool(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Dict[int, np.ndarray]:
    """Per 5-minute UTC slot: rows of (open, high, low, close) / previous close and spread / previous close,
    from bars in [start, end) whose previous bar is exactly 5 minutes earlier (gaps never make a move)."""
    seg = bars[(bars.index >= start) & (bars.index < end)]
    if len(seg) < 2:
        return {}
    idx = seg.index
    contiguous = (idx[1:] - idx[:-1]) == BAR
    prev_close = seg["close"].to_numpy()[:-1]
    cur = seg.iloc[1:]
    ratios = np.column_stack([cur["open"].to_numpy() / prev_close, cur["high"].to_numpy() / prev_close,
                              cur["low"].to_numpy() / prev_close, cur["close"].to_numpy() / prev_close,
                              cur["spread_price"].to_numpy() / prev_close])[contiguous]
    slots = ((idx[1:].hour * 60 + idx[1:].minute) // 5).to_numpy()[contiguous]
    return {int(s): ratios[slots == s] for s in np.unique(slots)}


@dataclass
class SimResult:
    replays: int
    seed: int
    target: int = 0
    stop: int = 0
    neither: int = 0
    ambiguous: int = 0
    steps: int = 0
    skipped_steps: int = 0
    pool_bars: int = 0

    def p(self, k: str) -> float:
        return getattr(self, k) / self.replays if self.replays else 0.0


def simulate(bars: pd.DataFrame, sig: Dict, horizon: timedelta, lookback_days: int, replays: int, seed: int,
             window: int = 2, target_key: str = "tp1") -> Optional[SimResult]:
    i = entry_index(bars, sig["created_at"])
    if i is None:
        return None
    entry_close = bars.index[i] + BAR
    pool = build_pool(bars, entry_close - pd.Timedelta(days=lookback_days), entry_close)  # strictly before entry
    res = SimResult(replays, seed, pool_bars=int(sum(len(v) for v in pool.values())))
    if not pool:
        return res
    rng = np.random.default_rng(seed)
    direction = sig["direction"]
    entry = float(sig["entry"])
    stop, target = levels(direction, entry, abs(entry - sig["stop_loss"]), abs(sig[target_key] - entry))
    fallback_ratio = float(sig.get("spread") or 0.0) / entry
    price = np.full(replays, entry)
    alive = np.ones(replays, dtype=bool)
    outcome = np.zeros(replays, dtype=np.int8)            # 0 neither, 1 target, 2 stop
    ambiguous = np.zeros(replays, dtype=bool)
    t = entry_close
    end = entry_close + pd.Timedelta(horizon)
    while t + BAR <= end and alive.any():
        res.steps += 1
        if not is_fx_market_open(t.to_pydatetime()):
            res.skipped_steps += 1
            t += BAR
            continue
        slot = (t.hour * 60 + t.minute) // 5
        rows = [pool[s % SLOTS] for s in range(slot - window, slot + window + 1) if (s % SLOTS) in pool]
        if not rows:                                       # no history at this time of day: market closed
            res.skipped_steps += 1
            t += BAR
            continue
        cand = np.concatenate(rows)
        k = np.flatnonzero(alive)
        draw = cand[rng.integers(0, len(cand), size=len(k))]
        p0 = price[k]
        hi, lo, cl = p0 * draw[:, 1], p0 * draw[:, 2], p0 * draw[:, 3]
        sp = np.where(np.isfinite(draw[:, 4]), p0 * draw[:, 4], p0 * fallback_ratio)
        if direction == "BUY":
            hit_sl, hit_tp = lo <= stop, hi >= target
        else:
            hit_sl, hit_tp = hi + sp >= stop, lo + sp <= target
        done = hit_sl | hit_tp
        outcome[k[hit_sl]] = 2
        outcome[k[hit_tp & ~hit_sl]] = 1
        ambiguous[k[hit_sl & hit_tp]] = True
        alive[k[done]] = False
        price[k] = cl
        t += BAR
    res.target = int((outcome == 1).sum())
    res.stop = int((outcome == 2).sum())
    res.neither = int((outcome == 0).sum())
    res.ambiguous = int(ambiguous.sum())
    return res
