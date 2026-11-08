"""
Smart Money Concepts analysis.

`SMCAnalyzer.analyze(df)` walks the frame bar by bar using only information
available at each bar (swings are added at their confirmation bar), and
returns an `SMCSnapshot` describing the state at the LAST bar:

* market structure (BOS / CHOCH events, current trend)
* order blocks with quality attributes (displacement, FVG, touches, invalidation)
* fair value gaps with fill tracking
* liquidity pools (swing highs/lows, equal highs/lows) and sweeps
* premium / discount position inside the current dealing range
* inducement (internal liquidity taken before a zone is tapped)

Definitions used (documented so the behaviour is auditable):
  BOS   - close beyond the last confirmed swing in the direction of the current trend.
  CHOCH - first close beyond the last confirmed swing AGAINST the current trend.
  Bullish OB - last bearish candle at/just before the low of the leg that produced a
               bullish BOS/CHOCH. Zone = [low, high] of that candle.
  FVG   - 3-candle imbalance: low[i] > high[i-2] (bull) / high[i] < low[i-2] (bear),
          at least `fvg_min_atr` * ATR in size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.price_action import Swing, list_swings, detect_sweeps, PriceEvent


@dataclass
class StructureEvent:
    kind: str        # BOS | CHOCH
    direction: int   # +1 bull, -1 bear
    level: float
    index: int


@dataclass
class OrderBlock:
    direction: int
    top: float
    bottom: float
    candle_index: int
    created_at: int
    displacement_atr: float
    has_fvg: bool
    cause: str                      # BOS | CHOCH
    touches: int = 0
    first_touch_at: Optional[int] = None
    last_touch_at: Optional[int] = None
    invalidated_at: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.invalidated_at is None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class FairValueGap:
    direction: int
    top: float
    bottom: float
    created_at: int
    filled_frac: float = 0.0
    first_touch_at: Optional[int] = None
    last_touch_at: Optional[int] = None
    filled_at: Optional[int] = None
    invalidated_at: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.filled_at is None and self.invalidated_at is None


@dataclass
class LiquidityPool:
    side: str            # "buy_side" (above highs) | "sell_side" (below lows)
    price: float
    count: int           # >=2 means equal highs/lows
    last_index: int
    swept_at: Optional[int] = None

    @property
    def is_equal(self) -> bool:
        return self.count >= 2


@dataclass
class SMCAssessment:
    direction: int
    score: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    zone: Optional[dict] = None
    sweep: Optional[PriceEvent] = None
    recent_choch: bool = False
    recent_bos: bool = False
    in_correct_zone: bool = False
    inducement: bool = False
    invalidation_price: Optional[float] = None


@dataclass
class SMCSnapshot:
    n: int
    last_close: float
    atr: float
    trend: int
    events: List[StructureEvent]
    order_blocks: List[OrderBlock]
    fvgs: List[FairValueGap]
    pools: List[LiquidityPool]
    sweeps: List[PriceEvent]
    swings: List[Swing]
    internal_swings: List[Swing]
    range_high: Optional[float]
    range_low: Optional[float]
    lows: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    highs: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------ helpers
    @property
    def last_event(self) -> Optional[StructureEvent]:
        return self.events[-1] if self.events else None

    @property
    def range_position(self) -> Optional[float]:
        """0 = range low, 1 = range high."""
        if self.range_high is None or self.range_low is None or self.range_high <= self.range_low:
            return None
        return (self.last_close - self.range_low) / (self.range_high - self.range_low)

    @property
    def zone_label(self) -> str:
        p = self.range_position
        if p is None:
            return "undefined"
        if p < 0.45:
            return "discount"
        if p > 0.55:
            return "premium"
        return "equilibrium"

    def recent_events(self, direction: int, within: int, kind: Optional[str] = None) -> List[StructureEvent]:
        return [e for e in self.events
                if e.direction == direction and e.index >= self.n - within and (kind is None or e.kind == kind)]

    def nearest_opposing_liquidity(self, direction: int, price: float) -> Optional[float]:
        """Closest un-swept pool in the trade direction (= natural target / obstacle)."""
        if direction > 0:
            cands = [p.price for p in self.pools if p.side == "buy_side" and p.swept_at is None and p.price > price]
            return min(cands) if cands else None
        cands = [p.price for p in self.pools if p.side == "sell_side" and p.swept_at is None and p.price < price]
        return max(cands) if cands else None

    def liquidity_targets(self, direction: int, price: float) -> List[float]:
        if direction > 0:
            return sorted(p.price for p in self.pools if p.side == "buy_side" and p.swept_at is None and p.price > price)
        return sorted((p.price for p in self.pools if p.side == "sell_side" and p.swept_at is None and p.price < price),
                      reverse=True)

    # ------------------------------------------------------------------ assessment
    def assess(self, direction: int, lookback: int, min_displacement: float) -> SMCAssessment:
        a = SMCAssessment(direction=direction, score=0.0)
        name = "bullish" if direction > 0 else "bearish"
        start = self.n - lookback

        # 1. structure
        if self.trend == direction:
            a.score += 0.25
            a.reasons.append(f"{name.capitalize()} market structure")
        chochs = self.recent_events(direction, lookback * 2, "CHOCH")
        boss = self.recent_events(direction, lookback * 2, "BOS")
        if chochs:
            a.recent_choch = True
            a.reasons.append(f"{name.capitalize()} CHOCH confirmation")
            if self.trend != direction:
                a.score += 0.15
        if boss:
            a.recent_bos = True
            a.reasons.append(f"{name.capitalize()} BOS")

        # 2. zone (OB / FVG) tapped recently
        best_zone = None
        best_q = 0.0
        px, max_dist = self.last_close, 0.5 * self.atr

        def near(top: float, bottom: float) -> bool:
            # price must still be at the zone (no chasing after the reaction already happened)
            return (px - top) <= max_dist if direction > 0 else (bottom - px) <= max_dist

        for ob in self.order_blocks:
            if ob.direction != direction or not ob.active or ob.last_touch_at is None or ob.last_touch_at < start:
                continue
            if not near(ob.top, ob.bottom):
                continue
            q = 0.5
            if ob.displacement_atr >= min_displacement * 1.5:
                q += 0.2
            elif ob.displacement_atr < min_displacement:
                q -= 0.25
            if ob.has_fvg:
                q += 0.2
            if ob.touches <= 2:
                q += 0.1
            else:
                q -= 0.2
            if ob.cause == "CHOCH":
                q += 0.05
            if q > best_q:
                best_q, best_zone = q, {"type": "OB", "top": ob.top, "bottom": ob.bottom,
                                        "touches": ob.touches, "displacement_atr": round(ob.displacement_atr, 2),
                                        "has_fvg": ob.has_fvg, "created_at": ob.created_at}
        for g in self.fvgs:
            if g.direction != direction or g.invalidated_at is not None or g.last_touch_at is None or g.last_touch_at < start:
                continue
            if not near(g.top, g.bottom):
                continue
            q = 0.45 if g.filled_at is None else 0.25
            if q > best_q:
                best_q, best_zone = q, {"type": "FVG", "top": g.top, "bottom": g.bottom,
                                        "filled_frac": round(g.filled_frac, 2), "created_at": g.created_at}
        if best_zone is not None:
            a.zone = best_zone
            a.score += 0.30 * min(1.0, best_q)
            label = "order block" if best_zone["type"] == "OB" else "fair value gap"
            if best_q >= 0.7:
                a.reasons.append(f"Retest of high-quality {name} {label}")
            elif best_q >= 0.45:
                a.reasons.append(f"Retest of {name} {label}")
            else:
                a.warnings.append(f"Weak {name} {label} (mitigated / low displacement)")

        # 3. premium / discount
        pos = self.range_position
        if pos is not None:
            ok = pos <= 0.5 if direction > 0 else pos >= 0.5
            if ok:
                a.in_correct_zone = True
                a.score += 0.15
                ote = (0.21 <= pos <= 0.38) if direction > 0 else (0.62 <= pos <= 0.79)
                a.reasons.append(("Price in discount" if direction > 0 else "Price in premium")
                                 + (" (OTE zone)" if ote else ""))
            else:
                a.warnings.append("Price in premium for a buy" if direction > 0 else "Price in discount for a sell")

        # 4. liquidity sweep in favour
        want = "bull" if direction > 0 else "bear"
        sweeps = [s for s in self.sweeps if s.direction == want and s.index >= start]
        if sweeps:
            a.sweep = sweeps[-1]
            a.score += 0.20
            side = "sell-side" if direction > 0 else "buy-side"
            a.reasons.append(f"{side.capitalize()} liquidity sweep")

        # 5. inducement
        if a.zone is not None and self._inducement_taken(direction, a.zone):
            a.inducement = True
            a.score += 0.10
            a.reasons.append("Inducement liquidity taken before zone tap")

        # invalidation price = beyond the protective structure
        cands = []
        if a.zone:
            cands.append(a.zone["bottom"] if direction > 0 else a.zone["top"])
        if a.sweep is not None and a.sweep.index < len(self.lows):
            cands.append(float(self.lows[a.sweep.index]) if direction > 0 else float(self.highs[a.sweep.index]))
        recent = self.lows[-lookback:] if direction > 0 else self.highs[-lookback:]
        if len(recent):
            cands.append(float(recent.min()) if direction > 0 else float(recent.max()))
        if cands:
            a.invalidation_price = min(cands) if direction > 0 else max(cands)
        a.score = min(1.0, a.score)
        return a

    def _inducement_taken(self, direction: int, zone: dict) -> bool:
        created = zone.get("created_at", 0)
        for s in self.internal_swings:
            if s.confirmed_at <= created or s.confirmed_at >= self.n - 1:
                continue
            if direction > 0 and s.kind == "low" and s.price > zone["top"]:
                after = self.lows[s.confirmed_at + 1:]
                if len(after) and after.min() < s.price:
                    return True
            if direction < 0 and s.kind == "high" and s.price < zone["bottom"]:
                after = self.highs[s.confirmed_at + 1:]
                if len(after) and after.max() > s.price:
                    return True
        return False


class SMCAnalyzer:
    def __init__(self, swing_strength: int = 3, internal_strength: int = 1, fvg_min_atr: float = 0.15,
                 ob_min_displacement_atr: float = 1.0, equal_tol_atr: float = 0.10,
                 sweep_lookback: int = 6, window: int = 300):
        self.swing_strength = swing_strength
        self.internal_strength = internal_strength
        self.fvg_min_atr = fvg_min_atr
        self.min_disp = ob_min_displacement_atr
        self.equal_tol = equal_tol_atr
        self.sweep_lookback = sweep_lookback
        self.window = window

    def analyze(self, df: pd.DataFrame) -> SMCSnapshot:
        df = df.tail(self.window)
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(df)
        if "atr" in df.columns:
            atr = df["atr"].to_numpy(dtype=float)
        else:
            atr = pd.Series(h - l).rolling(14, min_periods=1).mean().to_numpy()
        fallback = np.nanmean(h - l) if n else 0.0
        atr = np.where(np.isnan(atr) | (atr <= 0), fallback, atr)

        swings = list_swings(df, self.swing_strength)
        internal = list_swings(df, self.internal_strength)
        by_conf: Dict[int, List[Swing]] = {}
        for s in swings:
            by_conf.setdefault(s.confirmed_at, []).append(s)

        trend = 0
        last_sh: Optional[Swing] = None
        last_sl: Optional[Swing] = None
        sh_taken = sl_taken = False
        events: List[StructureEvent] = []
        obs: List[OrderBlock] = []
        fvgs: List[FairValueGap] = []

        active_obs: List[OrderBlock] = []
        active_fvgs: List[FairValueGap] = []
        for i in range(n):
            # -- maintain existing zones with this bar (zones created before i)
            if active_obs:
                keep = []
                for ob in active_obs:
                    if ob.created_at < i:
                        if ob.direction > 0:
                            if l[i] <= ob.top:
                                self._touch(ob, i)
                            if c[i] < ob.bottom:
                                ob.invalidated_at = i
                        else:
                            if h[i] >= ob.bottom:
                                self._touch(ob, i)
                            if c[i] > ob.top:
                                ob.invalidated_at = i
                    if ob.invalidated_at is None:
                        keep.append(ob)
                active_obs = keep
            if active_fvgs:
                keep = []
                for g in active_fvgs:
                    if g.created_at < i:
                        size = g.top - g.bottom
                        if g.direction > 0 and l[i] < g.top:
                            if g.first_touch_at is None:
                                g.first_touch_at = i
                            g.last_touch_at = i
                            g.filled_frac = max(g.filled_frac, min(1.0, (g.top - l[i]) / size))
                            if l[i] <= g.bottom:
                                g.filled_at = i
                            if c[i] < g.bottom:
                                g.invalidated_at = i
                        elif g.direction < 0 and h[i] > g.bottom:
                            if g.first_touch_at is None:
                                g.first_touch_at = i
                            g.last_touch_at = i
                            g.filled_frac = max(g.filled_frac, min(1.0, (h[i] - g.bottom) / size))
                            if h[i] >= g.top:
                                g.filled_at = i
                            if c[i] > g.top:
                                g.invalidated_at = i
                    if g.active:
                        keep.append(g)
                active_fvgs = keep

            # -- new FVGs (known at close of bar i)
            if i >= 2:
                if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= self.fvg_min_atr * atr[i]:
                    g = FairValueGap(1, top=float(l[i]), bottom=float(h[i - 2]), created_at=i)
                    fvgs.append(g)
                    active_fvgs.append(g)
                if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= self.fvg_min_atr * atr[i]:
                    g = FairValueGap(-1, top=float(l[i - 2]), bottom=float(h[i]), created_at=i)
                    fvgs.append(g)
                    active_fvgs.append(g)

            # -- structure breaks on close (swings known before this bar)
            if last_sh is not None and not sh_taken and c[i] > last_sh.price:
                kind = "CHOCH" if trend == -1 else "BOS"
                trend, sh_taken = 1, True
                events.append(StructureEvent(kind, 1, last_sh.price, i))
                ob = self._make_ob(1, last_sh.index, i, o, h, l, c, atr, fvgs, kind)
                if ob:
                    obs.append(ob)
                    active_obs.append(ob)
            if last_sl is not None and not sl_taken and c[i] < last_sl.price:
                kind = "CHOCH" if trend == 1 else "BOS"
                trend, sl_taken = -1, True
                events.append(StructureEvent(kind, -1, last_sl.price, i))
                ob = self._make_ob(-1, last_sl.index, i, o, h, l, c, atr, fvgs, kind)
                if ob:
                    obs.append(ob)
                    active_obs.append(ob)

            # -- swings confirmed at the close of bar i become usable from i+1
            for s in by_conf.get(i, []):
                if s.kind == "high":
                    last_sh, sh_taken = s, False
                else:
                    last_sl, sl_taken = s, False

        pools = self._liquidity_pools(swings, h, l, atr[-1] if n else 0.0)
        sweeps = detect_sweeps(df, swings, self.sweep_lookback) if n else []
        # sweeps of equal highs/lows (pools) on the recent bars
        for p in pools:
            if p.swept_at is not None and p.swept_at >= n - self.sweep_lookback:
                i = p.swept_at
                if p.side == "buy_side" and c[i] < p.price:
                    sweeps.append(PriceEvent("liquidity_sweep", "bear", p.price, i))
                elif p.side == "sell_side" and c[i] > p.price:
                    sweeps.append(PriceEvent("liquidity_sweep", "bull", p.price, i))
        sweeps.sort(key=lambda e: e.index)

        range_high = range_low = None
        if last_sh is not None and last_sl is not None:
            range_high = max(last_sh.price, float(h[last_sh.index:].max()))
            range_low = min(last_sl.price, float(l[last_sl.index:].min()))

        return SMCSnapshot(
            n=n, last_close=float(c[-1]) if n else float("nan"), atr=float(atr[-1]) if n else 0.0,
            trend=trend, events=events, order_blocks=obs, fvgs=fvgs, pools=pools, sweeps=sweeps,
            swings=swings, internal_swings=internal,
            range_high=range_high, range_low=range_low, lows=l, highs=h,
        )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _touch(ob: OrderBlock, i: int) -> None:
        # count a new touch only if the previous bar was not already in the zone
        if ob.last_touch_at is None or ob.last_touch_at < i - 1:
            ob.touches += 1
        if ob.first_touch_at is None:
            ob.first_touch_at = i
        ob.last_touch_at = i

    def _make_ob(self, direction, swing_index, i, o, h, l, c, atr, fvgs, cause) -> Optional[OrderBlock]:
        seg = slice(swing_index, i + 1)
        if direction > 0:
            k = swing_index + int(np.argmin(l[seg]))
            j = next((x for x in range(k, max(k - 6, -1), -1) if c[x] < o[x]), k)
            disp = (c[i] - l[k]) / atr[i]
            has_fvg = any(g.direction > 0 and j < g.created_at <= i for g in fvgs[-40:])
            return OrderBlock(1, top=float(h[j]), bottom=float(l[j]), candle_index=j, created_at=i,
                              displacement_atr=float(disp), has_fvg=has_fvg, cause=cause)
        k = swing_index + int(np.argmax(h[seg]))
        j = next((x for x in range(k, max(k - 6, -1), -1) if c[x] > o[x]), k)
        disp = (h[k] - c[i]) / atr[i]
        has_fvg = any(g.direction < 0 and j < g.created_at <= i for g in fvgs[-40:])
        return OrderBlock(-1, top=float(h[j]), bottom=float(l[j]), candle_index=j, created_at=i,
                          displacement_atr=float(disp), has_fvg=has_fvg, cause=cause)

    def _liquidity_pools(self, swings: List[Swing], h: np.ndarray, l: np.ndarray, atr_last: float) -> List[LiquidityPool]:
        tol = self.equal_tol * atr_last if atr_last > 0 else 0.0
        pools: List[LiquidityPool] = []
        for kind, side in (("high", "buy_side"), ("low", "sell_side")):
            pts = sorted([s for s in swings if s.kind == kind], key=lambda s: s.index)
            groups: List[List[Swing]] = []
            for s in pts:
                for g in groups:
                    if abs(np.mean([x.price for x in g]) - s.price) <= tol:
                        g.append(s)
                        break
                else:
                    groups.append([s])
            for g in groups:
                price = max(x.price for x in g) if kind == "high" else min(x.price for x in g)
                last_idx = max(x.index for x in g)
                after_from = last_idx + 1
                swept_at = None
                if kind == "high":
                    idx = np.flatnonzero(h[after_from:] > price)
                else:
                    idx = np.flatnonzero(l[after_from:] < price)
                if len(idx):
                    swept_at = int(after_from + idx[0])
                pools.append(LiquidityPool(side, float(price), len(g), last_idx, swept_at))
        return pools
