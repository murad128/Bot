"""
Signal engine: multi-timeframe confluence scoring.

Pipeline for one symbol at one point in time (all inputs are CLOSED bars):

  1. Hard gates      session, spread, data sufficiency, portfolio limits
  2. HTF bias        D1 / H4 / H1 trend score  -> direction (or "wait")
  3. Entry trigger   at least one concrete setup on M15 confirmed on M5:
                       OB/FVG retest, liquidity sweep reversal,
                       breakout-retest, trend pullback
                     no trigger -> no signal, regardless of score
  4. Fundamentals    news blackout gate + macro context (gold)
  5. Risk plan       structural SL, R-multiple TPs, obstacle check, min RR
  6. Score           weighted components -> 0..100, threshold MIN_SIGNAL_SCORE

The score is a confluence measure, NOT a probability of success. Whether a
given threshold has an edge must be established with the backtester.

The same `SignalEngine.evaluate` is used by the live bot and the backtester.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
from core.domain import (ComponentScore, Direction, EvaluationResult, SignalCandidate,
                         SymbolSpec, Timeframe)
from core.fundamental_analysis import FundamentalAnalyzer, FundamentalAssessment
from core.price_action import (candle_patterns, detect_breakout_retest, detect_false_breakouts,
                               list_swings, nearest_level, support_resistance)
from core.risk_management import PlanRequest, RiskManager
from core.sessions import (is_session_allowed, minutes_since_week_open, minutes_to_week_close,
                           primary_session, session_label)
from core.smc_analysis import SMCAnalyzer, SMCSnapshot
from core.technical_analysis import compute_indicators

log = logging.getLogger(__name__)

REQUIRED_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4)
MIN_BARS = 220


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Indicators + candlestick pattern columns (causal)."""
    out = compute_indicators(df)
    return out.join(candle_patterns(out, out["atr"]))


@dataclass
class MarketContext:
    symbol: str
    spec: SymbolSpec
    now: datetime
    frames: Dict[Timeframe, pd.DataFrame]       # closed bars only, prepared with prepare_frame
    bid: float
    ask: float
    portfolio_rejections: List[str] = field(default_factory=list)


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


class SignalEngine:
    def __init__(self, cfg: Config, fundamentals: Optional[FundamentalAnalyzer], risk: Optional[RiskManager] = None):
        self.cfg = cfg
        self.fundamentals = fundamentals
        self.risk = risk or RiskManager(cfg.risk)
        a = cfg.analysis
        self._analyzers = {
            tf: SMCAnalyzer(swing_strength=a.swing_strength.get(tf.value, 3),
                            internal_strength=a.internal_swing_strength, fvg_min_atr=a.fvg_min_atr,
                            ob_min_displacement_atr=a.ob_min_displacement_atr,
                            equal_tol_atr=a.equal_level_atr_tol, sweep_lookback=a.sweep_lookback_bars,
                            window=a.smc_window)
            for tf in Timeframe
        }
        self._smc_cache: Dict[Tuple[str, Timeframe], Tuple[pd.Timestamp, SMCSnapshot]] = {}
        w = cfg.scoring.weights
        total = sum(w.values())
        self.weights = {k: 100.0 * v / total for k, v in w.items()}

    # ------------------------------------------------------------------ utils
    def smc(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> SMCSnapshot:
        """SMC snapshot cached per closed bar (the snapshot only changes when a new bar closes)."""
        key = (symbol, tf)
        last = df.index[-1]
        hit = self._smc_cache.get(key)
        if hit and hit[0] == last:
            return hit[1]
        snap = self._analyzers[tf].analyze(df)
        self._smc_cache[key] = (last, snap)
        return snap

    @staticmethod
    def tf_trend(row: pd.Series, structure: int = 0) -> float:
        """Trend score in [-1, 1] from EMAs, slope, supertrend and structure."""
        close = row["close"]
        slow = row["ema200"] if not np.isnan(row["ema200"]) else row["ema100"]
        if np.isnan(slow) or np.isnan(row["ema50"]):
            return 0.0
        s = 0.35 * (1 if close > slow else -1)
        s += 0.25 * (1 if row["ema50"] > slow else -1)
        slope = row.get("ema50_slope", np.nan)
        if not np.isnan(slope) and abs(slope) > 0.05:
            s += 0.2 * _sign(slope)
        st = row.get("supertrend_dir", 0)
        if st and not np.isnan(st):
            s += 0.2 * st
        s += 0.3 * structure
        return float(max(-1.0, min(1.0, s / 1.3)))

    def invalidation_reason(self, symbol: str, direction: Direction, created_at: datetime,
                            m15: pd.DataFrame) -> Optional[str]:
        """Opposite M15 CHOCH confirmed after the signal -> setup invalidated (used before TP1 only)."""
        snap = self.smc(symbol, Timeframe.M15, m15)
        offset = len(m15) - snap.n
        for e in reversed(snap.events):
            if e.kind != "CHOCH" or e.direction != -direction.sign:
                continue
            bar_close = m15.index[offset + e.index] + pd.Timedelta(minutes=15)
            if bar_close > pd.Timestamp(created_at):
                return f"Opposite M15 CHOCH through {e.level:.5g}"
            break
        return None

    # ------------------------------------------------------------------ main
    def evaluate(self, ctx: MarketContext) -> EvaluationResult:
        res = EvaluationResult(symbol=ctx.symbol, time=ctx.now, candidate=None)
        cfg = self.cfg
        spec = ctx.spec

        # ---- 1. hard gates -------------------------------------------------
        for tf in REQUIRED_TFS:
            df = ctx.frames.get(tf)
            if df is None or len(df) < MIN_BARS:
                res.rejections.append(f"Insufficient {tf.value} history ({0 if df is None else len(df)} bars)")
        if res.rejections:
            return res
        allowed = cfg.analysis.gold_allowed_sessions if spec.is_gold else cfg.analysis.allowed_sessions
        if not is_session_allowed(ctx.now, allowed):
            res.rejections.append(f"Outside allowed sessions ({', '.join(allowed)})")
            return res
        spread = ctx.ask - ctx.bid
        spread_pips = spec.to_pips(spread)
        if spread_pips > spec.max_spread_pips:
            res.rejections.append(f"Spread {spread_pips:.1f} pips > max {spec.max_spread_pips}")
            return res
        a = cfg.analysis
        if minutes_since_week_open(ctx.now) < a.week_open_buffer_min:
            res.rejections.append(f"Weekly open buffer ({a.week_open_buffer_min} min): gaps and wide spreads")
            return res
        if minutes_to_week_close(ctx.now) < a.week_close_buffer_min:
            res.rejections.append(f"Too close to the weekly close ({a.week_close_buffer_min} min): weekend gap risk")
            return res
        m5_spreads = ctx.frames[Timeframe.M5]["spread_price"].tail(288) if "spread_price" in ctx.frames[Timeframe.M5] else None
        if m5_spreads is not None:
            typical = float(m5_spreads.median()) if m5_spreads.notna().sum() >= 50 else float("nan")
            if typical > 0 and spread > a.abnormal_spread_mult * typical:
                res.rejections.append(f"Abnormal spread {spread_pips:.1f} pips (> {a.abnormal_spread_mult:g}x typical "
                                      f"{spec.to_pips(typical):.1f})")
                return res
        if ctx.portfolio_rejections:
            res.rejections.extend(ctx.portfolio_rejections)
            return res

        f = ctx.frames
        m5, m15, h1, h4 = f[Timeframe.M5], f[Timeframe.M15], f[Timeframe.H1], f[Timeframe.H4]
        d1 = f.get(Timeframe.D1)
        r5, r15, rh1, rh4 = m5.iloc[-1], m15.iloc[-1], h1.iloc[-1], h4.iloc[-1]
        atr15 = float(r15["atr"])
        if np.isnan(atr15) or atr15 <= 0:
            res.rejections.append("M15 ATR unavailable")
            return res

        # ---- 2. HTF bias (cheap EMA part first, structure afterwards) --------
        pre_h4 = self.tf_trend(rh4)
        pre_h1 = self.tf_trend(rh1)
        if abs(pre_h4 + pre_h1) < 0.2:
            res.rejections.append("Higher-timeframe trend unclear")
            return res
        snap_h4 = self.smc(ctx.symbol, Timeframe.H4, h4)
        snap_h1 = self.smc(ctx.symbol, Timeframe.H1, h1)
        t_h4 = self.tf_trend(rh4, snap_h4.trend)
        t_h1 = self.tf_trend(rh1, snap_h1.trend)
        t_d1 = self.tf_trend(d1.iloc[-1]) if d1 is not None and len(d1) >= 60 else 0.0
        bias = 0.25 * t_d1 + 0.40 * t_h4 + 0.35 * t_h1
        res.score = None
        if abs(bias) < cfg.scoring.min_htf_bias:
            res.rejections.append(f"HTF bias too weak ({bias:+.2f})")
            return res
        if _sign(t_h4) * _sign(t_h1) < 0 and min(abs(t_h4), abs(t_h1)) > 0.3:
            res.rejections.append("H4 and H1 trends conflict")
            return res
        direction = Direction.from_sign(bias)
        d = direction.sign
        res.direction = direction
        name = "bullish" if d > 0 else "bearish"

        # ---- 3. entry trigger --------------------------------------------
        snap15 = self.smc(ctx.symbol, Timeframe.M15, m15)
        smc15 = snap15.assess(d, a.trigger_lookback_m15, a.ob_min_displacement_atr)

        bull = d > 0
        pat_cols = ("bull_engulf", "bull_pin") if bull else ("bear_engulf", "bear_pin")
        m5_tail, m15_tail = m5.tail(3), m15.tail(3)
        m5_pattern = next((c for c in pat_cols if m5_tail[c].any()), None)
        m15_pattern = next((c for c in pat_cols if m15_tail[c].any()), None)

        swings_h1 = list_swings(h1.tail(a.smc_window), a.swing_strength.get("H1", 3))
        swings_15 = snap15.swings
        tol = 0.25 * atr15
        levels = support_resistance(m15, swings_15 + [s for s in swings_h1], tol)
        bo = [e for e in detect_breakout_retest(m15.tail(30), levels, tol) if (e.direction == "bull") == bull]
        fb = [e for e in detect_false_breakouts(m15.tail(10), levels) if (e.direction == "bull") == bull]

        look = m15.tail(a.trigger_lookback_m15)
        if bull:
            pullback = bool((look["low"] <= look["ema20"]).any() and r15["close"] > r15["ema50"]
                            and 40 <= r15["rsi"] <= 65)
        else:
            pullback = bool((look["high"] >= look["ema20"]).any() and r15["close"] < r15["ema50"]
                            and 35 <= r15["rsi"] <= 60)
        pullback = pullback and m5_pattern is not None and snap15.trend == d
        if smc15.zone is None and smc15.sweep is None and not bo and not pullback:
            res.rejections.append(f"No {name} M15 setup (zone retest / sweep / breakout-retest / pullback)")
            return res

        # M5 confirmation (computed only when an M15 setup exists)
        snap5 = self.smc(ctx.symbol, Timeframe.M5, m5)
        smc5 = snap5.assess(d, a.trigger_lookback_m5, a.ob_min_displacement_atr)
        m5_struct = bool(snap5.recent_events(d, a.trigger_lookback_m5))
        m5_confirm = m5_struct or m5_pattern is not None

        setup = None
        if smc15.zone is not None and m5_confirm:
            setup = "OB_RETEST" if smc15.zone["type"] == "OB" else "FVG_RETEST"
        elif (smc15.sweep is not None or smc5.sweep is not None) and m5_struct:
            setup = "LIQUIDITY_SWEEP"
        elif bo and m5_confirm:
            setup = "BREAKOUT_RETEST"
        elif pullback:
            setup = "TREND_PULLBACK"
        if setup is None:
            res.rejections.append(f"No M5 confirmation for {name} M15 setup")
            return res

        # ---- 4. fundamentals ---------------------------------------------
        if self.fundamentals is not None:
            fa = self.fundamentals.assess(spec, direction, ctx.now)
        else:
            fa = FundamentalAssessment(False, None, 0.5, warnings=["Fundamental module disabled"])
        if fa.blocked:
            res.rejections.append(fa.block_reason or "News blackout")
            return res
        if spec.is_gold and fa.conflicts > a.gold_max_macro_conflicts:
            res.rejections.append(f"Gold macro contradicts the technical {direction.value} "
                                  f"({fa.conflicts} conflicting inputs)")
            return res

        # ---- 5. risk plan ------------------------------------------------
        entry = ctx.ask if bull else ctx.bid
        if setup == "TREND_PULLBACK":
            look = m15.tail(6)
            invalidation = float(look["low"].min()) if bull else float(look["high"].max())
        else:
            invalidation = smc15.invalidation_price
        targets = sorted(set(snap15.liquidity_targets(d, entry) + snap_h1.liquidity_targets(d, entry)),
                         key=lambda x: (x - entry) * d)
        h1_levels = support_resistance(h1, swings_h1, 0.25 * float(rh1["atr"]))
        obstacles = [lv.price for lv in h1_levels if lv.touches >= 2 and (lv.price - entry) * d > 0]
        plan, plan_rej = self.risk.build_plan(
            PlanRequest(direction, entry, atr15, invalidation, spread, targets, obstacles), spec)
        if plan is None:
            res.rejections.extend(plan_rej)
            return res

        # ---- 6. components -----------------------------------------------
        comps: List[ComponentScore] = []
        W = self.weights

        # HTF trend
        agree = all(_sign(x) == d for x in (t_d1, t_h4, t_h1) if x != 0)
        c = ComponentScore("htf_trend", min(1.0, abs(bias) / 0.8) * (1.0 if agree else 0.8), W["htf_trend"])
        lab = lambda x: "bullish" if x > 0.15 else "bearish" if x < -0.15 else "neutral"
        c.reasons.append(f"D1 {lab(t_d1)}, H4 {lab(t_h4)}, H1 {lab(t_h1)}")
        if bull and rh1["ema20"] > rh1["ema50"] > rh1["ema100"] > rh1["ema200"]:
            c.reasons.append("H1 EMA 20/50/100/200 bullish alignment")
            c.score = min(1.0, c.score + 0.1)
        elif not bull and rh1["ema20"] < rh1["ema50"] < rh1["ema100"] < rh1["ema200"]:
            c.reasons.append("H1 EMA 20/50/100/200 bearish alignment")
            c.score = min(1.0, c.score + 0.1)
        c.details = {"d1": round(t_d1, 2), "h4": round(t_h4, 2), "h1": round(t_h1, 2), "bias": round(bias, 2)}
        comps.append(c)

        # Market structure
        c = ComponentScore("market_structure", 0.0, W["market_structure"])
        if snap_h1.trend == d:
            c.score += 0.45
            c.reasons.append(f"H1 {name} market structure")
        elif snap_h1.trend == -d:
            c.warnings.append(f"H1 structure still {'bearish' if d > 0 else 'bullish'}")
        if snap_h4.trend == d:
            c.score += 0.25
            c.reasons.append(f"H4 {name} market structure")
        if snap15.trend == d or smc15.recent_choch:
            c.score += 0.30
            if smc15.recent_choch:
                c.reasons.append(f"M15 {name} CHOCH")
            else:
                c.reasons.append(f"M15 {name} structure")
        comps.append(c)

        # SMC
        c = ComponentScore("smc", 0.7 * smc15.score + 0.3 * smc5.score, W["smc"])
        c.reasons.extend(f"M15 {r}" for r in smc15.reasons if "market structure" not in r)
        if smc5.recent_choch:
            c.reasons.append(f"M5 {name} CHOCH confirmation")
        elif smc5.recent_bos:
            c.reasons.append(f"M5 {name} BOS confirmation")
        c.warnings.extend(smc15.warnings)
        c.details = {"zone": smc15.zone, "m15_zone": snap15.zone_label, "m15_score": round(smc15.score, 2),
                     "m5_score": round(smc5.score, 2)}
        comps.append(c)

        # Momentum
        c = ComponentScore("momentum", 0.0, W["momentum"])
        rsi15 = float(r15["rsi"])
        if (bull and 45 <= rsi15 <= 70) or (not bull and 30 <= rsi15 <= 55):
            c.score += 0.3
            c.reasons.append(f"M15 RSI {rsi15:.0f} supportive")
        elif (bull and rsi15 > 75) or (not bull and rsi15 < 25):
            c.warnings.append(f"M15 RSI {rsi15:.0f} {'overbought' if bull else 'oversold'}")
        hist = m15["macd_hist"].tail(3).to_numpy()
        if len(hist) == 3 and not np.isnan(hist).any() and (hist[-1] * d > 0 or (hist[-1] - hist[-2]) * d > 0):
            c.score += 0.25
            c.reasons.append("MACD momentum confirming")
        k, dd = m5["stochrsi_k"].tail(5).to_numpy(), m5["stochrsi_d"].tail(5).to_numpy()
        if len(k) == 5 and not (np.isnan(k).any() or np.isnan(dd).any()):
            cross = (bull and k[-1] > dd[-1] and k.min() < 30) or (not bull and k[-1] < dd[-1] and k.max() > 70)
            if cross:
                c.score += 0.2
                c.reasons.append(f"M5 Stoch RSI turning {'up from oversold' if bull else 'down from overbought'}")
        adx_h1 = float(rh1["adx"]) if not np.isnan(rh1["adx"]) else 0.0
        if adx_h1 >= 20 and ((rh1["plus_di"] > rh1["minus_di"]) == bull):
            c.score += 0.25
            c.reasons.append(f"H1 ADX {adx_h1:.0f} (trending)")
        elif adx_h1 < 15:
            c.warnings.append(f"H1 ADX {adx_h1:.0f} - weak trend")
        cci15 = float(r15["cci"]) if not np.isnan(r15["cci"]) else 0.0
        c.details = {"rsi_m15": round(rsi15, 1), "adx_h1": round(adx_h1, 1), "cci_m15": round(cci15, 1)}
        comps.append(c)

        # Volatility
        c = ComponentScore("volatility", 0.0, W["volatility"])
        pct = float(r15["atr_pct"]) if not np.isnan(r15["atr_pct"]) else None
        if pct is not None and 0.2 <= pct <= 0.9:
            c.score += 0.6
            c.reasons.append("Healthy volatility")
        elif pct is not None:
            c.score += 0.2
        else:
            c.score += 0.3
        sq = m15["bb_squeeze"].tail(6).to_numpy()
        if sq[:-1].any() and not sq[-1]:
            c.score += 0.2
            c.reasons.append("Bollinger/Keltner squeeze release")
        over = (bull and r15["close"] > r15["kc_upper"]) or (not bull and r15["close"] < r15["kc_lower"])
        if over:
            c.warnings.append("Price stretched beyond Keltner channel")
        else:
            c.score += 0.2
        c.details = {"atr_m15": atr15, "atr_percentile": pct}
        comps.append(c)

        # Price action / S-R / volume
        c = ComponentScore("price_action", 0.0, W["price_action"])
        pretty = {"bull_engulf": "Bullish engulfing", "bear_engulf": "Bearish engulfing",
                  "bull_pin": "Bullish pin bar", "bear_pin": "Bearish pin bar"}
        if m15_pattern or m5_pattern:
            c.score += 0.35
            c.reasons.append(f"{pretty[m15_pattern or m5_pattern]} on {'M15' if m15_pattern else 'M5'}")
        if bo:
            c.score += 0.2
            c.reasons.append(f"Breakout and retest of {bo[-1].level:.{spec.digits}f}")
        elif smc15.sweep is not None:
            c.score += 0.2
        if fb:
            c.score += 0.1
            c.reasons.append("False breakout against the opposite side")
        lv = nearest_level(levels, entry, "support" if bull else "resistance")
        if lv is not None and abs(lv.price - entry) <= 0.6 * atr15:
            c.score += 0.2
            c.reasons.append(f"Reacting from {'support' if bull else 'resistance'} {lv.price:.{spec.digits}f}")
        if m15["vol_spike"].tail(3).any() or m5["vol_spike"].tail(3).any():
            c.score += 0.15
            c.reasons.append("Tick-volume spike on trigger")
        div_col = "bear_vol_div" if bull else "bull_vol_div"
        if m15[div_col].tail(5).any():
            c.score -= 0.2
            c.warnings.append("Volume divergence against the trade")
        comps.append(c)

        # Fundamentals
        c = ComponentScore("fundamentals", fa.score, W["fundamentals"], list(fa.reasons), list(fa.warnings))
        comps.append(c)

        # Risk/reward
        room = plan.room_to_obstacle_r
        c = ComponentScore("risk_reward", 0.7 if room is None else min(1.0, room / 4.0), W["risk_reward"])
        c.reasons.append(f"RR 1:{plan.primary_rr:.1f}" + ("" if room is None else f", {room:.1f}R to next opposing level"))
        comps.append(c)

        # Execution
        ratio = spread / atr15
        c = ComponentScore("execution", max(0.0, 1.0 - ratio / cfg.risk.max_spread_atr_ratio), W["execution"])
        c.details = {"spread_pips": round(spread_pips, 2), "spread_atr": round(ratio, 3)}
        if ratio > 0.5 * cfg.risk.max_spread_atr_ratio:
            c.warnings.append(f"Spread {spread_pips:.1f} pips is elevated")
        comps.append(c)

        # optional M1 timing
        warnings: List[str] = []
        m1 = f.get(Timeframe.M1)
        if a.use_m1 and m1 is not None and len(m1) > 50:
            r1 = m1.iloc[-1]
            if (r1["close"] - r1["ema20"]) * d > 0:
                comps[2].reasons.append("M1 timing aligned")
            else:
                warnings.append("M1 not yet aligned - consider waiting for M1 confirmation")

        score = round(sum(x.points for x in comps), 1)
        res.score = score
        res.components = comps
        sc = cfg.scoring
        if score < sc.min_score:
            res.rejections.append(f"Score {score:.0f} below threshold {sc.min_score:.0f}")
            return res

        grade = ("Very strong" if score >= sc.very_strong_score else "Strong" if score >= sc.strong_score
                 else "Moderate" if score >= 70 else "Below standard")
        reasons: List[str] = []
        for x in comps:
            for r in x.reasons:
                if r not in reasons:
                    reasons.append(r)
        for x in comps:
            for w in x.warnings:
                if w not in warnings:
                    warnings.append(w)
        warnings.extend(w for w in self.risk.volatility_warnings(pct) if w not in warnings)

        strength = "Strong " if abs(bias) >= 0.7 else ""
        cand = SignalCandidate(
            symbol=ctx.symbol, direction=direction, setup_type=setup, created_at=ctx.now,
            entry_timeframe="M15", confirm_timeframe="H1",
            trend_label=f"{strength}{'Bullish' if bull else 'Bearish'}",
            session=session_label(ctx.now), score=score, grade=grade, plan=plan, components=comps,
            reasons=reasons, warnings=warnings, atr=atr15, spread=spread,
            context={
                "bias": round(bias, 3), "t_d1": round(t_d1, 3), "t_h4": round(t_h4, 3), "t_h1": round(t_h1, 3),
                "session_bucket": primary_session(ctx.now), "rsi_m15": round(rsi15, 1), "adx_h1": round(adx_h1, 1),
                "atr_pct": pct, "spread_pips": round(spread_pips, 2), "zone": smc15.zone,
                "fund_conflict": fa.conflict, "fund_data": fa.data_available,
                "components": {x.name: round(x.score, 3) for x in comps},
                "tags": sorted(set(_tags(setup, smc15, smc5, m5_pattern, m15_pattern, bo, fa))),
            },
        )
        res.candidate = cand
        return res


def _tags(setup, smc15, smc5, m5_pattern, m15_pattern, bo, fa) -> List[str]:
    """Discrete features stored with every signal for the trade-review module."""
    t = [f"setup:{setup}"]
    if smc15.zone:
        t.append(f"zone:{smc15.zone['type']}")
        if smc15.zone.get("has_fvg"):
            t.append("ob_with_fvg")
    if smc15.sweep is not None:
        t.append("m15_sweep")
    if smc15.recent_choch:
        t.append("m15_choch")
    if smc5.recent_choch:
        t.append("m5_choch")
    if smc15.inducement:
        t.append("inducement")
    if smc15.in_correct_zone:
        t.append("premium_discount_ok")
    if m5_pattern:
        t.append(f"m5:{m5_pattern}")
    if m15_pattern:
        t.append(f"m15:{m15_pattern}")
    if bo:
        t.append("breakout_retest")
    if fa.conflict:
        t.append("fundamental_conflict")
    return t
