"""
Risk management: stop-loss / take-profit construction and portfolio-level limits.

Stop loss  : beyond the structural invalidation (sweep low / zone edge) plus an
             ATR buffer and the spread, clamped to [sl_min_atr, sl_max_atr] x ATR.
             A structural stop wider than sl_max_atr is REJECTED (not shrunk),
             because shrinking it would put the stop inside the structure.
Targets    : R multiples (default 1R / 2R / 3R). TP3 is extended to the next
             liquidity pool when one exists further out (capped at 5R).
Obstacle   : if un-swept opposing liquidity sits closer than MIN_RR, the trade
             has no room and is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from config import RiskConfig
from core.domain import Direction, RiskPlan, SymbolSpec


@dataclass
class PlanRequest:
    direction: Direction
    entry: float                # expected fill (ask for buys, bid for sells)
    atr: float                  # ATR of the entry timeframe
    invalidation: Optional[float]
    spread: float
    liquidity_targets: List[float] = field(default_factory=list)   # in trade direction, nearest first
    obstacles: List[float] = field(default_factory=list)           # opposing S/R levels


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ plan
    def build_plan(self, req: PlanRequest, spec: SymbolSpec) -> Tuple[Optional[RiskPlan], List[str]]:
        cfg = self.cfg
        rej: List[str] = []
        d = req.direction.sign
        if req.atr <= 0:
            return None, ["ATR unavailable"]

        min_dist = cfg.sl_min_atr * req.atr
        max_dist = cfg.sl_max_atr * req.atr
        buffer = cfg.sl_atr_buffer * req.atr + req.spread
        if req.invalidation is not None:
            structural_sl = req.invalidation - d * buffer
            dist = (req.entry - structural_sl) * d
        else:
            dist = min_dist
        if dist <= 0:
            return None, ["Invalidation level is on the wrong side of entry"]
        if dist > max_dist:
            return None, [f"Structural stop too wide ({dist / req.atr:.1f} ATR > {cfg.sl_max_atr} ATR)"]
        dist = max(dist, min_dist)
        sl = req.entry - d * dist

        tps = [req.entry + d * m * dist for m in cfg.tp_r_multiples]
        # extend last TP to a liquidity target further out (max 5R)
        far = [t for t in req.liquidity_targets if (t - req.entry) * d > (tps[-1] - req.entry) * d]
        if far:
            tgt = far[0] - d * 0.1 * req.atr      # just before the pool
            if (tgt - req.entry) * d <= 5 * dist and (tgt - req.entry) * d > (tps[-1] - req.entry) * d:
                tps[-1] = tgt

        # obstacle check: first opposing liquidity / S-R in the way
        blockers = [x for x in list(req.liquidity_targets) + list(req.obstacles) if (x - req.entry) * d > 0]
        room_r = None
        if blockers:
            nearest = min(blockers, key=lambda x: (x - req.entry) * d)
            room_r = (nearest - req.entry) * d / dist
            if room_r < cfg.min_rr:
                rej.append(f"Opposing liquidity/level only {room_r:.2f}R away (min {cfg.min_rr}R)")

        rr = [(t - req.entry) * d / dist for t in tps]
        primary_rr = next((r for r in rr if r >= cfg.min_rr - 1e-9), rr[-1])
        if primary_rr < cfg.min_rr - 1e-9:
            rej.append(f"Risk/reward {primary_rr:.2f} below minimum {cfg.min_rr}")

        warnings = []
        if req.spread > cfg.max_spread_atr_ratio * req.atr:
            rej.append(f"Spread is {req.spread / req.atr:.0%} of ATR (max {cfg.max_spread_atr_ratio:.0%})")
        if rej:
            return None, rej

        zone = 0.15 * req.atr
        entry_low, entry_high = (req.entry - zone, req.entry) if d > 0 else (req.entry, req.entry + zone)
        r = spec.round_price
        plan = RiskPlan(
            entry=r(req.entry), entry_low=r(entry_low), entry_high=r(entry_high),
            stop_loss=r(sl), take_profits=[r(t) for t in tps], risk=dist,
            rr_targets=[round(x, 2) for x in rr], primary_rr=round(primary_rr, 2),
            room_to_obstacle_r=None if room_r is None else round(room_r, 2), warnings=warnings,
        )
        return plan, []

    # ------------------------------------------------------------------ portfolio limits
    def check_limits(self, symbol: str, currencies: Iterable[str], now: datetime,
                     todays_signals: List[dict], active_signals: List[dict],
                     consecutive_losses: int, last_signal_time: Optional[datetime]) -> List[str]:
        """
        todays_signals / active_signals: dicts with at least 'symbol' and 'currencies'.
        Returns a list of rejection reasons (empty = allowed).
        """
        cfg = self.cfg
        out: List[str] = []
        if len(todays_signals) >= cfg.max_signals_per_day:
            out.append(f"Daily signal limit reached ({cfg.max_signals_per_day})")
        if sum(1 for s in todays_signals if s["symbol"] == symbol) >= cfg.max_signals_per_symbol_per_day:
            out.append(f"Daily limit for {symbol} reached")
        if consecutive_losses >= cfg.max_consecutive_losses:
            out.append(f"{consecutive_losses} consecutive losses - new signals paused until next trading day")
        if len(active_signals) >= cfg.max_active_signals:
            out.append(f"Maximum active signals reached ({cfg.max_active_signals})")
        if any(s["symbol"] == symbol for s in active_signals):
            out.append(f"An active signal already exists on {symbol}")
        exposure: Dict[str, int] = {}
        for s in active_signals:
            for c in s.get("currencies", []):
                exposure[c] = exposure.get(c, 0) + 1
        for c in currencies:
            if exposure.get(c, 0) >= cfg.max_active_per_currency:
                out.append(f"Currency exposure limit reached for {c}")
        if last_signal_time is not None and now - last_signal_time < timedelta(minutes=cfg.symbol_cooldown_minutes):
            out.append(f"{symbol} cooldown active ({cfg.symbol_cooldown_minutes} min)")
        return out

    def volatility_warnings(self, atr_percentile: Optional[float]) -> List[str]:
        if atr_percentile is None:
            return []
        if atr_percentile >= self.cfg.high_volatility_percentile:
            return ["High volatility (ATR in top decile) - consider reducing position size"]
        if atr_percentile <= self.cfg.low_volatility_percentile:
            return ["Very low volatility - targets may take longer to reach"]
        return []
