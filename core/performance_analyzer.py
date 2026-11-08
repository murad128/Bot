"""
Performance statistics computed from closed signals.

Definitions (also printed in reports):
  Resolved trades : WIN + LOSS + BREAKEVEN (cancelled signals are excluded)
  Win rate        : WIN / resolved trades
  Realised R      : per-signal result including partial exits (see trade_state.py)
  Profit factor   : sum of positive R / |sum of negative R|   (all closed signals incl. cancelled)
  Expectancy      : mean realised R per closed signal (resolved + cancelled)
  Max drawdown    : largest peak-to-trough fall of the cumulative R curve (closing order)
  TP1/TP2/TP3 hit : milestones reached by signals; NOT counted as separate wins
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class Stats:
    total: int = 0
    open: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    cancelled: int = 0
    tp1: int = 0
    tp2: int = 0
    tp3: int = 0
    sl: int = 0
    expired: int = 0
    net_r: float = 0.0
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0
    max_drawdown_r: float = 0.0
    avg_planned_rr: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    closed: int = 0

    @property
    def resolved(self) -> int:
        return self.wins + self.losses + self.breakeven

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.resolved if self.resolved else None

    @property
    def profit_factor(self) -> Optional[float]:
        if self.gross_loss_r == 0:
            return None if self.gross_win_r == 0 else float("inf")
        return self.gross_win_r / abs(self.gross_loss_r)

    @property
    def expectancy_r(self) -> Optional[float]:
        return self.net_r / self.closed if self.closed else None

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d.update(resolved=self.resolved, win_rate=self.win_rate, profit_factor=self.profit_factor,
                 expectancy_r=self.expectancy_r)
        return d


def _get(rec: Any, key: str, default=None):
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


def compute_stats(records: Iterable[Any]) -> Stats:
    s = Stats()
    rr_sum = 0.0
    curve = peak = 0.0
    wins_r: List[float] = []
    losses_r: List[float] = []
    closed = []
    for r in records:
        s.total += 1
        rr_sum += float(_get(r, "rr", 0.0) or 0.0)
        outcome = _get(r, "outcome")
        hits = int(_get(r, "tp_hits", 0) or 0)
        s.tp1 += hits >= 1
        s.tp2 += hits >= 2
        s.tp3 += hits >= 3
        status = _get(r, "status")
        if outcome == "OPEN":
            s.open += 1
            continue
        closed.append(r)
        if status == "STOPPED" and hits == 0:
            s.sl += 1
        if status == "EXPIRED":
            s.expired += 1
        if outcome == "WIN":
            s.wins += 1
        elif outcome == "LOSS":
            s.losses += 1
        elif outcome == "BREAKEVEN":
            s.breakeven += 1
        elif outcome == "CANCELLED":
            s.cancelled += 1
    closed.sort(key=lambda r: _get(r, "closed_at") or _get(r, "created_at"))
    for r in closed:
        x = float(_get(r, "realized_r", 0.0) or 0.0)
        s.net_r += x
        if x > 0:
            s.gross_win_r += x
            if _get(r, "outcome") == "WIN":
                wins_r.append(x)
        elif x < 0:
            s.gross_loss_r += x
            if _get(r, "outcome") == "LOSS":
                losses_r.append(x)
        curve += x
        peak = max(peak, curve)
        s.max_drawdown_r = max(s.max_drawdown_r, peak - curve)
    s.closed = len(closed)
    s.avg_planned_rr = rr_sum / s.total if s.total else 0.0
    s.avg_win_r = sum(wins_r) / len(wins_r) if wins_r else 0.0
    s.avg_loss_r = sum(losses_r) / len(losses_r) if losses_r else 0.0
    s.net_r = round(s.net_r, 3)
    s.max_drawdown_r = round(s.max_drawdown_r, 3)
    return s


def breakdown(records: Iterable[Any], key: str) -> Dict[str, Stats]:
    groups: Dict[str, list] = defaultdict(list)
    for r in records:
        if key == "timeframe":
            k = f"{_get(r, 'entry_timeframe')}/{_get(r, 'confirm_timeframe')}"
        else:
            k = _get(r, key) or "unknown"
        groups[str(k)].append(r)
    return {k: compute_stats(v) for k, v in sorted(groups.items())}


def best_of(groups: Dict[str, Stats], min_resolved: int = 1) -> Optional[str]:
    cands = [(k, v) for k, v in groups.items() if v.resolved >= min_resolved]
    if not cands:
        return None
    return max(cands, key=lambda kv: (kv[1].net_r, kv[1].win_rate or 0))[0]


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def fmt_pf(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "∞ (no losses)"
    return f"{x:.2f}"


def text_summary(s: Stats, title: str = "Performance") -> str:
    lines = [
        f"{title}",
        f"  Signals: {s.total} (open {s.open}, cancelled {s.cancelled})",
        f"  Resolved: {s.resolved}  W/L/BE: {s.wins}/{s.losses}/{s.breakeven}",
        f"  Win rate: {fmt_pct(s.win_rate)}",
        f"  TP1/TP2/TP3 reached: {s.tp1}/{s.tp2}/{s.tp3}  SL before TP1: {s.sl}  Expired: {s.expired}",
        f"  Net R: {s.net_r:+.2f}  Expectancy: {'n/a' if s.expectancy_r is None else f'{s.expectancy_r:+.3f}R'}",
        f"  Profit factor: {fmt_pf(s.profit_factor)}  Max DD: {s.max_drawdown_r:.2f}R",
        f"  Avg planned RR: 1:{s.avg_planned_rr:.2f}  Avg win: {s.avg_win_r:+.2f}R  Avg loss: {s.avg_loss_r:+.2f}R",
    ]
    return "\n".join(lines)
