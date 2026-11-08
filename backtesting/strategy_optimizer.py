"""
Walk-forward validation / parameter selection.

Procedure
1. Collect every candidate once over the whole period with the LOWEST
   thresholds in the grid (outcomes are simulated independently).
2. Roll windows: optimise on `train_days`, then evaluate the chosen
   parameters on the following `test_days` (never seen during selection).
3. Report the concatenated out-of-sample (OOS) trades - this is the only
   number that estimates live behaviour. In-sample results are shown for
   comparison: a large IS/OOS gap is a sign of overfitting.

The grid is deliberately small (thresholds only, no indicator periods) to
limit the degrees of freedom. The result is a RECOMMENDATION; nothing is
written to the live configuration.
"""
from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from core.performance_analyzer import Stats, compute_stats, text_summary
from backtesting.backtest_engine import Backtester, Candidate, ReplayParams

log = logging.getLogger(__name__)

DEFAULT_GRID = {"min_score": [65.0, 70.0, 75.0, 80.0], "min_rr": [2.0, 2.5]}


@dataclass
class WindowResult:
    train_start: datetime
    train_end: datetime
    test_end: datetime
    params: Dict[str, float]
    train_stats: Stats
    test_stats: Stats


@dataclass
class WalkForwardResult:
    symbol: str
    windows: List[WindowResult]
    oos_trades: List[Candidate]
    oos_stats: Stats
    is_stats: Stats
    param_counts: Dict[str, int]
    recommendation: str
    n_candidates: int = 0
    notes: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [f"WALK-FORWARD {self.symbol}: {len(self.windows)} windows, {self.n_candidates} raw candidates"]
        for w in self.windows:
            lines.append(f"  train {w.train_start:%Y-%m-%d}->{w.train_end:%Y-%m-%d} test ->{w.test_end:%Y-%m-%d} "
                         f"params={w.params} | IS n={w.train_stats.resolved} netR={w.train_stats.net_r:+.2f} | "
                         f"OOS n={w.test_stats.resolved} netR={w.test_stats.net_r:+.2f}")
        lines.append(text_summary(self.is_stats, "In-sample (selected params, optimistic)"))
        lines.append(text_summary(self.oos_stats, "OUT-OF-SAMPLE (concatenated test windows)"))
        lines.append(f"Parameter stability: {self.param_counts}")
        lines.extend(self.notes)
        lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


def _objective(s: Stats, min_trades: int) -> float:
    if s.resolved < min_trades:
        return float("-inf")
    pf = s.profit_factor if s.profit_factor not in (None, float("inf")) else 3.0
    # expectancy first, mild preference for smoother curves
    return (s.expectancy_r or 0.0) - 0.02 * s.max_drawdown_r + 0.01 * min(pf, 3.0)


def walk_forward(bt: Backtester, data, base_params: ReplayParams, grid: Optional[Dict[str, list]] = None,
                 train_days: int = 90, test_days: int = 30, min_trades: int = 15) -> WalkForwardResult:
    grid = grid or DEFAULT_GRID
    keys = sorted(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    floor_score = min(grid.get("min_score", [base_params.min_score]))
    floor_rr = min(grid.get("min_rr", [base_params.min_rr]))
    cands = bt.collect(data, score_floor=floor_score, rr_floor=floor_rr, rejections=Counter())
    idx = data.base.index
    start = idx[0].to_pydatetime() + timedelta(days=45)   # warm-up for H4/D1 indicators
    end = idx[-1].to_pydatetime()
    windows: List[WindowResult] = []
    oos: List[Candidate] = []
    is_trades: List[Candidate] = []
    counts: Counter = Counter()
    t = start
    while t + timedelta(days=train_days + test_days) <= end + timedelta(days=1):
        tr_end = t + timedelta(days=train_days)
        te_end = min(tr_end + timedelta(days=test_days), end)
        train = [c for c in cands if t <= c.time < tr_end]
        test = [c for c in cands if tr_end <= c.time < te_end]
        best, best_val, best_trades = None, float("-inf"), []
        for combo in combos:
            p = replace(base_params, **combo)
            tr = Backtester.replay(train, p)
            val = _objective(compute_stats(tr), min_trades)
            if val > best_val:
                best, best_val, best_trades = combo, val, tr
        if best is None:
            best = {k: getattr(base_params, k) for k in keys}   # not enough trades -> keep defaults
            best_trades = Backtester.replay(train, base_params)
        p = replace(base_params, **best)
        te = Backtester.replay(test, p)
        windows.append(WindowResult(t, tr_end, te_end, best, compute_stats(best_trades), compute_stats(te)))
        oos.extend(te)
        is_trades.extend(best_trades)
        counts[str(best)] += 1
        t = t + timedelta(days=test_days)

    oos_stats = compute_stats(oos)
    is_stats = compute_stats(is_trades)
    notes = []
    if not windows:
        rec = (f"Not enough data for walk-forward (need > {train_days + test_days + 45} days). "
               "Download more history.")
    elif oos_stats.resolved < 30:
        rec = (f"Only {oos_stats.resolved} out-of-sample trades - too few to draw conclusions. "
               "Keep the default parameters and gather more data.")
    elif (oos_stats.expectancy_r or 0) <= 0:
        rec = ("Out-of-sample expectancy is not positive. Do NOT rely on these signals for this instrument "
               "until the strategy is revised.")
    else:
        top = counts.most_common(1)[0]
        rec = (f"Most frequently selected parameters: {top[0]} ({top[1]}/{len(windows)} windows). "
               f"OOS expectancy {oos_stats.expectancy_r:+.3f}R over {oos_stats.resolved} trades. "
               "Review before changing live settings.")
    if windows and is_stats.expectancy_r is not None and oos_stats.expectancy_r is not None:
        gap = is_stats.expectancy_r - oos_stats.expectancy_r
        if gap > 0.15:
            notes.append(f"Warning: in-sample expectancy exceeds OOS by {gap:.2f}R - likely overfitting.")
    return WalkForwardResult(data.symbol, windows, oos, oos_stats, is_stats, dict(counts), rec, len(cands), notes)


def simple_holdout(bt: Backtester, data, params: ReplayParams, split: float = 0.7):
    """Single in-sample / out-of-sample split with FIXED parameters (no optimisation)."""
    cands = bt.collect(data, rejections=Counter())
    idx = data.base.index
    cut = idx[0] + (idx[-1] - idx[0]) * split
    ins = Backtester.replay([c for c in cands if pd.Timestamp(c.time) < cut], params)
    out = Backtester.replay([c for c in cands if pd.Timestamp(c.time) >= cut], params)
    return compute_stats(ins), compute_stats(out), cut
