"""
Event-driven backtester that reuses the live SignalEngine and TradeState.

No look-ahead
* Higher timeframes are resampled from the base (M5) data. At base bar i
  (decision time = close of bar i) only HTF bars whose CLOSE time <= that
  moment are passed to the engine (np.searchsorted on close times).
* Indicators are causal (verified by tests/test_no_lookahead.py), so they are
  computed once on the full history.
* Trades are entered at the decision-time price and evaluated from the NEXT
  bar onward. Intrabar SL/TP ambiguity is resolved pessimistically.

Costs
* Spread: per-bar `spread_price` if present in the data, else a configurable
  default. Buys enter at ask and exit at bid; sells the opposite.
* Slippage: added against us on entry and stop exits.
* Commission: round-turn, in pips (InstrumentConfig.commission_pips).

Two-stage design (enables fast walk-forward optimisation)
1. collect(): run the engine with LOW gating floors and record every
   candidate with its independently simulated outcome.
2. replay(): apply the real thresholds, one-trade-per-symbol, daily limits,
   cooldown and consecutive-loss pause chronologically.
`run()` = collect + replay with the configured parameters.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time as _time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config
from core.domain import Direction, SymbolSpec, Timeframe
from core.fundamental_analysis import FundamentalAnalyzer
from core.performance_analyzer import Stats, breakdown, compute_stats, text_summary
from core.signal_engine import MarketContext, SignalEngine, prepare_frame
from core.smc_analysis import SMCAnalyzer
from core.trade_state import TradeState
from backtesting.historical_data import resample

log = logging.getLogger(__name__)

HTFS = (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)


@dataclass
class Candidate:
    symbol: str
    time: datetime
    bar_index: int
    direction: str
    setup_type: str
    score: float
    component_scores: Dict[str, float]
    entry: float
    stop_loss: float
    take_profits: List[float]
    rr: float
    room_r: Optional[float]
    session_bucket: str
    context: dict
    warnings: List[str]
    entry_timeframe: str = "M15"
    confirm_timeframe: str = "H1"
    # outcome (filled by simulate)
    status: str = "ACTIVE"
    outcome: str = "OPEN"
    tp_hits: int = 0
    realized_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    events: List[dict] = field(default_factory=list)

    def rescore(self, weights: Dict[str, float]) -> float:
        total = sum(weights.values())
        return sum(100.0 * weights[k] / total * max(0.0, min(1.0, v))
                   for k, v in self.component_scores.items() if k in weights)


@dataclass
class ReplayParams:
    min_score: float
    min_rr: float
    max_per_day: int
    max_per_symbol_day: int
    cooldown_min: int
    max_consecutive_losses: int
    weights: Optional[Dict[str, float]] = None

    @staticmethod
    def from_config(cfg: Config) -> "ReplayParams":
        r = cfg.risk
        return ReplayParams(cfg.scoring.min_score, r.min_rr, r.max_signals_per_day,
                            r.max_signals_per_symbol_per_day, r.symbol_cooldown_minutes, r.max_consecutive_losses)


@dataclass
class PreparedData:
    symbol: str
    spec: SymbolSpec
    base: pd.DataFrame
    frames: Dict[Timeframe, pd.DataFrame]
    close_ns: Dict[Timeframe, np.ndarray]
    choch_bull_ns: np.ndarray
    choch_bear_ns: np.ndarray
    default_spread: float


@dataclass
class BacktestResult:
    symbol: str
    start: Optional[datetime]
    end: Optional[datetime]
    trades: List[Candidate]
    n_candidates: int
    stats: Stats
    by_setup: Dict[str, Stats]
    by_session: Dict[str, Stats]
    by_direction: Dict[str, Stats]
    rejections: List[tuple]
    assumptions: Dict[str, object]
    runtime_s: float

    def to_text(self) -> str:
        lines = [f"BACKTEST {self.symbol}  {self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d}" if self.start else
                 f"BACKTEST {self.symbol}",
                 f"Candidates evaluated (pre-filter): {self.n_candidates}   runtime {self.runtime_s:.1f}s",
                 text_summary(self.stats, "Overall")]
        for title, grp in (("By setup", self.by_setup), ("By session", self.by_session),
                           ("By direction", self.by_direction)):
            lines.append(title)
            for k, s in grp.items():
                lines.append(f"  {k:<18} n={s.total:<4} WR={'n/a' if s.win_rate is None else f'{s.win_rate*100:.1f}%':<7} "
                             f"netR={s.net_r:+.2f}  PF={'n/a' if s.profit_factor is None else f'{s.profit_factor:.2f}'}")
        lines.append("Most common rejection reasons:")
        for r, n in self.rejections[:10]:
            lines.append(f"  {n:>7}  {r}")
        lines.append("Assumptions: " + json.dumps(self.assumptions, default=str))
        return "\n".join(lines)

    def trades_frame(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            d = asdict(t)
            d.pop("context", None)
            d["take_profits"] = ",".join(f"{x}" for x in t.take_profits)
            d["tags"] = ";".join(t.context.get("tags", []))
            rows.append(d)
        return pd.DataFrame(rows)


def _norm_reason(r: str) -> str:
    return re.sub(r"[-+]?\d+(\.\d+)?", "#", r)


class Backtester:
    def __init__(self, cfg: Config, calendar=None, default_spread_pips: Optional[float] = None):
        self.cfg = cfg
        self.calendar = calendar
        self.default_spread_pips = default_spread_pips

    # ------------------------------------------------------------------ preparation
    def prepare(self, symbol: str, base: pd.DataFrame, spec: SymbolSpec) -> PreparedData:
        base = base.sort_index()
        base.index = base.index.as_unit("ns")      # pandas 3 may default to us resolution
        frames = {Timeframe.M5: prepare_frame(base)}
        for tf in HTFS:
            frames[tf] = prepare_frame(resample(base, tf))
        close_ns = {tf: (df.index + pd.Timedelta(minutes=tf.minutes)).as_unit("ns").asi8
                    for tf, df in frames.items()}
        # full-history M15 structure for the cancellation rule (opposite CHOCH)
        a = self.cfg.analysis
        full = SMCAnalyzer(swing_strength=a.swing_strength.get("M15", 3), window=10 ** 9).analyze(frames[Timeframe.M15])
        m15_close = close_ns[Timeframe.M15]
        bull = np.array(sorted(m15_close[e.index] for e in full.events if e.kind == "CHOCH" and e.direction > 0), dtype=np.int64)
        bear = np.array(sorted(m15_close[e.index] for e in full.events if e.kind == "CHOCH" and e.direction < 0), dtype=np.int64)
        dsp = self.default_spread_pips if self.default_spread_pips is not None else 0.4 * spec.max_spread_pips
        return PreparedData(symbol, spec, base, frames, close_ns, bull, bear, dsp * spec.pip_size)

    def _frames_at(self, data: PreparedData, now_ns: int) -> Dict[Timeframe, pd.DataFrame]:
        out = {}
        bars = self.cfg.analysis.bars
        for tf, df in data.frames.items():
            k = int(np.searchsorted(data.close_ns[tf], now_ns, side="right"))
            n = bars.get(tf.value, 600)
            out[tf] = df.iloc[max(0, k - n):k]
        return out

    # ------------------------------------------------------------------ stage 1
    def collect(self, data: PreparedData, start: Optional[datetime] = None, end: Optional[datetime] = None,
                score_floor: Optional[float] = None, rr_floor: Optional[float] = None,
                rejections: Optional[Counter] = None) -> List[Candidate]:
        cfg = copy.deepcopy(self.cfg)
        if score_floor is not None:
            cfg.scoring.min_score = score_floor
        if rr_floor is not None:
            cfg.risk.min_rr = rr_floor
        fund = FundamentalAnalyzer(cfg.news, self.calendar, macro=None, news_provider=None,
                                   block_if_unavailable=False)
        engine = SignalEngine(cfg, fund)
        base = data.frames[Timeframe.M5]
        base_close_ns = data.close_ns[Timeframe.M5]
        spreads = base["spread_price"].to_numpy() if "spread_price" in base else np.full(len(base), np.nan)
        closes = base["close"].to_numpy()
        lo = 0 if start is None else int(base.index.searchsorted(pd.Timestamp(start)))
        hi = len(base) if end is None else int(base.index.searchsorted(pd.Timestamp(end)))
        cands: List[Candidate] = []
        last_key: Dict[str, int] = {}
        rej = rejections if rejections is not None else Counter()
        t0 = _time.monotonic()
        step = max(1, (hi - lo) // 10)
        for i in range(lo, hi):
            now_ns = int(base_close_ns[i])
            now = pd.Timestamp(now_ns, unit="ns", tz="UTC").to_pydatetime()
            frames = self._frames_at(data, now_ns)
            sp = spreads[i] if not np.isnan(spreads[i]) else data.default_spread
            bid = float(closes[i])
            ctx = MarketContext(data.symbol, data.spec, now, frames, bid, bid + sp)
            res = engine.evaluate(ctx)
            if res.candidate is None:
                if res.rejections:
                    rej[_norm_reason(res.rejections[0])] += 1
            else:
                c = res.candidate
                key = f"{c.direction.value}:{c.setup_type}"
                if i - last_key.get(key, -10 ** 9) <= 3:      # same setup on consecutive bars
                    continue
                last_key[key] = i
                p = c.plan
                cand = Candidate(
                    symbol=data.symbol, time=now, bar_index=i, direction=c.direction.value, setup_type=c.setup_type,
                    score=c.score, component_scores={x.name: x.score for x in c.components},
                    entry=p.entry, stop_loss=p.stop_loss, take_profits=list(p.take_profits), rr=p.primary_rr,
                    room_r=p.room_to_obstacle_r, session_bucket=c.context.get("session_bucket", ""),
                    context=c.context, warnings=list(c.warnings))
                self.simulate(cand, data)
                cands.append(cand)
            if (i - lo) % step == 0 and i > lo:
                log.info("%s backtest %d%% (%d candidates, %.0fs)", data.symbol, 100 * (i - lo) // (hi - lo),
                         len(cands), _time.monotonic() - t0)
        return cands

    def simulate(self, cand: Candidate, data: PreparedData) -> None:
        spec, r = data.spec, self.cfg.risk
        d = 1 if cand.direction == "BUY" else -1
        slip = spec.slippage_pips * spec.pip_size
        st = TradeState(
            direction=Direction(cand.direction), entry=cand.entry + d * slip, stop_loss=cand.stop_loss,
            take_profits=list(cand.take_profits), close_fractions=list(r.tp_close_fractions),
            opened_at=cand.time, max_hold=timedelta(hours=r.max_holding_hours),
            move_sl_to_be=r.move_sl_to_be_after_tp1, exit_slippage=slip,
            commission=spec.commission_pips * spec.pip_size)
        opp = data.choch_bear_ns if d > 0 else data.choch_bull_ns
        k = int(np.searchsorted(opp, pd.Timestamp(cand.time).as_unit("ns").value, side="right"))
        cancel_ns = int(opp[k]) if k < len(opp) else None
        base = data.base
        hs, ls, cs = base["high"].to_numpy(), base["low"].to_numpy(), base["close"].to_numpy()
        sps = base["spread_price"].to_numpy() if "spread_price" in base else np.full(len(base), np.nan)
        close_ns = data.close_ns[Timeframe.M5]
        for j in range(cand.bar_index + 1, len(base)):
            sp = sps[j] if not np.isnan(sps[j]) else data.default_spread
            t = pd.Timestamp(int(close_ns[j]), unit="ns", tz="UTC").to_pydatetime()
            st.update_bar(float(hs[j]), float(ls[j]), float(cs[j]), t, sp)
            if not st.is_open:
                break
            if cancel_ns is not None and close_ns[j] >= cancel_ns and st.tp_hits == 0:
                px = float(cs[j]) if d > 0 else float(cs[j]) + sp
                st.cancel(px, t)
                cand.context = dict(cand.context, cancel_reason="Opposite M15 CHOCH")
                break
        cand.status = st.status.value
        cand.outcome = st.outcome.value
        cand.tp_hits = st.tp_hits
        cand.realized_r = round(st.realized_r, 4)
        cand.mfe_r, cand.mae_r = round(st.mfe_r, 3), round(st.mae_r, 3)
        cand.closed_at = st.closed_at
        cand.exit_price = st.exit_price
        cand.events = [{"kind": e.kind, "time": e.time.isoformat(), "price": e.price, "r": e.r} for e in st.events]

    # ------------------------------------------------------------------ stage 2
    @staticmethod
    def replay(cands: List[Candidate], p: ReplayParams) -> List[Candidate]:
        accepted: List[Candidate] = []
        open_until: Dict[str, datetime] = {}
        last_time: Dict[str, datetime] = {}
        for c in sorted(cands, key=lambda x: x.time):
            score = c.rescore(p.weights) if p.weights else c.score
            if score < p.min_score:
                continue
            if c.rr < p.min_rr - 1e-9 or (c.room_r is not None and c.room_r < p.min_rr):
                continue
            if c.outcome == "OPEN":            # not resolvable within the data
                continue
            if c.symbol in open_until and c.time < open_until[c.symbol]:
                continue
            if c.symbol in last_time and c.time - last_time[c.symbol] < timedelta(minutes=p.cooldown_min):
                continue
            day = c.time.date()
            today = [a for a in accepted if a.time.date() == day]
            if len(today) >= p.max_per_day:
                continue
            if sum(1 for a in today if a.symbol == c.symbol) >= p.max_per_symbol_day:
                continue
            # consecutive losses already known (closed) today
            closed_today = sorted([a for a in accepted if a.closed_at and a.closed_at <= c.time
                                   and a.closed_at.date() == day and a.outcome in ("WIN", "LOSS", "BREAKEVEN")],
                                  key=lambda a: a.closed_at)
            streak = 0
            for a in reversed(closed_today):
                if a.outcome != "LOSS":
                    break
                streak += 1
            if streak >= p.max_consecutive_losses:
                continue
            accepted.append(c)
            open_until[c.symbol] = c.closed_at or c.time
            last_time[c.symbol] = c.time
        return accepted

    # ------------------------------------------------------------------ full run
    def run(self, symbol: str, base: pd.DataFrame, spec: SymbolSpec, start: Optional[datetime] = None,
            end: Optional[datetime] = None, params: Optional[ReplayParams] = None,
            data: Optional[PreparedData] = None) -> BacktestResult:
        t0 = _time.monotonic()
        data = data or self.prepare(symbol, base, spec)
        rej: Counter = Counter()
        cands = self.collect(data, start, end, rejections=rej)
        p = params or ReplayParams.from_config(self.cfg)
        trades = self.replay(cands, p)
        return self.result(symbol, data, trades, len(cands), rej, start, end, p, _time.monotonic() - t0)

    def result(self, symbol, data, trades, n_cands, rej, start, end, p, runtime) -> BacktestResult:
        stats = compute_stats(trades)
        s = start or data.base.index[0].to_pydatetime()
        e = end or data.base.index[-1].to_pydatetime()
        assumptions = {
            "data": "user-supplied historical bars", "base_tf": "M5",
            "spread": "per-bar spread column" if data.base["spread_price"].notna().any() else
                      f"fixed {data.default_spread / data.spec.pip_size:.2f} pips",
            "slippage_pips": data.spec.slippage_pips, "commission_pips": data.spec.commission_pips,
            "tp_fractions": self.cfg.risk.tp_close_fractions, "tp_r": self.cfg.risk.tp_r_multiples,
            "be_after_tp1": self.cfg.risk.move_sl_to_be_after_tp1,
            "news_filter": "historical calendar CSV" if self.calendar is not None else "NOT applied (no historical calendar)",
            "gold_macro": "not available in backtest (neutral fundamentals score)",
            "params": asdict(p),
        }
        return BacktestResult(symbol, s, e, trades, n_cands, stats, breakdown(trades, "setup_type"),
                              breakdown(trades, "session_bucket"), breakdown(trades, "direction"),
                              rej.most_common(15), assumptions, runtime)
