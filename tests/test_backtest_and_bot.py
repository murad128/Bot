"""
End-to-end tests on SYNTHETIC data. They verify the mechanics (no crashes,
limits respected, consistent accounting). The resulting numbers are not a
measure of strategy quality.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import Backtester, Candidate, ReplayParams
from backtesting.historical_data import load_csv, resample, save_csv, synthetic_ohlc
from backtesting.strategy_optimizer import walk_forward
from core.domain import Timeframe, Tick
from core.market_data import DataProvider, closed_bars


@pytest.fixture(scope="module")
def synth():
    return synthetic_ohlc(n=16000, seed=3)


def test_csv_roundtrip_and_resample(tmp_path, synth):
    p = tmp_path / "XAUUSD_M5.csv"
    save_csv(synth.head(500), str(p))
    df = load_csv(str(p))
    assert len(df) == 500 and df.index.tz is not None
    h1 = resample(df, Timeframe.H1)
    assert h1["high"].iloc[0] == df["high"].iloc[:12].max()


def test_mt5_export_format(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                 "2024.03.04\t02:00:00\t2000.0\t2001.0\t1999.0\t2000.5\t120\t0\t25\n")
    df = load_csv(str(p), server_utc_offset=2, point=0.01)
    assert df.index[0] == pd.Timestamp("2024-03-04 00:00", tz="UTC")
    assert df["spread_price"].iloc[0] == pytest.approx(0.25)


def test_replay_respects_limits():
    t0 = datetime(2024, 3, 4, 8, 0, tzinfo=timezone.utc)

    def c(minutes, outcome="LOSS", dur=30, score=80):
        t = t0 + timedelta(minutes=minutes)
        return Candidate("XAUUSD", t, 0, "BUY", "X", score, {}, 1, 0.9, [1.1, 1.2, 1.3], 2.0, None, "London", {}, [],
                         outcome=outcome, realized_r=-1.0 if outcome == "LOSS" else 1.0,
                         closed_at=t + timedelta(minutes=dur))

    p = ReplayParams(min_score=70, min_rr=2.0, max_per_day=10, max_per_symbol_day=10, cooldown_min=0,
                     max_consecutive_losses=2)
    cands = [c(0), c(10), c(40), c(100), c(200), c(300, score=60)]
    acc = Backtester.replay(cands, p)
    times = [(a.time - t0).total_seconds() / 60 for a in acc]
    # 10 overlaps the open trade; after two losses (closed at 30 and 70) the day is paused
    assert times == [0, 40]


@pytest.mark.slow
def test_backtest_runs_and_is_consistent(cfg, gold_spec, synth):
    bt = Backtester(cfg)
    data = bt.prepare("XAUUSD", synth, gold_spec)
    res = bt.run("XAUUSD", synth, gold_spec, start=synth.index[0].to_pydatetime() + timedelta(days=45), data=data)
    assert res.n_candidates >= res.stats.total
    for a, b in zip(res.trades, res.trades[1:]):
        assert b.time >= a.closed_at                   # one trade at a time per symbol
    for t in res.trades:
        assert t.score >= cfg.scoring.min_score and t.rr >= cfg.risk.min_rr
        assert t.outcome in ("WIN", "LOSS", "BREAKEVEN", "CANCELLED")
        d = 1 if t.direction == "BUY" else -1
        assert (t.entry - t.stop_loss) * d > 0 and all((tp - t.entry) * d > 0 for tp in t.take_profits)
    assert "Most common rejection reasons" in res.to_text()
    wf = walk_forward(bt, data, ReplayParams.from_config(cfg), train_days=20, test_days=10, min_trades=1)
    assert wf.windows and "OUT-OF-SAMPLE" in wf.to_text()


# ----------------------------------------------------------------------------- live loop
class ReplayProvider(DataProvider):
    """In-memory provider replaying synthetic M1 data up to a movable 'now'."""
    name = "replay"

    def __init__(self, m1: pd.DataFrame):
        self.m1 = m1
        self.frames = {Timeframe.M1: m1}
        for tf in Timeframe:
            if tf is not Timeframe.M1:
                self.frames[tf] = resample(m1, tf)
        self._now = m1.index[0].to_pydatetime()

    def set_now(self, t):
        self._now = t

    def now(self):
        return self._now

    def connect(self):
        return True

    def shutdown(self):
        pass

    def is_connected(self):
        return True

    def get_rates(self, s, tf, count):
        df = self.frames[tf]
        return df[df.index <= pd.Timestamp(self._now)].tail(count)   # includes the forming bar

    def get_rates_range(self, s, tf, a, b):
        df = self.frames[tf]
        return df[(df.index >= pd.Timestamp(a)) & (df.index < pd.Timestamp(b))]

    def get_tick(self, s):
        df = self.frames[Timeframe.M1]
        last = df[df.index < pd.Timestamp(self._now)].iloc[-1]
        return Tick(s, float(last["close"]), float(last["close"]) + 0.2, self._now)

    def resolve_symbol(self, inst):
        return inst.symbol


def test_closed_bars_filter():
    idx = pd.date_range("2024-01-01 10:00", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    out = closed_bars(df, Timeframe.M5, datetime(2024, 1, 1, 10, 12, tzinfo=timezone.utc))
    assert list(out.index) == list(idx[:2])


@pytest.mark.slow
def test_live_loop_end_to_end(cfg, tmp_path):
    from core.orchestrator import SignalBot
    from integrations.economic_calendar import EconomicCalendar, NullCalendar
    from integrations.news_provider import NullNewsProvider
    m1 = synthetic_ohlc(n=60 * 24 * 80, tf=Timeframe.M1, seed=9, vol=0.00025)
    cfg.instruments = {"XAUUSD": cfg.instrument("XAUUSD")}
    cfg.analysis.gold_allowed_sessions = []
    cfg.scoring.min_score = 55
    cfg.news.block_if_calendar_unavailable = False
    prov = ReplayProvider(m1)
    bot = SignalBot(cfg, provider=prov, calendar=EconomicCalendar(NullCalendar()), news=NullNewsProvider())
    bot.fund.block_if_unavailable = False
    start = m1.index[0].to_pydatetime() + timedelta(days=60)
    t = start.replace(minute=0, second=1)
    sent_signals = 0
    for _ in range(12 * 24 * 4):            # 4 days of 5-minute ticks
        prov.set_now(t)
        bot.tick(t)
        bot.tick(t)                          # same moment twice: must not duplicate anything
        t += timedelta(minutes=5)
    with bot.db.session() as s:
        from database.models import NotificationLog, SignalRecord
        sigs = s.query(SignalRecord).all()
        keys = [n.key for n in s.query(NotificationLog).all()]
    assert len(keys) == len(set(keys))
    for r in sigs:
        assert keys.count(f"signal:{r.id}") == 1
    report = bot.build_daily_report(t)
    assert "DAILY PERFORMANCE REPORT" in report
    dry = (tmp_path / "logs" / "telegram_dryrun.log")
    if sigs:
        assert dry.exists() and "SIGNAL" in dry.read_text()
