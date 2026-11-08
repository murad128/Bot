"""Regression tests for the issues found in the audit (each one was reproduced before fixing)."""
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from config import MT5Config
from core.domain import Direction, Timeframe
from core.fundamental_analysis import FundamentalAssessment
from core.instance_lock import AlreadyRunning, InstanceLock
from core.market_data import DataProviderError
from core.sessions import minutes_since_week_open, minutes_to_week_close
from core.signal_tracker import SignalTracker
from database.database import Database
from integrations.mt5_connector import MT5DataProvider
from tests.test_mt5_connector import FakeMT5
from tests.test_telegram_db import candidate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- 1. tracker skipped the M1 bar containing the entry moment ------------------------------------
def test_stop_inside_entry_minute_is_detected(cfg):
    db = Database(cfg.database_url)
    t = datetime(2024, 3, 4, 10, 5, 12, tzinfo=timezone.utc)
    rec = SignalTracker(db, cfg.risk).register(candidate(t=t))          # entry 2000, SL 1997
    idx = pd.date_range("2024-03-04 10:05", periods=2, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": [2000, 1998], "high": [2000.5, 1999], "low": [1996, 1997.6],
                         "close": [1998, 1998.5]}, index=idx)
    ev = SignalTracker(db, cfg.risk).update_with_bars(rec, bars, 0.2)
    assert [e.kind for e in ev] == ["SL_HIT"]


def test_bars_before_signal_are_ignored(cfg):
    db = Database(cfg.database_url)
    t = datetime(2024, 3, 4, 10, 5, 12, tzinfo=timezone.utc)
    rec = SignalTracker(db, cfg.risk).register(candidate(t=t))
    idx = pd.date_range("2024-03-04 10:03", periods=2, freq="1min", tz="UTC")   # closed at 10:04 / 10:05
    bars = pd.DataFrame({"open": 2000, "high": 2000.5, "low": 1990, "close": 2000}, index=idx)
    assert SignalTracker(db, cfg.risk).update_with_bars(rec, bars, 0.2) == []


# -- 2. MT5 offset silently assumed 0 when the market was closed ----------------------------------
def _stale(f):
    f.symbol_info_tick = lambda n: SimpleNamespace(time=int(f._server_now()) - 3 * 86400, bid=1, ask=1.1)
    return f


def test_offset_persisted_and_reused_on_weekend_restart(tmp_path):
    st = str(tmp_path / "off.json")
    p = MT5DataProvider(MT5Config(server_utc_offset="auto"), mt5_module=FakeMT5(), state_path=st)
    p.connect()
    p.get_rates("XAUUSD+", Timeframe.M5, 3)
    p2 = MT5DataProvider(MT5Config(server_utc_offset="auto"), mt5_module=_stale(FakeMT5()), state_path=st)
    p2.connect()
    df = p2.get_rates("XAUUSD+", Timeframe.M5, 3)
    assert datetime.now(timezone.utc) - df.index[-1].to_pydatetime() < timedelta(minutes=5)


def test_unknown_offset_raises_instead_of_guessing(tmp_path):
    p = MT5DataProvider(MT5Config(server_utc_offset="auto"), mt5_module=_stale(FakeMT5()),
                        state_path=str(tmp_path / "missing.json"))
    p.connect()
    with pytest.raises(DataProviderError, match="offset unknown"):
        p.get_rates("XAUUSD+", Timeframe.M5, 3)


# -- 3. single instance -----------------------------------------------------------------------------
def test_instance_lock_blocks_second_holder_across_processes(tmp_path):
    path = str(tmp_path / "bot.lock")
    with InstanceLock(path):
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from core.instance_lock import InstanceLock, AlreadyRunning\n"
                "try:\n InstanceLock(%r).acquire(); print('ACQUIRED')\n"
                "except AlreadyRunning: print('BLOCKED')") % (ROOT, path)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30).stdout
        assert "BLOCKED" in out
    InstanceLock(path).acquire().release()        # free again after release


# -- 4. week open / close buffers (DST-aware) -------------------------------------------------------
@pytest.mark.parametrize("now,since,to_close", [
    (datetime(2024, 1, 7, 22, 30, tzinfo=timezone.utc), 30, None),       # winter: open 22:00 UTC
    (datetime(2024, 7, 7, 21, 30, tzinfo=timezone.utc), 30, None),       # summer: open 21:00 UTC
    (datetime(2024, 1, 12, 21, 0, tzinfo=timezone.utc), None, 60),       # Fri, close 22:00 UTC
    (datetime(2024, 7, 12, 19, 0, tzinfo=timezone.utc), None, 120),      # Fri summer, close 21:00 UTC
])
def test_week_buffers(now, since, to_close):
    if since is not None:
        assert minutes_since_week_open(now) == pytest.approx(since)
    if to_close is not None:
        assert minutes_to_week_close(now) == pytest.approx(to_close)
    assert minutes_since_week_open(datetime(2024, 1, 6, 12, tzinfo=timezone.utc)) == float("inf")


# -- 5/6. engine execution gates and gold macro rejection ------------------------------------------
@pytest.fixture(scope="module")
def prepared():
    from backtesting.backtest_engine import Backtester
    from backtesting.historical_data import synthetic_ohlc
    from config import load_config
    from core.domain import SymbolSpec
    cfg = load_config()
    spec = SymbolSpec("XAUUSD", "XAUUSD", 0.01, 2, 0.10, 50.0, True, ["USD"], 0.0, 0.0)
    df = synthetic_ohlc(n=16000, seed=3)
    bt = Backtester(cfg)
    return cfg, spec, bt, bt.prepare("XAUUSD", df, spec)


def _ctx_at(bt, data, spec, i, spread):
    from core.signal_engine import MarketContext
    now_ns = int(data.close_ns[Timeframe.M5][i])
    now = pd.Timestamp(now_ns, unit="ns", tz="UTC").to_pydatetime()
    bid = float(data.frames[Timeframe.M5]["close"].iloc[i])
    return MarketContext("XAUUSD", spec, now, bt._frames_at(data, now_ns), bid, bid + spread)


def test_abnormal_spread_rejected(prepared):
    from core.signal_engine import SignalEngine
    cfg, spec, bt, data = prepared
    cfg.analysis.allowed_sessions = cfg.analysis.gold_allowed_sessions = []
    eng = SignalEngine(cfg, None)
    i = len(data.base) - 500
    res = eng.evaluate(_ctx_at(bt, data, spec, i, spread=1.0))     # synthetic typical spread is 0.25
    assert res.rejections and res.rejections[0].startswith("Abnormal spread")
    res = eng.evaluate(_ctx_at(bt, data, spec, i, spread=0.25))
    assert not any(r.startswith("Abnormal spread") for r in res.rejections)


class _FakeFund:
    def __init__(self, conflicts):
        self.conflicts, self.calls = conflicts, 0

    def assess(self, spec, direction, now):
        self.calls += 1
        return FundamentalAssessment(False, None, 0.3, conflict=self.conflicts > 0, conflicts=self.conflicts,
                                     warnings=["Fundamental conflict: test"])


@pytest.mark.slow
def test_gold_double_macro_conflict_rejects(prepared):
    from core.signal_engine import SignalEngine
    cfg, spec, bt, data = prepared
    cfg.analysis.allowed_sessions = cfg.analysis.gold_allowed_sessions = []
    cfg.scoring.min_score = 0
    fund2, fund1 = _FakeFund(2), _FakeFund(1)
    e2, e1 = SignalEngine(cfg, fund2), SignalEngine(cfg, fund1)
    accepted_single = 0
    start = int(data.close_ns[Timeframe.M5].searchsorted(data.close_ns[Timeframe.H4][230]))
    for i in range(start, len(data.base) - 10):
        before = fund2.calls
        r2 = e2.evaluate(_ctx_at(bt, data, spec, i, 0.25))
        if fund2.calls > before:
            assert r2.candidate is None and "Gold macro contradicts" in r2.rejections[0]
            r1 = e1.evaluate(_ctx_at(bt, data, spec, i, 0.25))
            accepted_single += r1.candidate is not None
        if fund2.calls >= 5 and accepted_single:
            break
    assert fund2.calls >= 1, "engine never reached the fundamentals step on this data"
    assert accepted_single >= 1, "a single conflict should only warn, not reject"
