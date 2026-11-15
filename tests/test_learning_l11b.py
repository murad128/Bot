"""L1.1b: signal-time market / SMC / news + indicator snapshots (temporary databases, fake data only)."""
import copy
import json
import sqlite3
from contextlib import closing
from datetime import timedelta, timezone

import pandas as pd
import pytest

from backtesting.historical_data import resample, synthetic_ohlc
from core.domain import Timeframe
from core.learning import capture as cap
from core.learning.capture import capture_signal, smc_for_capture, store_snapshot
from core.learning.schema import LEARNING_SCHEMA_VERSION
from core.learning.snapshot import FEATURE_VERSION, build_snapshot, indicator_rows
from core.signal_engine import MarketContext, SignalEngine, prepare_frame
from core.signal_tracker import SignalTracker
from database.database import Database
from database.learning_models import SchemaVersion, SignalIndicatorSnapshot, SignalSnapshot
from integrations.economic_calendar import CalendarEvent
from tests.test_telegram_db import candidate

TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)


@pytest.fixture(scope="module")
def m1():
    return synthetic_ohlc(n=60 * 24 * 40, tf=Timeframe.M1, seed=11, vol=0.00025)


def _ctx(m1, gold_spec, now=None, drop=()):
    """MarketContext built like the orchestrator: closed bars before `now`, prepared frames."""
    now = now or (m1.index[0] + timedelta(days=30)).to_pydatetime().replace(minute=0, second=1)
    frames = {}
    for tf in TFS:
        if tf in drop:
            continue
        df = resample(m1, tf)
        df = df[df.index + tf.delta <= pd.Timestamp(now)].tail(600)      # closed bars only
        frames[tf] = prepare_frame(df)
    px = float(m1[m1.index < pd.Timestamp(now)].iloc[-1]["close"])
    return MarketContext("XAUUSD", gold_spec, now, frames, px, px + 0.2)


class _Cal:
    def __init__(self, events=(), available=True):
        self.events, self.available = list(events), available

    def events_between(self, a, b, currencies=None):
        return [e for e in self.events if a <= e.time <= b and (not currencies or e.currency in currencies)]


def _signal(db, t):
    return SignalTracker(db, _risk_cfg()).register(candidate("XAUUSD", t))


def _risk_cfg():
    from config import RiskConfig
    return RiskConfig()


def _count(db, model):
    with db.session() as s:
        return s.query(model).count()


# ----------------------------------------------------------------------------- pure builders
def test_snapshot_is_json_safe_complete_and_has_no_lookahead(cfg, m1, gold_spec):
    ctx = _ctx(m1, gold_spec)
    engine = SignalEngine(cfg, None)
    smc = smc_for_capture(engine, ctx.symbol, ctx.frames)
    ev = [CalendarEvent(ctx.now + timedelta(hours=2), "USD", "High", "CPI m/m", source="t"),
          CalendarEvent(ctx.now + timedelta(hours=30), "USD", "High", "too far", source="t")]
    cal = _Cal(ev)
    fields, rows = build_snapshot(ctx, smc, TFS, cal.events_between(ctx.now - timedelta(hours=24),
                                                                    ctx.now + timedelta(hours=24)), "ok", 2.0)
    json.dumps(fields, default=str, allow_nan=False)                    # strict JSON: no NaN / numpy
    json.dumps({k: v[1] for k, v in rows.items()}, allow_nan=False)
    assert fields["capture_status"] == "complete" and fields["missing"] == []
    assert fields["feature_version"] == FEATURE_VERSION
    assert set(fields["smc"]) == set(rows) == {tf.value for tf in TFS}
    assert [e["title"] for e in fields["news"]["events"]] == ["CPI m/m"]
    assert fields["news"]["events"][0]["minutes_from_signal"] == 120.0
    for tf in TFS:
        bar_time, values = rows[tf.value]
        df = ctx.frames[tf]
        assert bar_time == df.index[-1].to_pydatetime()
        assert pd.Timestamp(bar_time) + tf.delta <= pd.Timestamp(ctx.now)          # closed before the signal
        for col in ("open", "high", "low", "close", "atr"):
            assert values[col] == pytest.approx(float(df.iloc[-1][col]))
        assert set(values) == set(df.columns)                                     # every existing column, no new one
    # no look-ahead: removing every bar after `now` from the source data changes nothing captured
    past_only = m1[m1.index < pd.Timestamp(ctx.now)]
    assert indicator_rows(_ctx(past_only, gold_spec, ctx.now).frames) == rows


def test_partial_snapshot_lists_what_is_missing(cfg, m1, gold_spec):
    ctx = _ctx(m1, gold_spec, drop=(Timeframe.D1,))
    smc = smc_for_capture(SignalEngine(cfg, None), ctx.symbol, ctx.frames)
    fields, rows = build_snapshot(ctx, smc, TFS, [], "unavailable", None)
    assert fields["capture_status"] == "partial"
    assert fields["missing"] == ["bars:D1", "calendar"]
    assert "D1" not in rows and fields["market"]["spread_atr"] is None


def test_capture_does_not_touch_the_engine_cache(cfg, m1, gold_spec):
    ctx = _ctx(m1, gold_spec)
    engine = SignalEngine(cfg, None)
    cached = engine.smc(ctx.symbol, Timeframe.H1, ctx.frames[Timeframe.H1])
    before = dict(engine._smc_cache)
    out = smc_for_capture(engine, ctx.symbol, ctx.frames)
    assert engine._smc_cache == before                                   # nothing added or replaced
    assert out[Timeframe.H1] is cached                                   # same object the engine used
    fresh = engine._analyzers[Timeframe.H4].analyze(ctx.frames[Timeframe.H4])
    assert out[Timeframe.H4].trend == fresh.trend and len(out[Timeframe.H4].events) == len(fresh.events)


# ----------------------------------------------------------------------------- storage
def test_snapshot_stored_once_per_signal(cfg, m1, gold_spec):
    db = Database(cfg.database_url)
    ctx = _ctx(m1, gold_spec)
    rec = _signal(db, ctx.now)
    engine = SignalEngine(cfg, None)
    assert capture_signal(db, engine, _Cal(), cfg, rec.id, candidate("XAUUSD", ctx.now), ctx, "f" * 64, TFS)
    assert not capture_signal(db, engine, _Cal(), cfg, rec.id, candidate("XAUUSD", ctx.now), ctx, "f" * 64, TFS)
    assert _count(db, SignalSnapshot) == 1 and _count(db, SignalIndicatorSnapshot) == len(TFS)
    with db.session() as s:
        snap = s.get(SignalSnapshot, rec.id)
        assert snap.settings_fingerprint == "f" * 64 and snap.capture_status == "complete"
        assert snap.symbol == "XAUUSD" and snap.capture_ms is not None


def test_calendar_disabled_is_not_reported_as_missing(cfg, m1, gold_spec):
    db = Database(cfg.database_url)
    ctx = _ctx(m1, gold_spec)
    rec = _signal(db, ctx.now)
    cfg.news.calendar_provider = "none"
    capture_signal(db, SignalEngine(cfg, None), _Cal(available=False), cfg, rec.id, candidate(), ctx, None, TFS)
    with db.session() as s:
        snap = s.get(SignalSnapshot, rec.id)
        assert snap.news["status"] == "disabled" and snap.capture_status == "complete"


def test_snapshot_contains_no_secrets(cfg, m1, gold_spec):
    cfg.telegram.bot_token, cfg.mt5.password = "123456789:SECRETsecretSECRETsecret", "MT5-pass-XYZ"
    cfg.mt5.login = 87654321
    db = Database(cfg.database_url)
    ctx = _ctx(m1, gold_spec)
    rec = _signal(db, ctx.now)
    capture_signal(db, SignalEngine(cfg, None), _Cal(), cfg, rec.id, candidate(), ctx, None, TFS)
    db.engine.dispose()
    path = Path_(cfg.database_url.replace("sqlite:///", ""))
    raw = b"".join(f.read_bytes() for f in path.parent.glob(path.name + "*"))     # db + wal
    for secret in (b"SECRETsecret", b"MT5-pass-XYZ", b"87654321"):
        assert secret not in raw


# ----------------------------------------------------------------------------- schema / backup
def test_l11a_database_upgrades_without_changing_existing_rows(tmp_path):
    from sqlalchemy import create_engine
    from database.models import Base
    path = tmp_path / "v1.db"
    old = [t for t in Base.metadata.tables if t not in ("signal_snapshots", "signal_indicator_snapshots")]
    eng = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(eng, tables=[Base.metadata.tables[t] for t in old])
    eng.dispose()
    with closing(sqlite3.connect(str(path))) as c:
        c.execute("insert into schema_versions(component, version, updated_at) values ('learning', 1, '2026-11-01')")
        c.execute("insert into kv_state(key, value, updated_at) values ('k', '{\"v\": 1}', '2026-11-01')")
        c.commit()
        before = {t: c.execute(f'select * from "{t}"').fetchall() for t in old if t != "schema_versions"}
    db = Database(f"sqlite:///{path}")
    with closing(sqlite3.connect(str(path))) as c:
        names = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
        assert {"signal_snapshots", "signal_indicator_snapshots"} <= names
        assert before == {t: c.execute(f'select * from "{t}"').fetchall() for t in before}
    with db.session() as s:
        assert s.get(SchemaVersion, "learning").version == LEARNING_SCHEMA_VERSION == 2


def test_backup_and_restore_include_snapshots(cfg, m1, gold_spec, tmp_path):
    from database.backup import create_backup, restore_backup
    db = Database(cfg.database_url)
    ctx = _ctx(m1, gold_spec)
    rec = _signal(db, ctx.now)
    capture_signal(db, SignalEngine(cfg, None), _Cal(), cfg, rec.id, candidate(), ctx, None, TFS)
    db.engine.dispose()
    src = cfg.database_url.replace("sqlite:///", "")
    res = create_backup(src, str(tmp_path / "bk"))
    assert res.counts["signal_snapshots"] == 1 and res.counts["signal_indicator_snapshots"] == len(TFS)
    target = tmp_path / "restored.db"
    restore_backup(str(res.path), str(target))
    with closing(sqlite3.connect(str(target))) as a, closing(sqlite3.connect(src)) as b:
        for t in ("signal_snapshots", "signal_indicator_snapshots"):
            assert a.execute(f"select * from {t}").fetchall() == b.execute(f"select * from {t}").fetchall()


def test_capture_flag_is_not_part_of_the_strategy_fingerprint(cfg):
    from core.learning.fingerprint import settings_fingerprint
    fp = settings_fingerprint(cfg)[0]
    cfg.runtime.learning_capture = False
    assert settings_fingerprint(cfg)[0] == fp


# ----------------------------------------------------------------------------- live loop: no effect on signals
def _replay(base_cfg, tmp_path, name, capture=True):
    from core.orchestrator import SignalBot
    from integrations.economic_calendar import EconomicCalendar, NullCalendar
    from integrations.news_provider import NullNewsProvider
    from tests.test_backtest_and_bot import ReplayProvider
    cfg = copy.deepcopy(base_cfg)
    cfg.database_url = f"sqlite:///{tmp_path / name / 'bot.db'}"
    cfg.log_dir = str(tmp_path / name / "logs")
    cfg.runtime.learning_capture = capture
    m1 = synthetic_ohlc(n=60 * 24 * 80, tf=Timeframe.M1, seed=9, vol=0.00025)
    cfg.instruments = {"XAUUSD": cfg.instrument("XAUUSD")}
    cfg.analysis.gold_allowed_sessions = []
    cfg.scoring.min_score = 55
    cfg.news.block_if_calendar_unavailable = False
    prov = ReplayProvider(m1)
    bot = SignalBot(cfg, provider=prov, calendar=EconomicCalendar(NullCalendar()), news=NullNewsProvider())
    bot.fund.block_if_unavailable = False
    t = (m1.index[0].to_pydatetime() + timedelta(days=60)).replace(minute=0, second=1)
    for _ in range(12 * 24 * 2):                                         # 2 days of 5-minute ticks
        prov.set_now(t)
        bot.tick(t)
        bot.tick(t)
        t += timedelta(minutes=5)
    from database.models import SignalRecord
    with bot.db.session() as s:
        sigs = [(r.id, r.symbol, r.direction, r.created_at, r.entry, r.stop_loss, r.tp1, r.tp2, r.tp3, r.score,
                 r.outcome, r.closed_at, r.realized_r, r.status, json.dumps(r.context, sort_keys=True),
                 json.dumps(r.events, sort_keys=True)) for r in s.query(SignalRecord).order_by(SignalRecord.id)]
    dry = Path_(cfg.log_dir) / "telegram_dryrun.log"
    import re
    from datetime import datetime
    wall = re.compile(r"%s \d\d:\d\d UTC" % datetime.now(timezone.utc).strftime("%Y-%m-%d"))   # system-message clock
    msgs = [wall.sub("<wall clock>", ln) for ln in dry.read_text(encoding="utf-8").splitlines()
            if not ln.startswith("----- ")]
    return bot, sigs, msgs


def Path_(p):
    from pathlib import Path
    return Path(p)


@pytest.mark.slow
def test_capture_on_off_or_failing_gives_identical_signals_and_messages(cfg, tmp_path, monkeypatch):
    bot_on, sigs_on, msgs_on = _replay(cfg, tmp_path, "on", capture=True)
    bot_off, sigs_off, msgs_off = _replay(cfg, tmp_path, "off", capture=False)
    assert sigs_on, "the replay is expected to produce at least one signal"
    assert sigs_on == sigs_off
    assert msgs_on == msgs_off
    # one snapshot per signal, linked by id, nothing for the disabled run
    with bot_on.db.session() as s:
        snaps = s.query(SignalSnapshot).all()
        assert sorted(x.signal_id for x in snaps) == [r[0] for r in sigs_on]
        assert all(x.capture_status == "complete" or x.missing == ["calendar"] for x in snaps)
        assert all(x.settings_fingerprint for x in snaps)
        n_ind = s.query(SignalIndicatorSnapshot).count()
    assert n_ind == len(sigs_on) * len(TFS)
    from database.learning_models import SettingsSnapshot
    with bot_off.db.session() as s:                                      # LEARNING_CAPTURE=false records nothing
        assert s.query(SignalSnapshot).count() == 0 and s.query(SettingsSnapshot).count() == 0
    # a crashing capture must not change, block or delay anything
    def boom(*a, **k):
        raise RuntimeError("capture exploded")
    monkeypatch.setattr(cap, "capture_signal", boom)
    bot_err, sigs_err, msgs_err = _replay(cfg, tmp_path, "err", capture=True)
    assert sigs_err == sigs_on
    assert msgs_err == msgs_on
    with bot_err.db.session() as s:
        assert s.query(SignalSnapshot).count() == 0
