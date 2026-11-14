"""L1.1a: schema version, settings fingerprint, economic-calendar archive (temporary databases only)."""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.learning import calendar_archive as ca
from core.learning.fingerprint import ALLOWLIST, record_settings, settings_fingerprint, strategy_settings
from core.learning.schema import LEARNING_SCHEMA_VERSION, ensure_learning_schema
from database.database import Database
from database.learning_models import CalendarEventRecord, SchemaVersion, SettingsSnapshot
from database.models import Base
from integrations.economic_calendar import CalendarEvent

OLD_TABLES = ("signals", "notifications", "outbox", "recommendations", "kv_state")
NEW_TABLES = ("schema_versions", "settings_snapshots", "calendar_events")
U = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _tables(path):
    with closing(sqlite3.connect(str(path))) as c:
        return sorted(r[0] for r in c.execute("select name from sqlite_master where type='table'"))


def _old_state(path):
    """Schema and every row of the pre-L1.1a tables."""
    with closing(sqlite3.connect(str(path))) as c:
        schema = {t: c.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0] for t in OLD_TABLES}
        rows = {t: c.execute(f'select * from "{t}" order by 1').fetchall() for t in OLD_TABLES}
    return hashlib.sha256(repr((schema, rows)).encode()).hexdigest()


# ----------------------------------------------------------------------------- schema / compatibility
def test_old_database_opens_with_new_tables_and_existing_rows_unchanged(tmp_path):
    from sqlalchemy import create_engine
    path = tmp_path / "old.db"
    eng = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(eng, tables=[Base.metadata.tables[t] for t in OLD_TABLES])   # a pre-L1.1a database
    eng.dispose()
    with closing(sqlite3.connect(str(path))) as c:
        c.execute("insert into kv_state(key, value, updated_at) values ('last_daily_report', '{\"v\": \"2026-11-01\"}', '2026-11-01')")
        c.execute("insert into notifications(key, kind, sent_at) values ('signal:1', 'signal', '2026-11-01')")
        c.execute("insert into recommendations(created_at, category, text, evidence, status) "
                  "values ('2026-11-01', 'tag', 'x', '{}', 'PENDING_REVIEW')")
        c.commit()
    assert _tables(path) == sorted(OLD_TABLES)
    before = _old_state(path)
    db = Database(f"sqlite:///{path}")
    assert set(NEW_TABLES) <= set(_tables(path))
    assert _old_state(path) == before                                  # schema + rows byte-identical
    assert db.kv_get("last_daily_report") == "2026-11-01"
    with db.session() as s:
        row = s.get(SchemaVersion, "learning")
        assert row.version == LEARNING_SCHEMA_VERSION
    Database(f"sqlite:///{path}")                                      # reopening is idempotent
    with closing(sqlite3.connect(str(path))) as c:
        assert c.execute("select count(*) from schema_versions").fetchone()[0] == 1


def test_schema_version_newer_than_code_is_kept(cfg, caplog):
    db = Database(cfg.database_url)
    with db.session() as s:
        s.get(SchemaVersion, "learning").version = LEARNING_SCHEMA_VERSION + 5
    with caplog.at_level("WARNING"):
        assert ensure_learning_schema(db.Session) == LEARNING_SCHEMA_VERSION + 5
    assert "newer than this code" in caplog.text


# ----------------------------------------------------------------------------- settings fingerprint
SECRETS = {
    "mt5_password": "MyMt5Pass99!", "mt5_login": 24681357, "mt5_server": "VantageInternational-Live",
    "terminal": r"C:\Users\Murad Private\Vantage MT5\terminal64.exe",
    "token": "123456789:FAKEfakeFAKEfakeFAKEfakeFAKEfake00", "chat": "-1009988776655",
    "finnhub": "fh_FAKEKEY_1234567890", "fred": "fred_FAKEKEY_0987654321",
    "db": "sqlite:///C:/Users/Murad Private/fxbot/data/bot.db", "logs": r"C:\Users\Murad Private\fxbot\logs",
    "csv": r"D:\private\history", "backups": r"E:\backups\fxbot", "calendar_csv": r"C:\Users\Murad Private\cal.csv",
}


def _secret_cfg(cfg):
    cfg.mt5.password, cfg.mt5.login, cfg.mt5.server = SECRETS["mt5_password"], SECRETS["mt5_login"], SECRETS["mt5_server"]
    cfg.mt5.terminal_path = SECRETS["terminal"]
    cfg.telegram.bot_token, cfg.telegram.chat_id = SECRETS["token"], SECRETS["chat"]
    cfg.news.finnhub_api_key, cfg.news.fred_api_key = SECRETS["finnhub"], SECRETS["fred"]
    cfg.news.calendar_csv_path = SECRETS["calendar_csv"]
    cfg.database_url, cfg.log_dir, cfg.csv_data_dir = SECRETS["db"], SECRETS["logs"], SECRETS["csv"]
    cfg.runtime.backup_dir = SECRETS["backups"]
    return cfg


def test_fingerprint_never_contains_secrets_ids_or_paths(cfg, tmp_path):
    db = Database(cfg.database_url)                         # real temp DB; cfg values changed afterwards
    _secret_cfg(cfg)
    fp, settings, _ = settings_fingerprint(cfg)
    text = json.dumps(settings)
    for name, value in SECRETS.items():
        assert str(value) not in text, name
    for fragment in ("Murad Private", "C:\\\\", "C:/", "sqlite:", "\\\\fxbot", "password", "bot_token", "chat_id",
                     "api_key", "terminal_path", "login", "database_url", "log_dir", "backup_dir"):
        assert fragment not in text, fragment
    record_settings(db, cfg)
    with db.session() as s:
        stored = json.dumps(s.get(SettingsSnapshot, fp).settings)
    for value in SECRETS.values():
        assert str(value) not in stored


def test_fingerprint_bytes_on_disk_contain_no_secret(cfg):
    db = Database(cfg.database_url)
    path = Path(cfg.database_url.replace("sqlite:///", ""))
    _secret_cfg(cfg)
    record_settings(db, cfg)
    db.engine.dispose()
    blob = path.read_bytes() + (Path(str(path) + "-wal").read_bytes() if Path(str(path) + "-wal").exists() else b"")
    for name, value in SECRETS.items():
        assert str(value).encode() not in blob, name


def test_same_settings_same_hash_and_changes_change_it(cfg):
    fp1, _, _ = settings_fingerprint(cfg)
    fp2, _, _ = settings_fingerprint(cfg)
    assert fp1 == fp2 and len(fp1) == 64
    _secret_cfg(cfg)                                         # secrets and paths do not affect the hash
    assert settings_fingerprint(cfg)[0] == fp1
    cfg.scoring.min_score = cfg.scoring.min_score + 5        # a strategy setting does
    fp3 = settings_fingerprint(cfg)[0]
    assert fp3 != fp1
    cfg.scoring.min_score -= 5
    assert settings_fingerprint(cfg)[0] == fp1


def test_new_config_fields_are_excluded_by_default(cfg):
    cfg.telegram.some_future_secret = "SHOULD-NOT-APPEAR"
    cfg.brand_new_section_secret = "ALSO-NOT"
    text = json.dumps(strategy_settings(cfg))
    assert "SHOULD-NOT-APPEAR" not in text and "ALSO-NOT" not in text
    assert "telegram" not in ALLOWLIST and set(strategy_settings(cfg)) <= set(ALLOWLIST)


def test_record_settings_upserts_one_row_per_fingerprint(cfg):
    db = Database(cfg.database_url)
    a = record_settings(db, cfg)
    b = record_settings(db, cfg)
    cfg.risk.min_rr = cfg.risk.min_rr + 0.5
    c = record_settings(db, cfg)
    with db.session() as s:
        rows = s.query(SettingsSnapshot).all()
    assert a == b != c and len(rows) == 2


def test_code_version_reads_git_without_running_git(tmp_path):
    from core.learning.fingerprint import code_version
    g = tmp_path / ".git" / "refs" / "heads" / "b"
    g.parent.mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/b\n")
    g.write_text("a" * 40 + "\n")
    assert code_version(tmp_path) == "a" * 40
    assert code_version(tmp_path / "nowhere") == "unknown"


# ----------------------------------------------------------------------------- calendar archive
def _ev(title="CPI m/m", t="2026-11-12 13:30", cur="USD", impact="High", forecast="0.3%", actual=None):
    return CalendarEvent(time=U(t), currency=cur, impact=impact, title=title, forecast=forecast,
                         previous="0.2%", actual=actual, source="forexfactory")


def _rows(db):
    with db.session() as s:
        return {r.event_key: r for r in s.query(CalendarEventRecord).all()}


def test_calendar_upsert_inserts_updates_and_never_duplicates(cfg):
    db = Database(cfg.database_url)
    t0 = U("2026-11-10 08:00")
    st = ca.archive_events(db, [_ev(), _ev("NFP", "2026-11-13 13:30"), _ev()], t0)   # same key twice in one feed
    assert st == {"inserted": 2, "updated": 0, "unchanged": 0} and len(_rows(db)) == 2
    st = ca.archive_events(db, [_ev(), _ev("NFP", "2026-11-13 13:30")], t0 + timedelta(hours=1))
    assert st == {"inserted": 0, "updated": 0, "unchanged": 2} and len(_rows(db)) == 2
    st = ca.archive_events(db, [_ev(forecast="0.4%", actual="0.5%")], t0 + timedelta(hours=2))  # values revised
    assert st == {"inserted": 0, "updated": 1, "unchanged": 0}
    r = _rows(db)[_ev().key]
    assert r.forecast == "0.4%" and r.actual == "0.5%" and r.revisions == 1
    assert ca._aware(r.first_seen) == t0 and ca._aware(r.last_changed) == t0 + timedelta(hours=2)
    nfp = _rows(db)[_ev("NFP", "2026-11-13 13:30").key]
    assert ca._aware(nfp.last_seen) == t0 + timedelta(hours=1)        # not in the 3rd feed: kept, not deleted


def test_rescheduled_event_is_kept_and_new_one_added(cfg):
    db = Database(cfg.database_url)
    ca.archive_events(db, [_ev(t="2026-11-12 13:30")], U("2026-11-10 08:00"))
    ca.archive_events(db, [_ev(t="2026-11-13 13:30")], U("2026-11-11 08:00"))
    assert len(_rows(db)) == 2


def test_large_feed_is_archived_in_one_pass(cfg):
    db = Database(cfg.database_url)
    events = [_ev(title=f"E{i}", t=f"2026-11-{10 + i % 5:02d} {i % 24:02d}:00") for i in range(1200)]
    st = ca.archive_events(db, events, U("2026-11-09 08:00"))
    assert st["inserted"] == len({e.key for e in events}) and len(_rows(db)) == st["inserted"]
    assert ca.archive_events(db, events, U("2026-11-09 09:00"))["unchanged"] == st["inserted"]


# ----------------------------------------------------------------------------- orchestrator hooks
class _Cal:
    """Calendar stand-in: refresh_if_needed returns the configured result, like EconomicCalendar."""

    def __init__(self, ok, events):
        self.ok, self.events, self.available = ok, events, ok
        self.last_error = None if ok else "HTTP 429"

    def refresh_if_needed(self, now=None, force=False):
        return self.ok

    def events_between(self, a, b, currencies=None):
        return []


def _bot(cfg, cal, monkeypatch):
    from core.orchestrator import SignalBot
    from integrations.news_provider import NullNewsProvider
    from tests.test_backtest_and_bot import ReplayProvider
    from backtesting.historical_data import synthetic_ohlc
    from core.domain import Timeframe
    prov = ReplayProvider(synthetic_ohlc(n=2000, tf=Timeframe.M1, seed=2))
    cfg.news.calendar_provider = "none"
    bot = SignalBot(cfg, provider=prov, calendar=cal, news=NullNewsProvider())
    scanned = []
    monkeypatch.setattr(bot, "scan_symbol", lambda sym, now: scanned.append(sym))
    monkeypatch.setattr(bot, "update_open_signals", lambda now: None)
    monkeypatch.setattr("core.orchestrator.is_fx_market_open", lambda now: True)
    sent = []
    monkeypatch.setattr(bot.notifier, "send", lambda text, kind="info", key=None: sent.append((kind, key)) or True)
    return bot, scanned, sent, prov


def test_calendar_archived_only_after_successful_refresh(cfg, monkeypatch):
    ok_bot, *_ = _bot(cfg, _Cal(True, [_ev()]), monkeypatch)
    ok_bot.tick(U("2026-11-11 10:00"))
    assert len(_rows(ok_bot.db)) == 1
    fail_bot, *_ = _bot(cfg, _Cal(False, [_ev("Stale cached event")]), monkeypatch)
    fail_bot.tick(U("2026-11-11 10:00"))
    assert _ev("Stale cached event").key not in _rows(fail_bot.db)        # failed refresh -> nothing archived


def test_calendar_archive_failure_does_not_affect_signals_or_telegram(cfg, monkeypatch, caplog):
    bot, scanned, sent, _ = _bot(cfg, _Cal(True, [_ev()]), monkeypatch)
    bot.tick(U("2026-11-11 10:00"))
    baseline_scanned, baseline_sent = list(scanned), list(sent)
    scanned.clear(); sent.clear()

    def boom(*a, **k):
        raise sqlite3.OperationalError("database is locked (simulated)")
    monkeypatch.setattr(ca, "archive_events", boom)
    with caplog.at_level("WARNING"):
        bot.tick(U("2026-11-11 10:00"))                                    # must not raise
    assert scanned == baseline_scanned == bot.cfg.active_symbols          # every symbol still scanned
    assert sent == baseline_sent                                          # no extra (or missing) Telegram message
    assert "calendar archive failed" in caplog.text


def test_settings_recorded_at_start_and_failure_is_isolated(cfg, monkeypatch, caplog):
    bot, *_ = _bot(cfg, _Cal(True, []), monkeypatch)
    fp = bot._record_settings()
    with bot.db.session() as s:
        assert s.get(SettingsSnapshot, fp) is not None
    import core.learning.fingerprint as fpm
    monkeypatch.setattr(fpm, "record_settings", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with caplog.at_level("WARNING"):
        assert bot._record_settings() is None                              # logged, not raised
    assert "settings fingerprint not recorded" in caplog.text


# ----------------------------------------------------------------------------- backup / restore
def test_backup_and_restore_include_the_new_tables(cfg, tmp_path):
    from database.backup import create_backup, inspect_db, restore_backup
    db = Database(cfg.database_url)
    record_settings(db, cfg)
    ca.archive_events(db, [_ev(), _ev("NFP", "2026-11-13 13:30")], U("2026-11-10 08:00"))
    db.engine.dispose()
    src = Path(cfg.database_url.replace("sqlite:///", ""))
    r = create_backup(str(src), cfg.runtime.backup_dir)
    assert r.counts["calendar_events"] == 2 and r.counts["settings_snapshots"] == 1 and r.counts["schema_versions"] == 1
    out = tmp_path / "restored.db"
    res = restore_backup(str(r.path), str(out))
    assert res.counts == r.counts and inspect_db(out)["integrity"] == "ok"
    restored = Database(f"sqlite:///{out}")
    assert len(_rows(restored)) == 2


def test_old_backup_restored_then_opened_gets_new_tables(cfg, tmp_path):
    """A backup taken before L1.1a (no learning tables) still restores; the bot adds the tables on start."""
    from sqlalchemy import create_engine
    from database.backup import create_backup, restore_backup
    old = tmp_path / "pre_l11a.db"
    eng = create_engine(f"sqlite:///{old}")
    Base.metadata.create_all(eng, tables=[Base.metadata.tables[t] for t in OLD_TABLES])
    eng.dispose()
    r = create_backup(str(old), str(tmp_path / "bk"))
    assert "calendar_events" not in r.counts
    out = tmp_path / "restored_old.db"
    restore_backup(str(r.path), str(out))
    Database(f"sqlite:///{out}")
    assert set(NEW_TABLES) <= set(_tables(out))
