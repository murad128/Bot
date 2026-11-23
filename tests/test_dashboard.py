"""Read-only dashboard: real data only, N/A when missing, GET-only, localhost-only, never writes bot files."""
import ast
import hashlib
import http.client
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard import server as dash
from database.database import Database

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dashboard" / "server.py"


def _log(path, rows):
    """rows: (naive UTC time, logger, message) written in the bot's own log format."""
    with open(path, "a", encoding="utf-8") as f:
        for t, name, msg in rows:
            f.write(f"{t:%Y-%m-%d %H:%M:%S},123 INFO    {name}: {msg}\n")


@pytest.fixture
def bot(tmp_path, cfg):
    """A bot folder with a real bot.db (real schema + tracker), a system.log and a calendar cache."""
    from core.signal_tracker import SignalTracker
    from tests.test_telegram_db import candidate
    root = tmp_path / "fxbot"
    (root / "logs").mkdir(parents=True)
    (root / "data").mkdir()
    db = Database(f"sqlite:///{root / 'data' / 'bot.db'}")
    tr = SignalTracker(db, cfg.risk)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    closed = tr.register(candidate("XAUUSD", now - timedelta(hours=5)))
    tr.cancel(closed, 1999.0, now - timedelta(hours=4), "test close")
    tr.register(candidate("EURUSD", now - timedelta(minutes=30)))          # stays OPEN
    db.engine.dispose()
    n = now.replace(tzinfo=None)
    _log(root / "logs" / "system.log", [
        (n - timedelta(minutes=50), "integrations.mt5_connector", "MT5 connected: terminal build=1 server=X account=***1 mode=DEMO"),
        (n - timedelta(minutes=40), "integrations.economic_calendar", "Calendar refreshed from forexfactory: 97 events"),
        (n - timedelta(minutes=30), "core.orchestrator", "EURUSD BUY score=84 SIGNAL"),
        (n - timedelta(minutes=20), "core.orchestrator", "XAUUSD SELL score=61 no signal: Score 61 below threshold 70"),
        (n - timedelta(minutes=10), "core.orchestrator", "GBPUSD - no signal: HTF bias too weak (+0.12)"),
        (n - timedelta(minutes=9), "core.orchestrator", "USDJPY: latest M5 bar is stale (2026-11-20 22:00:00+00:00) - skipping"),
        (n - timedelta(minutes=8), "core.orchestrator", "AUDUSD BUY no signal: News blackout: USD CPI m/m (High) in 12 min"),
        (n - timedelta(minutes=7), "core.orchestrator", "USDCAD - no signal: Spread is 18% of ATR (max 15%)"),
        (n - timedelta(minutes=5), "database.backup", "Restored /tmp/pytest-of-x/y.db.gz -> z"),   # test-run line: ignored
    ])
    soon = (now + timedelta(hours=2)).isoformat()
    (root / "data" / "calendar_cache.json").write_text(json.dumps({
        "fetched_at": (now - timedelta(minutes=40)).isoformat(), "provider": "forexfactory",
        "events": [{"time": soon, "currency": "USD", "impact": "High", "title": "CPI m/m", "source": "forexfactory"},
                   {"time": soon, "currency": "EUR", "impact": "Low", "title": "Minor", "source": "forexfactory"}]}),
        encoding="utf-8")
    return root


def _hashes(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------- data
def test_state_reports_real_data_and_pipeline_stages(bot):
    s = dash.collect_state(bot)
    i = s["instruments"]
    assert i["EURUSD"]["status"] == "signal" and i["EURUSD"]["stage"] == len(dash.STAGES)
    assert i["XAUUSD"]["stage"] == dash.STAGES.index("Score") and i["XAUUSD"]["score"] == 61
    assert i["GBPUSD"]["stage"] == dash.STAGES.index("HTF trend") and i["GBPUSD"]["direction"] is None
    assert i["USDJPY"]["status"] == "skipped"
    assert i["AUDUSD"]["stage"] == dash.STAGES.index("News & macro")
    assert i["USDCAD"]["stage"] == dash.STAGES.index("Risk plan")          # spread/ATR is a risk-plan check
    assert i["NZDUSD"] is None and i["USDCHF"] is None and i["GBPJPY"] is None   # nothing logged -> N/A
    assert s["mt5"]["state"] == "CONNECTED" and s["mt5"]["account_mode"] == "DEMO"
    assert s["calendar_log"]["provider"] == "forexfactory" and s["calendar_log"]["events"] == 97
    assert s["calendar"]["test_data"] is False and [e["currency"] for e in s["calendar"]["upcoming_high"]] == ["USD"]
    d = s["db"]
    assert [x["symbol"] for x in d["open"]] == ["EURUSD"]
    assert d["open"][0]["tp1"] is not None and d["open"][0]["stop_loss"] is not None
    assert [x["symbol"] for x in d["recent"]] == ["XAUUSD"] and d["recent"][0]["outcome"] == "CANCELLED"
    assert d["stats"]["total"] == 2 and d["stats"]["open"] == 1
    assert d["learning"]["schema"] and d["telegram"]["outbox_pending"] == 0
    json.dumps(s, default=str)                                            # serialisable as served


def test_numbers_match_the_bots_own_performance_code(bot):
    from core.performance_analyzer import compute_stats
    from database.database import Database
    db = Database(f"sqlite:///{bot / 'data' / 'bot.db'}")
    expected = compute_stats(db.all_closed() + db.open_signals())
    db.engine.dispose()
    got = dash.collect_state(bot)["db"]["stats"]
    for k in ("total", "open", "closed", "wins", "losses", "cancelled", "net_r"):
        assert got[k] == getattr(expected, k), k


def test_missing_sources_give_na_and_create_nothing(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    s = dash.collect_state(empty)
    assert s["db"] is None and s["calendar"] is None and s["mt5"] is None and s["bot_last_log"] is None
    assert all(v is None for v in s["instruments"].values())
    assert not any(empty.rglob("*")), "the dashboard must never create files in the bot folder"


def test_test_suite_calendar_event_is_flagged(bot):
    (bot / "data" / "calendar_cache.json").write_text(json.dumps({"fetched_at": "2026-11-21T23:08:57+00:00", "provider": "csv",
        "events": [{"time": "2030-01-01T13:30:00+00:00", "currency": "USD", "impact": "High", "title": "CPI m/m", "source": "csv"}]}))
    assert dash.collect_state(bot)["calendar"]["test_data"] is True


def test_reading_never_modifies_bot_files(bot):
    before = _hashes(bot)
    for _ in range(3):
        dash.collect_state(bot)
    after = {k: v for k, v in _hashes(bot).items() if not k.endswith(("-wal", "-shm"))}
    assert after == {k: v for k, v in before.items() if not k.endswith(("-wal", "-shm"))}


# ----------------------------------------------------------------------------- HTTP
@pytest.fixture
def served(bot):
    srv = dash.DashboardServer(bot, 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, host="127.0.0.1"):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request(method, path, headers={"Host": host})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def test_http_serves_page_and_state_on_localhost_only(served):
    assert served.server_address[0] == "127.0.0.1"
    r, body = _req(served, "GET", "/")
    assert r.status == 200 and b"Signal Bot Live" in body
    csp = r.getheader("Content-Security-Policy")
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp and "frame-ancestors 'none'" in csp
    r, body = _req(served, "GET", "/api/state")
    assert r.status == 200 and json.loads(body)["instruments"]["EURUSD"]["status"] == "signal"


def test_http_is_get_only_and_rejects_foreign_hosts_and_other_paths(served):
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert _req(served, method, "/api/state")[0].status in (405, 501)
    assert _req(served, "GET", "/api/state", host="evil.example.com")[0].status == 403
    for path in ("/../.env", "/.env", "/data/bot.db", "/server.py", "/api/../../.env"):
        assert _req(served, "GET", path)[0].status == 404


# ----------------------------------------------------------------------------- static guarantees
def test_dashboard_never_imports_trading_telegram_or_config():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
           {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = {"MetaTrader5", "requests", "config", "integrations.telegram_bot", "integrations.mt5_connector",
                 "core.orchestrator", "core.signal_engine", "database.database"}
    assert not (mods & forbidden), mods & forbidden
    assert ".env" not in SRC.read_text(encoding="utf-8").replace("no .env is read", "")


def test_dashboard_sql_is_read_only_and_db_opened_read_only():
    src = SRC.read_text(encoding="utf-8")
    sql = " ".join(s.upper() for s in __import__("re").findall(r'"(SELECT[^"]*|[A-Z]+ [^"]*FROM[^"]*)"', src))
    for word in ("INSERT", "UPDATE ", "DELETE", "DROP", "CREATE", "ALTER", "REPLACE", "ATTACH", "PRAGMA"):
        assert word not in sql, word
    assert "mode=ro" in src and src.count("sqlite3.connect(") == 1


def test_page_renders_data_as_text_only():
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html and "insertAdjacentHTML" not in html and "eval(" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html  # no external loads
