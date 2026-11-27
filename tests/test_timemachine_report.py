"""Offline Time Machine report tool: read-only, isolated, no MT5 / Telegram / network, never overwrites."""
import ast
import dataclasses
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from backtesting.historical_data import save_csv, synthetic_ohlc
from core.domain import Timeframe
from tools import timemachine_report as tr

ROOT = Path(__file__).resolve().parents[1]
NEW_MODULES = [ROOT / "tools" / "timemachine_report.py", ROOT / "backtesting" / "timemachine_odds.py"]


def make_bot_folder(tmp_path):
    """A bot folder with a real bot.db (XAUUSD signals + one EURUSD signal without cached data) and a
    separate history folder holding XAUUSD_M5.csv."""
    from config import RiskConfig
    from core.domain import Direction, RiskPlan
    from core.signal_tracker import SignalTracker
    from database.database import Database
    from tests.test_telegram_db import candidate
    root, hist = tmp_path / "fxbot", tmp_path / "history_timemachine" / "20261127"
    (root / "data").mkdir(parents=True)
    hist.mkdir(parents=True)
    bars = synthetic_ohlc(n=16000, tf=Timeframe.M5, seed=3)
    save_csv(bars.assign(spread_price=0.25), str(hist / "XAUUSD_M5.csv"))
    db = Database(f"sqlite:///{root / 'data' / 'bot.db'}")
    tr_ = SignalTracker(db, RiskConfig())
    for k, (day, direction) in enumerate([(48, "SELL"), (50, "BUY"), (52, "SELL")]):
        t = bars.index[0] + pd.Timedelta(days=day, hours=10)
        t = bars.index[bars.index.searchsorted(t)]
        px = float(bars.loc[t, "close"])
        d = 1 if direction == "BUY" else -1
        plan = RiskPlan(entry=round(px, 2), entry_low=round(px, 2), entry_high=round(px, 2),
                        stop_loss=round(px - d * 4.0, 2), take_profits=[round(px + d * 6.0, 2), round(px + d * 12, 2),
                                                                        round(px + d * 18, 2)],
                        risk=4.0, rr_targets=[1.5, 3, 4.5], primary_rr=1.5, room_to_obstacle_r=None)
        c = dataclasses.replace(candidate("XAUUSD", (t + pd.Timedelta(minutes=5, seconds=4)).to_pydatetime()),
                                direction=Direction(direction), plan=plan, atr=3.2)
        tr_.register(c)
    tr_.register(candidate("EURUSD", (bars.index[0] + pd.Timedelta(days=50)).to_pydatetime()))
    db.engine.dispose()
    return root, hist


def _hashes(*folders):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for f in folders for p in sorted(Path(f).rglob("*"))
            if p.is_file() and not p.name.endswith(("-wal", "-shm"))}


@pytest.fixture
def folder(tmp_path):
    return make_bot_folder(tmp_path)


def _run(root, hist, out, *extra):
    return tr.main(["--bot", str(root), "--history-dir", str(hist), "--out", str(out), "--replays", "500", *extra])


# ----------------------------------------------------------------------------- the report
def test_report_states_sample_sizes_dates_seed_and_limitations(folder, tmp_path):
    root, hist = folder
    assert _run(root, hist, tmp_path / "out") == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert js["aggregate"]["signals_total"] == 4 and js["aggregate"]["signals_analysed"] == 3
    assert js["aggregate"]["signals_insufficient"] == 1
    by_symbol = {s["symbol"]: s for s in js["signals"]}
    assert by_symbol["EURUSD"]["status"].startswith("INSUFFICIENT DATA")
    for s in (x for x in js["signals"] if x["symbol"] == "XAUUSD"):
        assert s["status"] == "ok" and s["target_r"] == pytest.approx(1.5) and s["break_even"] == pytest.approx(0.4)
        assert s["history"]["n"] > 0 and s["simulation"]["replays"] == 500 and s["simulation"]["seed"] == 19 + s["id"]
        assert s["actual"]["result"] in ("target", "stop", "neither")
        assert s["sl_distance_atr"] == pytest.approx(4.0 / 3.2)
    for text in ("## Limitations", "Signal dates:", "## Data (cached M5, read-only)", "seed 19 + signal id",
                 "counted as the STOP (conservative)", "INSUFFICIENT DATA", "information only"):
        assert text in md, text
    assert js["data"]["XAUUSD"]["sha256"] and js["data"]["EURUSD"]["status"] == "missing"


def test_report_is_reproducible(folder, tmp_path):
    root, hist = folder
    _run(root, hist, tmp_path / "a")
    _run(root, hist, tmp_path / "b")
    a = json.loads((tmp_path / "a" / "report.json").read_text(encoding="utf-8"))["signals"]
    b = json.loads((tmp_path / "b" / "report.json").read_text(encoding="utf-8"))["signals"]
    assert a == b


# ----------------------------------------------------------------------------- read-only & isolation
def test_bot_db_and_history_files_are_never_modified(folder, tmp_path):
    root, hist = folder
    before = _hashes(root, hist)
    assert _run(root, hist, tmp_path / "out") == 0
    assert _hashes(root, hist) == before
    assert sorted(p.name for p in hist.iterdir()) == ["XAUUSD_M5.csv"]       # nothing written into the history dir


def test_refuses_to_overwrite_an_existing_report(folder, tmp_path):
    root, hist = folder
    out = tmp_path / "out"
    assert _run(root, hist, out) == 0
    before = _hashes(out)
    assert _run(root, hist, out) == 2
    assert _hashes(out) == before


def test_no_network_is_used(folder, tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    root, hist = folder
    assert _run(root, hist, tmp_path / "out") == 0


def test_running_the_report_imports_no_mt5_telegram_engine_or_config(folder, tmp_path):
    root, hist = folder
    code = ("import sys; sys.path.insert(0, %r); from tools import timemachine_report as tr; "
            "rc = tr.main(['--bot', %r, '--history-dir', %r, '--out', %r, '--replays', '100']); "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('MetaTrader5', 'requests', 'config', 'sqlalchemy') "
            "or m in ('core.orchestrator', 'core.signal_engine', 'integrations.telegram_bot', "
            "'integrations.mt5_connector', 'database.database')]; print('RC', rc, 'BAD', bad)"
            % (str(ROOT), str(root), str(hist), str(tmp_path / "out")))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    assert "RC 0 BAD []" in res.stdout, res.stdout + res.stderr


def test_new_modules_have_no_forbidden_imports_and_read_only_sql():
    forbidden = {"MetaTrader5", "requests", "config", "core.orchestrator", "core.signal_engine",
                 "integrations.telegram_bot", "integrations.mt5_connector", "database.database", "dotenv"}
    for path in NEW_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not (mods & forbidden), (path.name, mods & forbidden)
    import re
    src = NEW_MODULES[0].read_text(encoding="utf-8")
    assert "mode=ro" in src and src.count("sqlite3.connect(") == 1
    strings = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    sql = [s for s in strings if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|PRAGMA|ATTACH|REPLACE)\b", s)]
    assert sql, "the tool's SQL statements were not found"
    for s in sql:                                       # every SQL statement in the tool is a read
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|PRAGMA|ATTACH|REPLACE)\b", s), s
