"""
Read-only live dashboard for the signal bot.

    python dashboard/server.py --bot C:\\fxbot            # then open http://127.0.0.1:8765

* Separate process: it never imports MetaTrader5, Telegram, the engine or config (no .env is read).
* Read-only: bot.db is opened with SQLite mode=ro; logs and the calendar cache are only read
  (shared read on Windows, so the bot can keep writing and rotating its logs).
* Real data only: every value comes from bot.db, logs/system.log or data/calendar_cache.json and is
  shown with its timestamp; anything not available is null (shown as N/A). Nothing is estimated.
* Local only: binds 127.0.0.1, answers GET only, and rejects foreign Host headers.
  Reach it from your own computer through an SSH tunnel or RDP (see dashboard/README.md).

Live instrument status = the last evaluation result the bot logged for that instrument. The bot logs a
result when it changes (and once after each start), and only the FIRST failing filter, so the pipeline
shows where the last evaluation stopped - not what later filters would have said.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.performance_analyzer import breakdown, compute_stats  # noqa: E402  (pure, standard library only)

HOST = "127.0.0.1"
SYMBOLS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF")
STAGES = ["Data", "Session & hours", "Spread", "Portfolio limits", "HTF trend", "M15 setup (SMC/ICT/PA)",
          "M5 confirmation", "News & macro", "Risk plan", "Score"]
# first rejection text (exact wording from core/signal_engine.py & core/risk_management.py) -> failed stage
STAGE_RULES = [
    (8, r"of ATR \(max|Structural stop too wide|wrong side of entry|Opposing liquidity|Risk/reward"),
    (0, r"Insufficient .* history|ATR unavailable"),
    (1, r"Outside allowed sessions|Weekly open buffer|weekly close"),
    (2, r"Spread .* pips > max|Abnormal spread"),
    (3, r"limit|Maximum active|active signal already|exposure|cooldown|consecutive losses"),
    (4, r"Higher-timeframe trend unclear|HTF bias too weak|H4 and H1 trends conflict"),
    (5, r"No \w+ M15 setup"),
    (6, r"No M5 confirmation"),
    (7, r"News blackout|calendar unavailable|Gold macro contradicts"),
    (9, r"Score \d+ below threshold"),
]
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ (\w+)\s+([\w.]+): (.*)$")
EVAL = re.compile(r"^([A-Z0-9]{6}) (BUY|SELL|-) (?:score=(\d+) )?(SIGNAL|no signal: (.*))$")
SKIP = re.compile(r"^([A-Z0-9]{6}): (latest M5 bar is stale.*|no fresh tick.*|incomplete multi-timeframe data)")
FIXTURE_EVENT = ("2030-01-01T13:30:00+00:00", "USD", "CPI m/m")      # test-suite event, never real


def stage_of(reason: str) -> int:
    for idx, pat in STAGE_RULES:
        if re.search(pat, reason, re.I):
            return idx
    return 0


def _iso(v):
    if v is None:
        return None
    s = str(v)
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    d = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    return d.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------- read-only file access
def read_tail(path: Path, max_bytes: int = 4_000_000) -> str:
    """Last max_bytes of a text file. On Windows the handle allows the bot to write/rename the file."""
    if not path.exists():
        return ""
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k32.CreateFileW.restype = wintypes.HANDLE
        h = k32.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x80, None)    # GENERIC_READ, share all
        if h is None or h == wintypes.HANDLE(-1).value:
            return ""
        fh = os.fdopen(msvcrt.open_osfhandle(h, os.O_RDONLY | getattr(os, "O_BINARY", 0)), "rb")
    else:
        fh = open(path, "rb")
    with fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    return text.split("\n", 1)[1] if size > max_bytes and "\n" in text else text


def db_connect(path: Path):
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------------------- state
def _log_state(bot: Path, to_utc):
    out = {"instruments": {s: None for s in SYMBOLS}, "mt5": {}, "calendar_log": {}, "last_log": None}
    lines = [m.groups() for m in map(LINE.match, read_tail(bot / "logs" / "system.log").splitlines()) if m]
    lines = [x for x in lines if "pytest-of-" not in x[3]]
    for ts, level, name, msg in lines:
        t = to_utc(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        out["last_log"] = t
        if name == "core.orchestrator":
            m = EVAL.match(msg)
            if m:
                sym, direction, score, res, reason = m.groups()
                ok = res == "SIGNAL"
                out["instruments"][sym] = {"time": t, "status": "signal" if ok else "rejected",
                                           "direction": None if direction == "-" else direction,
                                           "score": int(score) if score else None, "reason": reason,
                                           "stage": len(STAGES) if ok else stage_of(reason or "")}
                continue
            m = SKIP.match(msg)
            if m:
                out["instruments"][m.group(1)] = {"time": t, "status": "skipped", "direction": None, "score": None,
                                                  "reason": m.group(2), "stage": 0}
                continue
        if "MT5 connected:" in msg:
            mode = re.search(r"mode=(\w+)", msg)
            out["mt5"] = {"state": "CONNECTED", "since": t, "account_mode": mode.group(1) if mode else None}
        elif "disconnected; reconnect failed" in msg or "Cannot connect to data provider" in msg:
            out["mt5"] = {**out["mt5"], "state": "DISCONNECTED", "since": t}
        elif msg.startswith(("Shutdown requested", "Shutdown complete")):
            out["mt5"] = {**out["mt5"], "state": "BOT STOPPED", "since": t}
        m = re.match(r"Calendar refreshed from (\w+): (\d+) events", msg)
        if m:
            out["calendar_log"] = {"ok": True, "provider": m.group(1), "events": int(m.group(2)), "time": t}
        elif msg.startswith("Calendar refresh failed"):
            out["calendar_log"] = {**out["calendar_log"], "ok": False, "error": msg[:200], "time": t}
    return out


def _calendar(bot: Path, now: datetime):
    p = bot / "data" / "calendar_cache.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"error": "calendar cache unreadable"}
    events = j.get("events") or []
    fixture = [e for e in events if (e.get("time"), e.get("currency"), e.get("title")) == FIXTURE_EVENT]
    upcoming = []
    for e in events:
        try:
            t = datetime.fromisoformat(str(e["time"]).replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if now - timedelta(hours=1) <= t <= now + timedelta(hours=24) and str(e.get("impact")).lower() == "high":
            upcoming.append({"time": _iso(t), "currency": e.get("currency"), "title": e.get("title"),
                             "forecast": e.get("forecast"), "previous": e.get("previous"), "actual": e.get("actual")})
    return {"provider": j.get("provider"), "fetched_at": _iso(j.get("fetched_at")), "events": len(events),
            "test_data": bool(fixture), "upcoming_high": sorted(upcoming, key=lambda e: e["time"])[:12]}


def _db_state(bot: Path):
    con = db_connect(bot / "data" / "bot.db")
    if con is None:
        return None
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "signals" not in tables:
            return {"tables": sorted(tables)}
        cols = ("id, symbol, direction, setup_type, session, score, grade, entry, stop_loss, current_sl, tp1, tp2, "
                "tp3, rr, risk, tp_hits, status, outcome, realized_r, mfe_r, mae_r, created_at, closed_at, context, "
                "events, exit_price")
        rows = [dict(r) for r in con.execute(f"SELECT {cols} FROM signals WHERE source='live' ORDER BY id")]
        for r in rows:
            for k in ("context", "events"):
                try:
                    r[k] = json.loads(r[k]) if r[k] else ({} if k == "context" else [])
                except ValueError:
                    r[k] = {} if k == "context" else []
            r["created_at"], r["closed_at"] = _iso(r["created_at"]), _iso(r["closed_at"])
        snaps = {}
        if "signal_snapshots" in tables:
            for s in con.execute("SELECT signal_id, capture_status, missing, smc, news FROM signal_snapshots"):
                try:
                    smc = json.loads(s["smc"] or "{}")
                except ValueError:
                    smc = {}
                snaps[s["signal_id"]] = {"status": s["capture_status"], "smc": {
                    tf: {"trend": v.get("trend"), "zone": v.get("zone"),
                         "last_structure": (v.get("structure") or [None])[-1]} for tf, v in smc.items()}}
        count = lambda t, where="": (con.execute(f"SELECT COUNT(*) FROM {t} {where}").fetchone()[0]  # noqa: E731
                                     if t in tables else None)
        learning = {
            "snapshots": count("signal_snapshots"),
            "snapshots_complete": count("signal_snapshots", "WHERE capture_status='complete'"),
            "indicator_rows": count("signal_indicator_snapshots"),
            "settings_versions": count("settings_snapshots"),
            "calendar_archive": count("calendar_events"),
            "recommendations_pending": count("recommendations", "WHERE status='PENDING_REVIEW'"),
            "schema": [dict(r) for r in con.execute("SELECT component, version FROM schema_versions")]
            if "schema_versions" in tables else None,
            "code_version": (con.execute("SELECT code_version FROM settings_snapshots ORDER BY last_seen DESC LIMIT 1")
                             .fetchone() or [None])[0] if "settings_snapshots" in tables else None,
        }
        telegram = {
            "last_sent": _iso(con.execute("SELECT MAX(sent_at) FROM notifications").fetchone()[0])
            if "notifications" in tables else None,
            "outbox_pending": count("outbox"),
        }
    finally:
        con.close()
    closed = sorted((r for r in rows if r["outcome"] != "OPEN"), key=lambda r: r["closed_at"] or "")
    curve, eq = [], 0.0
    for r in closed:
        eq += float(r["realized_r"] or 0)
        curve.append({"t": r["closed_at"], "r": round(eq, 3), "id": r["id"]})
    stats = compute_stats(rows).to_dict()
    if stats.get("profit_factor") == float("inf"):
        stats["profit_factor"] = None
    by = lambda k: {g: {kk: (None if vv == float("inf") else vv) for kk, vv in s.to_dict().items()}  # noqa: E731
                    for g, s in breakdown(rows, k).items()}
    for r in rows:
        r["snapshot"] = snaps.get(r["id"])
    return {"open": [r for r in rows if r["outcome"] == "OPEN"], "recent": list(reversed(closed))[:20],
            "stats": stats, "by_symbol": by("symbol"), "by_setup": by("setup_type"), "equity": curve,
            "learning": learning, "telegram": telegram, "total": len(rows)}


def collect_state(bot: Path, log_utc_offset=None) -> dict:
    now = datetime.now(timezone.utc)
    if log_utc_offset is None:
        to_utc = lambda naive: naive.astimezone(timezone.utc)          # noqa: E731  log clock = this PC
    else:
        to_utc = lambda naive: (naive - timedelta(hours=log_utc_offset)).replace(tzinfo=timezone.utc)  # noqa
    errors = []
    try:
        logs = _log_state(bot, to_utc)
    except Exception as exc:
        errors.append(f"logs: {type(exc).__name__}")
        logs = {"instruments": {s: None for s in SYMBOLS}, "mt5": {}, "calendar_log": {}, "last_log": None}
    try:
        db = _db_state(bot)
    except Exception as exc:
        errors.append(f"database: {type(exc).__name__}")
        db = None
    try:
        cal = _calendar(bot, now)
    except Exception as exc:
        errors.append(f"calendar: {type(exc).__name__}")
        cal = None
    inst = {}
    for s, v in logs["instruments"].items():
        if v:
            v = {**v, "age_min": round((now - v["time"]).total_seconds() / 60, 1), "time": _iso(v["time"])}
        inst[s] = v
    mt5 = {k: (_iso(v) if isinstance(v, datetime) else v) for k, v in logs["mt5"].items()} or None
    cl = {k: (_iso(v) if isinstance(v, datetime) else v) for k, v in logs["calendar_log"].items()} or None
    last = logs["last_log"]
    return {"generated_at": _iso(now), "stages": STAGES, "instruments": inst, "mt5": mt5, "calendar_log": cl,
            "calendar": cal, "db": db, "errors": errors,
            "bot_last_log": _iso(last), "bot_last_log_age_min": round((now - last).total_seconds() / 60, 1) if last else None}


# ---------------------------------------------------------------------------- HTTP (GET only, localhost only)
class Handler(BaseHTTPRequestHandler):
    server_version = "fxbot-dashboard"
    sys_version = ""

    def log_message(self, fmt, *args):                       # quiet: no request logging to the console
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; "
                         "form-action 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):          # blocks DNS-rebinding style access
            return self._send(403, b"forbidden", "text/plain")
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            body = json.dumps(self.server.state(), default=str).encode("utf-8")
            return self._send(200, body, "application/json")
        return self._send(404, b"not found", "text/plain")


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bot: Path, port: int, log_utc_offset=None, cache_s: float = 2.0):
        super().__init__((HOST, port), Handler)
        self.bot, self.offset, self.cache_s = bot, log_utc_offset, cache_s
        self._lock, self._cached, self._at = threading.Lock(), None, 0.0

    def state(self) -> dict:
        with self._lock:                                    # at most one read of bot data every cache_s seconds
            if self._cached is None or time.monotonic() - self._at > self.cache_s:
                self._cached, self._at = collect_state(self.bot, self.offset), time.monotonic()
            return self._cached


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only live dashboard (binds 127.0.0.1 only)")
    ap.add_argument("--bot", default=str(ROOT), help="bot folder (default: this repository)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log-utc-offset", type=float, default=None,
                    help="hours the log clock is ahead of UTC (default: this PC's local time zone)")
    a = ap.parse_args(argv)
    srv = DashboardServer(Path(a.bot), a.port, a.log_utc_offset)
    print(f"Dashboard (read-only) on http://{HOST}:{srv.server_address[1]}  - bot folder: {a.bot}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
