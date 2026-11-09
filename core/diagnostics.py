"""
Connection & data diagnostics for the live MT5 setup (`python main.py diagnose`).

Runs every check needed before the bot may run against a real terminal and
writes a redacted Markdown report to logs/diagnostics/. It never sends orders
(the connector is wrapped in ReadOnlyMT5) and never prints credentials:
the login is masked, the account holder's name is not read, and the whole
report goes through core.redact before it is written.

Status per check: PASS | WARN (works, but needs attention) | FAIL | SKIP.
"""
from __future__ import annotations

import logging
import platform
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import BASE_DIR, Config
from core.domain import Timeframe
from core.redact import redact

log = logging.getLogger(__name__)

HISTORY_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)


@dataclass
class Check:
    component: str
    status: str
    detail: str
    action: str = ""


@dataclass
class DiagnosticReport:
    started_at: datetime
    checks: List[Check] = field(default_factory=list)
    sections: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, component, status, detail, action=""):
        self.checks.append(Check(component, status, detail, action))
        log.log(logging.ERROR if status == "FAIL" else logging.WARNING if status == "WARN" else logging.INFO,
                "[%s] %s: %s", status, component, redact(detail))

    def section(self, title: str, lines: List[str]) -> None:
        self.sections.setdefault(title, []).extend(lines)

    @property
    def ok(self) -> bool:
        return not any(c.status == "FAIL" for c in self.checks)

    def counts(self) -> Dict[str, int]:
        out = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def to_markdown(self) -> str:
        c = self.counts()
        lines = [f"# MT5 connection diagnostic report",
                 f"Generated {self.started_at:%Y-%m-%d %H:%M:%S} UTC - signal-only, read-only, no orders sent.",
                 "", f"**Result: {'OK' if self.ok else 'PROBLEMS FOUND'}** - "
                     f"{c['PASS']} pass, {c['WARN']} warn, {c['FAIL']} fail, {c['SKIP']} skipped", "",
                 "| # | Component | Status | Detail |", "|---|---|---|---|"]
        for i, ch in enumerate(self.checks, 1):
            lines.append(f"| {i} | {ch.component} | {ch.status} | {ch.detail.replace('|', '/')} |")
        actions = [ch for ch in self.checks if ch.action]
        if actions:
            lines += ["", "## Manual configuration still required", ""]
            lines += [f"- **{ch.component}** ({ch.status}): {ch.action}" for ch in actions]
        for title, body in self.sections.items():
            lines += ["", f"## {title}", ""] + body
        return redact("\n".join(lines) + "\n")

    def to_console(self) -> str:
        w = max(len(c.component) for c in self.checks) if self.checks else 10
        out = [f"{'COMPONENT':<{w}}  STATUS  DETAIL"]
        out += [f"{c.component:<{w}}  {c.status:<6}  {c.detail}" for c in self.checks]
        actions = [c for c in self.checks if c.action]
        if actions:
            out += ["", "MANUAL CONFIGURATION REQUIRED:"] + [f"  - {c.component}: {c.action}" for c in actions]
        c = self.counts()
        out += ["", f"RESULT: {'OK' if self.ok else 'PROBLEMS FOUND'} "
                    f"({c['PASS']} pass / {c['WARN']} warn / {c['FAIL']} fail / {c['SKIP']} skip)"]
        return redact("\n".join(out))


# ---------------------------------------------------------------------------
# data quality
# ---------------------------------------------------------------------------
def classify_gaps(df: pd.DataFrame, tf: Timeframe) -> Dict[str, list]:
    """Gaps larger than one bar: weekend, daily break (~1h around the NY close), other."""
    out = {"weekend": [], "daily_break": [], "other": []}
    if len(df) < 2:
        return out
    step = pd.Timedelta(minutes=tf.minutes)
    idx = df.index
    d = idx[1:] - idx[:-1]
    for prev, nxt, gap in zip(idx[:-1], idx[1:], d):
        if gap <= step:
            continue
        missing = gap - step
        start, end = prev + step, nxt
        if start.dayofweek >= 4 and end.dayofweek in (6, 0) and missing >= pd.Timedelta(hours=40):
            out["weekend"].append((start, end))
        elif missing <= pd.Timedelta(minutes=75) and start.hour in (20, 21, 22, 23):
            out["daily_break"].append((start, end))
        elif tf.minutes >= 1440 and missing <= pd.Timedelta(days=3):
            out["weekend"].append((start, end))
        else:
            out["other"].append((start, end))
    return out


def expected_missing_bars(last_close: datetime, now: datetime, tf: Timeframe,
                          grace: timedelta = timedelta(minutes=5)) -> int:
    """
    Number of bars that should have closed after `last_close` (and before
    now - grace) but are not in the data.

    A bar ending at t is expected only if its last minute is inside market
    hours, so the weekend closure (Friday 17:00 -> Sunday 17:00 New York) produces
    no expected bars, whatever the timeframe. H4/D1 boundaries follow the New
    York wall clock, as Vantage builds them, so DST weeks are counted correctly. Bars whose last minute falls in
    the hour after the New York close are not counted either: gold has a daily
    break there and FX rollover can delay the first bars.
    """
    from zoneinfo import ZoneInfo
    from core.sessions import is_fx_market_open
    ny = ZoneInfo("America/New_York")
    step = timedelta(minutes=tf.minutes)
    limit = now - grace
    # H4/D1 bars sit on the New York wall clock (17:00 + k*4h / 17:00 daily), which moves by one
    # hour in UTC when DST changes; step along that clock so the grid stays correct across DST.
    # Up to H1 every boundary is a whole UTC hour, so UTC steps are exact.
    wall_clock = tf in (Timeframe.H4, Timeframe.D1)
    cursor = last_close.astimezone(ny).replace(tzinfo=None) if wall_clock else last_close
    missing = 0
    while True:
        cursor = cursor + step
        t = cursor.replace(tzinfo=ny).astimezone(timezone.utc) if wall_clock else cursor
        if t > limit:
            break
        last_minute = t - timedelta(minutes=1)
        in_break = last_minute.astimezone(ny).hour == 17   # 17:00-17:59 New York: daily break / rollover
        if is_fx_market_open(last_minute) and not in_break:
            missing += 1
    return missing


def is_history_stale(last_close: datetime, now: datetime, tf: Timeframe, tolerance: int = 1) -> bool:
    """
    True if MORE than `tolerance` expected bars are missing (default: one late
    bar is tolerated, two or more missing bars are flagged - the same policy as
    the original `lag > 2 bars + 5 min` rule, but weekend-correct).
    """
    return expected_missing_bars(last_close, now, tf) > tolerance


def consistency(base: pd.DataFrame, broker: pd.DataFrame, tf: Timeframe, alignment: str, point: float) -> Optional[float]:
    """Share of broker bars whose high/low/close equal the bars rebuilt from M5 (within 1 point)."""
    from backtesting.historical_data import resample
    rebuilt = resample(base, tf, alignment)
    common = broker.index.intersection(rebuilt.index)
    # ignore the first and last rebuilt bar (may be incomplete within the downloaded window)
    common = common[(common > rebuilt.index[0]) & (common < rebuilt.index[-1])]
    if len(common) < 10:
        # bar boundaries never coincide (e.g. wrong H4/D1 alignment) -> 0 % match; too little data -> unknown
        return 0.0 if len(broker) >= 10 and len(rebuilt) >= 10 else None
    a, b = broker.loc[common], rebuilt.loc[common]
    tol = point * 1.01
    ok = ((a["high"] - b["high"]).abs() <= tol) & ((a["low"] - b["low"]).abs() <= tol) & \
         ((a["close"] - b["close"]).abs() <= tol)
    return float(ok.mean())


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run_diagnostics(cfg: Config, provider=None, notifier=None, days: int = 60, send_telegram: bool = True,
                    run_engine: bool = True, history_dir: Optional[str] = None) -> DiagnosticReport:
    from core.market_data import DataProviderError, MarketDataService, validate_ohlc
    from integrations.mt5_connector import MT5DataProvider, ReadOnlyMT5

    rep = DiagnosticReport(datetime.now(timezone.utc))
    history_dir = Path(history_dir or cfg.csv_data_dir)

    # 1. environment ---------------------------------------------------------
    rep.add("Environment", "PASS" if sys.platform == "win32" else "WARN",
            f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
            "" if sys.platform == "win32" else "The MetaTrader5 package only works on Windows next to the terminal.")
    env_file = BASE_DIR / ".env"
    if cfg.mt5.login:
        cred = "login from .env (MT5_LOGIN set; password hidden)"
    else:
        cred = "attach to the account already logged in to the terminal (no password stored - recommended)"
    rep.add("Credentials mode", "PASS", f"{cred}; .env file {'found' if env_file.exists() else 'not found'}")

    # 2. production mode -----------------------------------------------------
    if cfg.data_provider.lower() != "mt5":
        rep.add("Production data source", "FAIL", f"DATA_PROVIDER={cfg.data_provider} (offline/CSV source)",
                "Set DATA_PROVIDER=mt5 in .env.")
        return rep
    if provider is None:
        provider = MT5DataProvider(cfg.mt5, state_path=str(BASE_DIR / "data" / "mt5_server_offset.json"))
    if not isinstance(provider, MT5DataProvider):
        rep.add("Production data source", "FAIL", f"provider is {type(provider).__name__}, not MT5DataProvider")
        return rep
    try:
        raw = provider.mt5._m
    except DataProviderError as exc:
        rep.add("MetaTrader5 package", "FAIL", str(exc), "pip install MetaTrader5 (Windows, same Python as the bot).")
        return rep
    real_module = isinstance(raw, types.ModuleType) and getattr(raw, "__name__", "") == "MetaTrader5"
    module_file = str(getattr(raw, "__file__", "") or "")
    if real_module and "tests" not in Path(module_file).parts:
        rep.add("Production data source", "PASS",
                f"official MetaTrader5 package {getattr(raw, '__version__', '?')} ({module_file}); "
                "no CSV, synthetic or mock provider")
    else:
        rep.add("Production data source", "FAIL",
                f"MT5 module is a {type(raw).__name__} ('{getattr(raw, '__name__', '?')}') - not the official package",
                "Production mode must use the real MetaTrader5 package.")

    # 3. safety ----------------------------------------------------------------
    try:
        provider.mt5.order_send
        rep.add("Trading lock", "FAIL", "order_send is reachable through the connector")
    except PermissionError:
        blocked = []
        for fn in ("order_send", "order_check", "positions_get", "orders_get", "history_deals_get"):
            try:
                getattr(provider.mt5, fn)
            except PermissionError:
                blocked.append(fn)
        rep.add("Trading lock", "PASS", f"read-only wrapper blocks {', '.join(blocked)}; signal-only mode")
    rep.add("Demo-only guard", "PASS" if cfg.mt5.require_demo else "WARN",
            f"MT5_REQUIRE_DEMO={cfg.mt5.require_demo}",
            "" if cfg.mt5.require_demo else "Set MT5_REQUIRE_DEMO=true while testing.")

    # 4. connection + account ------------------------------------------------------
    try:
        connected = provider.connect()
    except DataProviderError as exc:
        rep.add("MT5 connection", "FAIL", str(exc), "Log the terminal in to your Vantage DEMO account.")
        return rep
    if not connected:
        rep.add("MT5 connection", "FAIL", f"initialize() failed: {provider.mt5.last_error()}",
                "Start the MT5 terminal and log in, or set MT5_TERMINAL_PATH to terminal64.exe.")
        return rep
    try:
        term = provider.terminal_summary()
        rep.add("MT5 connection", "PASS" if term.get("connected") else "FAIL",
                f"terminal build {term.get('build')}, connected to trade server: {term.get('connected')}",
                "" if term.get("connected") else "Terminal is not connected to the Vantage server (check internet/login).")
        rep.add("Algo trading button", "PASS",
                f"{'enabled' if term.get('algo_trading_enabled') else 'disabled'} - not required (no orders are sent)")
        acc = provider.account_summary()
        is_vantage = "vantage" in (str(acc.get("server", "")) + str(acc.get("company", ""))).lower()
        rep.add("Account", "PASS" if acc.get("trade_mode") == "DEMO" else "FAIL",
                f"{acc.get('trade_mode')} account {acc.get('login')} on {acc.get('server')} ({acc.get('company')})")
        rep.add("Broker", "PASS" if is_vantage else "WARN",
                "Vantage server detected" if is_vantage else f"server '{acc.get('server')}' does not look like Vantage",
                "" if is_vantage else "Check that the terminal is logged in to the Vantage demo account.")
        rep.add("Account balance", "PASS",
                f"balance {acc.get('balance'):,.2f} {acc.get('currency')}, equity {acc.get('equity'):,.2f} "
                f"{acc.get('currency')}, leverage 1:{acc.get('leverage')}")

        md = MarketDataService(provider, cfg)

        # 5. gold symbol --------------------------------------------------------
        cands = provider.gold_candidates()
        gold = md.broker_symbol("XAUUSD")
        if gold is None:
            rep.add("Gold symbol", "FAIL", f"no XAUUSD symbol resolved; candidates: {[c['name'] for c in cands]}",
                    'Set INSTRUMENT_OVERRIDES={"XAUUSD": {"broker_symbol": "<name from Market Watch>"}}.')
            return rep
        spec = md.spec("XAUUSD")
        info = next((c for c in cands if c["name"] == gold), {})
        rep.add("Gold symbol", "PASS",
                f"'{gold}' (digits {spec.digits}, point {spec.point}, contract {info.get('contract_size')}); "
                f"all XAU/USD symbols: {', '.join(c['name'] for c in cands) or gold}",
                "" if gold == "XAUUSD" else f'Optional: pin it with INSTRUMENT_OVERRIDES={{"XAUUSD": {{"broker_symbol": "{gold}"}}}}')
        missing = [s for s in cfg.active_symbols if s != "XAUUSD" and md.broker_symbol(s) is None]
        found = {s: md.broker_symbol(s) for s in cfg.active_symbols if s not in missing}
        rep.add("Other symbols", "PASS" if not missing else "WARN",
                ", ".join(f"{k}->{v}" for k, v in found.items()) + (f"; NOT FOUND: {', '.join(missing)}" if missing else ""),
                f"Add {', '.join(missing)} to Market Watch or remove them from SYMBOLS." if missing else "")

        # 6. live prices ---------------------------------------------------------
        offset = provider.server_offset(gold)
        tick = md.get_tick("XAUUSD")
        if tick is None:
            rep.add("Live price", "FAIL", "no tick for gold", "Open the XAUUSD chart/Market Watch and retry.")
        else:
            age = (datetime.now(timezone.utc) - tick.time).total_seconds()
            pips = spec.to_pips(tick.spread)
            st = "PASS" if age < cfg.runtime.stale_tick_seconds else "WARN"
            rep.add("Live price", st,
                    f"bid {tick.bid:.{spec.digits}f} / ask {tick.ask:.{spec.digits}f}, spread "
                    f"{tick.spread / spec.point:.0f} points = {pips:.1f} pips (limit {spec.max_spread_pips}), "
                    f"tick age {age:.0f}s",
                    "" if st == "PASS" else "Tick is stale - market closed? Re-run during market hours.")
            if pips > spec.max_spread_pips:
                rep.add("Spread filter", "WARN", f"current spread {pips:.1f} pips exceeds the {spec.max_spread_pips} pip limit",
                        "If this is your account's normal spread, raise max_spread_pips via INSTRUMENT_OVERRIDES.")
        source = "configured (MT5_SERVER_UTC_OFFSET)" if cfg.mt5.server_utc_offset not in ("auto", "") \
            else "auto-detected from a live tick, saved for restarts"
        mode = provider.time_mode
        rep.add("Server time offset", "PASS",
                f"broker server time = UTC{offset:+d}h now, {source}; conversion mode '{mode}'"
                + (" (per-date DST: UTC+3 summer / UTC+2 winter)" if mode == "ny_close" else " (one fixed offset)"),
                "" if mode == "ny_close" or not ("vantage" in str(acc.get("server", "")).lower()) else
                "Vantage servers follow New York DST: set MT5_SERVER_TIME_MODE=ny_close so history before/after "
                "a DST change is converted correctly.")

        # 7. history + quality ---------------------------------------------------
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        frames: Dict[Timeframe, pd.DataFrame] = {}
        history_dir.mkdir(parents=True, exist_ok=True)
        lines = ["| TF | bars | first (UTC) | last closed (UTC) | dup/invalid | weekend | daily break | other gaps |",
                 "|---|---|---|---|---|---|---|---|"]
        for tf in HISTORY_TFS:
            raw_df = provider.get_rates_range(gold, tf, start, end)
            n_raw = len(raw_df)
            dups = int(raw_df.index.duplicated().sum())
            valid = validate_ohlc(raw_df, f"XAUUSD {tf.value}")
            invalid = max(0, n_raw - dups - len(valid))
            from core.market_data import closed_bars
            df = closed_bars(valid, tf, datetime.now(timezone.utc))      # drop the still-forming bar
            future = int((raw_df.index > pd.Timestamp(end)).sum())
            frames[tf] = df
            from backtesting.historical_data import save_csv
            save_csv(df, str(history_dir / f"XAUUSD_{tf.value}.csv"))
            gaps = classify_gaps(df, tf)
            lines.append(f"| {tf.value} | {len(df)} | {df.index[0]:%Y-%m-%d %H:%M} | "
                         f"{df.index[-1]:%Y-%m-%d %H:%M} | {dups}/{invalid} | {len(gaps['weekend'])} | "
                         f"{len(gaps['daily_break'])} | {len(gaps['other'])} |")
            if future:
                rep.add(f"History {tf.value}", "FAIL", f"{future} bars stamped in the future - server offset wrong",
                        "Set MT5_SERVER_UTC_OFFSET explicitly (2 in winter, 3 in summer for Vantage).")
                continue
            last_close = df.index[-1] + pd.Timedelta(minutes=tf.minutes)
            now_utc = datetime.now(timezone.utc)
            stale = tick is not None and (now_utc - tick.time).total_seconds() < 300 \
                and is_history_stale(last_close.to_pydatetime(), now_utc, tf)
            others = gaps["other"]
            status = "FAIL" if stale else "WARN" if len(others) > 3 else "PASS"
            detail = f"{len(df)} bars, last closed bar {last_close:%Y-%m-%d %H:%M} UTC, " \
                     f"{len(others)} unexpected gaps, {dups} duplicates removed"
            rep.add(f"History {tf.value}", status, detail,
                    "Latest bars are missing although the market is open - check the terminal connection." if stale else
                    (f"Inspect unexpected gaps (first: {others[0][0]:%Y-%m-%d %H:%M})." if len(others) > 3 else ""))
            if others:
                lines.append(f"|  | unexpected gaps ({tf.value}): " +
                             "; ".join(f"{a:%m-%d %H:%M}->{b:%m-%d %H:%M}" for a, b in others[:5]) + " | | | | | | |")
        rep.section("XAUUSD history downloaded", lines + ["", f"CSV files saved to `{history_dir}` (git-ignored)."])

        m5 = frames[Timeframe.M5]
        for tf in (Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1):
            share = consistency(m5, frames[tf], tf, cfg.analysis.htf_alignment, spec.point)
            if share is None:
                rep.add(f"Consistency M5->{tf.value}", "SKIP", "not enough overlapping bars")
                continue
            alt = consistency(m5, frames[tf], tf, "utc", spec.point) if tf in (Timeframe.H4, Timeframe.D1) else None
            st = "PASS" if share >= 0.97 else "WARN"
            msg = f"{share:.1%} of broker {tf.value} bars match bars rebuilt from M5 ({cfg.analysis.htf_alignment} alignment)"
            if alt is not None:
                msg += f"; UTC alignment would match {alt:.1%}"
            act = ""
            if st == "WARN" and alt is not None and alt > share:
                act = "Set HTF_ALIGNMENT=utc in .env so backtests match the broker's H4/D1 bars."
            elif st == "WARN":
                act = "Broker bars and M5 disagree - inspect the downloaded CSVs."
            rep.add(f"Consistency M5->{tf.value}", st, msg, act)

        # 8. Telegram -------------------------------------------------------------------
        _check_telegram(cfg, rep, notifier, send_telegram)
        # 9. calendar
        _check_calendar(cfg, rep)
        # 10. engine on real data
        if run_engine:
            _run_engine(cfg, rep, provider, md)
    except DataProviderError as exc:
        rep.add("MT5 data", "FAIL", str(exc))
    finally:
        provider.shutdown()
    return rep


def _check_telegram(cfg, rep, notifier, send):
    from integrations.telegram_bot import TelegramNotifier
    from integrations.telegram_messages import system_message
    if not cfg.telegram.enabled:
        rep.add("Telegram", "FAIL", "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set",
                "Create a bot with @BotFather and put the token and your chat id in .env.")
        return
    n = notifier or TelegramNotifier(cfg.telegram)
    try:
        who = n.test_connection()
    except Exception as exc:
        rep.add("Telegram", "FAIL", f"getMe failed: {redact(str(exc))}", "Check TELEGRAM_BOT_TOKEN.")
        return
    if not send:
        rep.add("Telegram", "PASS", f"{who}; test message skipped (--no-telegram)")
        return
    ok = n.send(system_message("ok", "Diagnostic test message - MT5 signal bot (signal-only, demo)."), kind="system")
    rep.add("Telegram", "PASS" if ok else "FAIL", f"{who}; test message {'delivered' if ok else 'NOT delivered'}",
            "" if ok else "Send /start to your bot from the target chat and check TELEGRAM_CHAT_ID.")


def _check_calendar(cfg, rep):
    from integrations.economic_calendar import build_calendar
    cal = build_calendar(cfg.news, str(BASE_DIR / "data"))
    cal.refresh_if_needed(force=True)
    if not cal.available:
        rep.add("Economic calendar", "FAIL", f"{cfg.news.calendar_provider}: {cal.last_error}",
                "New signals stay blocked without a calendar (safety default). Check internet access to "
                "nfs.faireconomy.media or use CALENDAR_PROVIDER=csv.")
        return
    now = datetime.now(timezone.utc)
    hi = [e for e in cal.events_between(now, now + timedelta(days=7)) if e.is_high]
    rep.add("Economic calendar", "PASS", f"{len(cal.events)} events this week, {len(hi)} high-impact ahead")
    if hi:
        rep.section("Upcoming high-impact events", [f"- {e.time:%a %Y-%m-%d %H:%M} UTC {e.currency} {e.title}" for e in hi[:10]])


def _run_engine(cfg, rep, provider, md):
    """One full analysis pass per symbol on live MT5 data. Nothing is sent or stored."""
    from core.fundamental_analysis import FundamentalAnalyzer
    from core.signal_engine import MarketContext, SignalEngine, prepare_frame
    from integrations.economic_calendar import build_calendar
    from integrations.macro_data import FredClient, MacroDataService
    cal = build_calendar(cfg.news, str(BASE_DIR / "data"))
    cal.refresh_if_needed(force=True)
    fund = FundamentalAnalyzer(cfg.news, cal, MacroDataService(FredClient(cfg.news.fred_api_key), md, cfg.mt5.dxy_symbol))
    engine = SignalEngine(cfg, fund)
    lines = ["| Symbol | Direction | Score | Decision |", "|---|---|---|---|"]
    evaluated = 0
    for sym in cfg.active_symbols:
        if md.broker_symbol(sym) is None:
            continue
        try:
            frames = {}
            for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1):
                df = md.get_candles(sym, tf, cfg.analysis.bars.get(tf.value, 600))
                frames[tf] = prepare_frame(df)
            tick = md.get_tick(sym)
            res = engine.evaluate(MarketContext(sym, md.spec(sym), provider.now(), frames, tick.bid, tick.ask))
            evaluated += 1
            decision = "SIGNAL (not sent - diagnostic)" if res.candidate else "; ".join(res.rejections[:2])
            lines.append(f"| {sym} | {res.direction.value if res.direction else '-'} | "
                         f"{'-' if res.score is None else f'{res.score:.0f}'} | {decision} |")
        except Exception as exc:
            lines.append(f"| {sym} | - | - | ERROR: {redact(str(exc))[:120]} |")
    rep.section("Signal engine on live data (one pass, nothing sent)", lines)
    rep.add("Signal engine", "PASS" if evaluated else "FAIL",
            f"evaluated {evaluated}/{len(cfg.active_symbols)} symbols on live MT5 data (see table)")


def write_report(rep: DiagnosticReport, out_dir: Optional[str] = None) -> Path:
    d = Path(out_dir or (BASE_DIR / "logs" / "diagnostics"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"diagnostic_{rep.started_at:%Y%m%d_%H%M%S}.md"
    p.write_text(rep.to_markdown(), encoding="utf-8")
    return p
