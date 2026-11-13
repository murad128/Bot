#!/usr/bin/env python3
"""
Vantage MT5 history probe - READ-ONLY.

Measures the historical bars the MT5 terminal actually serves for every
configured symbol and timeframe (M1, M5, M15, M30, H1, H4, D1) and prints a
short, shareable report:

  earliest / latest bar, total bars, missing intervals, spread coverage

Safety
* Connects through the bot's existing MT5DataProvider, which enforces the
  demo-only guard (MT5_REQUIRE_DEMO) and wraps the MetaTrader5 module in
  ReadOnlyMT5: only data functions are reachable, every trading function
  raises PermissionError. This tool calls nothing but symbol lookup,
  account/terminal info and copy_rates_from_pos.
* Writes nothing unless --out is given. Never touches .env, settings,
  the database or strategy code.
* The login is masked, the account holder's name is never read, and the
  whole report passes through core.redact before it is printed or written.

How bars are measured
* copy_rates_from_pos is paged backwards from the newest bar until the
  terminal returns no more data. This is the exact set the terminal serves.
* The terminal serves at most "Max bars in chart" bars per symbol and
  timeframe (MT5: Tools > Options > Charts). Set it to Unlimited first, or
  the earliest bar will look later than the broker's real history.

Status per symbol/timeframe
  OK              bars returned
  NO BARS         the request succeeded but the terminal has zero bars
  UNAVAILABLE     the first request failed (MT5 error code or the exact
                  TypeError message and argument types are shown)
  INCOMPLETE      paging stopped on an error after some bars were read;
                  the bars counted so far are shown
  NOT FOUND       the symbol does not exist at the broker

MetaTrader5 call contract (package 5.0.x, compiled module):
  copy_rates_from_pos(symbol, timeframe, start_pos, count) - positional only,
  symbol: str, timeframe/start_pos/count: built-in int. Arguments are
  converted to exactly these types before every call. Paging continues until
  the terminal returns no data (a short batch does not end paging), and bars
  are de-duplicated by timestamp, because a new bar forming during paging
  shifts every position by one.

Missing intervals (gaps longer than one bar) are classified using the broker
server clock (Vantage: New York time + 7h, so the trading week runs Monday
00:00 to Friday 24:00 server time):
  weekend         Friday -> Monday gap of at most 3 days          (expected)
  daily break     <= 75 min around server midnight (gold break)    (expected)
  unexpected      everything else - includes market holidays, which
                  you should recognise by their dates in the report

Usage (Windows PowerShell, from the fxbot folder, venv active, MT5 open and
logged in to the DEMO account):
  python tools\\history_probe.py
  python tools\\history_probe.py --symbols XAUUSD,EURUSD --timeframes M5,H1
  python tools\\history_probe.py --out logs\\history_probe.md
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # allow `python tools\history_probe.py`
    sys.path.insert(0, str(ROOT))

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}
MT5_SUCCESS_CODES = (1,)                          # RES_S_OK
DAY = 86400


@dataclass
class Gap:
    start: int        # server epoch of the bar before the gap
    end: int          # server epoch of the first bar after the gap
    kind: str         # weekend | daily_break | unexpected

    @property
    def missing_seconds(self) -> int:
        return self.end - self.start


@dataclass
class ProbeResult:
    symbol: str
    timeframe: str
    status: str                       # OK | NO BARS | UNAVAILABLE | NOT FOUND
    broker_symbol: Optional[str] = None
    bars: int = 0
    first: Optional[int] = None       # server epoch seconds
    last: Optional[int] = None
    with_spread: int = 0
    gaps: Dict[str, int] = field(default_factory=lambda: {"weekend": 0, "daily_break": 0, "unexpected": 0})
    largest_unexpected: List[Gap] = field(default_factory=list)
    error: Optional[str] = None
    capped: bool = False

    @property
    def spread_coverage(self) -> Optional[float]:
        return self.with_spread / self.bars if self.bars else None


# ---------------------------------------------------------------------------
# gap classification on the broker server clock
# ---------------------------------------------------------------------------
def _weekday(t: np.ndarray) -> np.ndarray:
    return ((t // DAY) + 3) % 7                    # 1970-01-01 was a Thursday; Monday = 0


def classify_gaps(times: np.ndarray, step: int) -> List[Gap]:
    """Gaps between consecutive bars (server epoch seconds, ascending) longer than one bar."""
    if len(times) < 2:
        return []
    t = np.asarray(times, dtype=np.int64)
    d = np.diff(t)
    idx = np.flatnonzero(d > step)
    out: List[Gap] = []
    for i in idx:
        a = int(t[i]) + step                      # first missing bar
        b = int(t[i + 1])
        missing = b - a
        wa, wb = int(_weekday(np.array([a]))[0]), int(_weekday(np.array([b]))[0])
        ha, hb = (a % DAY) // 3600, (b % DAY) // 3600
        if wa in (4, 5, 6) and wb in (0, 6) and missing <= 3 * DAY:
            kind = "weekend"
        elif missing <= 75 * 60 and (ha >= 22 or hb <= 2):
            kind = "daily_break"
        else:
            kind = "unexpected"
        out.append(Gap(a, b, kind))
    return out


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
def _last_error(mt5) -> Tuple[Optional[int], str]:
    try:
        err = mt5.last_error()
        code, msg = (err[0], str(err[1])) if isinstance(err, (tuple, list)) and len(err) >= 2 else (None, str(err))
        return code, msg
    except Exception as exc:  # pragma: no cover - defensive
        return None, str(exc)


class _RatesCallError(Exception):
    """copy_rates_from_pos raised (e.g. TypeError from the compiled MT5 module)."""


def fetch_rates(mt5, broker_symbol, tf, pos, count):
    """
    One copy_rates_from_pos call with exactly the argument types the compiled
    MetaTrader5 module accepts: (str, int, int, int), positional. Raises
    _RatesCallError with the exception text and the argument types if the
    package rejects the call, so the report shows the real cause.
    """
    args = (str(broker_symbol), int(tf), int(pos), int(count))
    if args[2] < 0 or args[3] < 1:
        raise _RatesCallError(f"invalid paging values start_pos={args[2]} count={args[3]}")
    try:
        return mt5.copy_rates_from_pos(*args)
    except TypeError as exc:
        types = ", ".join(type(a).__name__ for a in (broker_symbol, tf, pos, count))
        raise _RatesCallError(f"TypeError: {exc} (argument types: {types})") from exc


def probe_one(mt5, broker_symbol: str, symbol: str, tf_name: str, chunk: int, max_bars: int) -> ProbeResult:
    res = ProbeResult(symbol, tf_name, "OK", broker_symbol=broker_symbol)
    tf = getattr(mt5, f"TIMEFRAME_{tf_name}")
    step = TF_SECONDS[tf_name]
    pos = 0
    newer_first: Optional[int] = None            # oldest bar time already counted (from newer chunks)
    all_gaps: List[Gap] = []
    while True:
        try:
            rates = fetch_rates(mt5, broker_symbol, tf, pos, chunk)
        except _RatesCallError as exc:
            res.error = str(exc)
            res.status = "UNAVAILABLE" if res.bars == 0 else "INCOMPLETE"
            break
        if rates is None:
            code, msg = _last_error(mt5)
            if pos == 0 and code not in MT5_SUCCESS_CODES:
                res.status, res.error = "UNAVAILABLE", f"MT5 error {code}: {msg}"
                return res
            break                                 # end of history (or empty with success code)
        n = len(rates)
        if n == 0:
            break
        times = np.asarray(rates["time"], dtype=np.int64)
        spreads = np.asarray(rates["spread"]) if "spread" in rates.dtype.names else np.zeros(n)
        if newer_first is not None:
            # a bar formed while paging shifts positions by one: drop bars already counted
            keep = times < newer_first
            times, spreads = times[keep], spreads[keep]
        pos += n                                  # advance by what the terminal returned
        if len(times) == 0:
            continue
        res.bars += len(times)
        res.with_spread += int((spreads > 0).sum())
        if res.last is None:
            res.last = int(times[-1])
        res.first = int(times[0])
        all_gaps.extend(classify_gaps(times, step))
        if newer_first is not None:               # gap across the chunk boundary
            all_gaps.extend(classify_gaps(np.array([times[-1], newer_first]), step))
        newer_first = int(times[0])
        if res.bars >= max_bars:
            res.capped = True
            break
    if res.bars == 0 and res.status == "OK":
        res.status = "NO BARS"
        return res
    for g in all_gaps:
        res.gaps[g.kind] += 1
    res.largest_unexpected = sorted((g for g in all_gaps if g.kind == "unexpected"),
                                    key=lambda g: g.missing_seconds, reverse=True)[:3]
    return res


def run_probe(cfg, provider=None, symbols: Optional[List[str]] = None, timeframes=TIMEFRAMES,
              chunk: int = 100_000, max_bars: int = 20_000_000) -> Tuple[int, dict, List[ProbeResult]]:
    """Returns (exit_code, header, results). Exit 2 = connection or safety refusal."""
    from core.market_data import DataProviderError, MarketDataService
    from integrations.mt5_connector import MT5DataProvider

    provider = provider or MT5DataProvider(cfg.mt5, state_path=str(ROOT / "data" / "mt5_server_offset.json"))
    header: dict = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    try:
        if not provider.connect():
            header["error"] = "MT5 connection failed (is the terminal running and logged in?)"
            return 2, header, []
    except DataProviderError as exc:              # demo-only guard refusal, package missing, ...
        header["error"] = str(exc)
        return 2, header, []
    results: List[ProbeResult] = []
    try:
        acc = provider.account_summary()
        term = provider.terminal_summary()
        header.update(account=f"{acc.get('trade_mode')} {acc.get('login')}", server=acc.get("server"),
                      terminal_build=term.get("build"))
        md = MarketDataService(provider, cfg)
        mt5 = provider.mt5                        # ReadOnlyMT5 wrapper
        for sym in symbols or cfg.active_symbols:
            bs = md.broker_symbol(sym)
            if bs is None:
                results.append(ProbeResult(sym, "-", "NOT FOUND"))
                continue
            for tf in timeframes:
                results.append(probe_one(mt5, bs, sym, tf, chunk, max_bars))
        try:
            off = provider.server_offset(next((r.broker_symbol for r in results if r.broker_symbol), None))
            header["server_time"] = f"UTC{off:+d}h now, mode {provider.time_mode}"
        except Exception:
            header["server_time"] = "offset unknown (market closed?) - dates below are broker server time"
    finally:
        provider.shutdown()
    return 0, header, results


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _d(t: Optional[int]) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if t is not None else "-"


def _span_days(r: ProbeResult) -> str:
    return f"{(r.last - r.first) / DAY:,.0f}" if r.first is not None and r.last is not None else "-"


def format_report(code: int, header: dict, results: List[ProbeResult]) -> str:
    from core.redact import redact
    lines = ["# MT5 history probe (read-only)", f"Generated {header.get('generated')}"]
    if code != 0:
        lines += ["", f"**FAILED:** {header.get('error')}"]
        return redact("\n".join(lines) + "\n")
    lines += [f"Account: {header.get('account')} on {header.get('server')} | terminal build {header.get('terminal_build')}",
              f"Server time: {header.get('server_time')}",
              "Dates are broker SERVER time. Gaps: weekend / daily break are expected; 'unexpected' includes holidays.",
              "",
              "| Symbol | TF | Status | First bar | Last bar | Bars | Days | Spread data | Gaps wk/brk/unexp |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.status == "NOT FOUND":
            lines.append(f"| {r.symbol} | - | NOT FOUND | - | - | - | - | - | - |")
            continue
        if r.status in ("NO BARS", "UNAVAILABLE"):
            detail = r.error or "request succeeded, 0 bars"
            lines.append(f"| {r.symbol} | {r.timeframe} | {r.status} | - | - | 0 | - | - | {detail} |")
            continue
        cov = f"{r.spread_coverage:.0%}" if r.spread_coverage is not None else "-"
        g = r.gaps
        bars = f"{r.bars:,}" + (" (capped)" if r.capped else "")
        lines.append(f"| {r.symbol} | {r.timeframe} | {r.status} | {_d(r.first)} | {_d(r.last)} | {bars} | "
                     f"{_span_days(r)} | {cov} | {g['weekend']}/{g['daily_break']}/{g['unexpected']}"
                     + (f" - stopped: {r.error}" if r.status == "INCOMPLETE" else "") + " |")
    unexpected = [(r, gap) for r in results if r.status in ("OK", "INCOMPLETE") for gap in r.largest_unexpected]
    if unexpected:
        lines += ["", "Largest unexpected gaps (check against market holidays):"]
        for r, gap in sorted(unexpected, key=lambda x: x[1].missing_seconds, reverse=True)[:10]:
            lines.append(f"- {r.symbol} {r.timeframe}: {_d(gap.start)} -> {_d(gap.end)} "
                         f"({gap.missing_seconds / 3600:,.1f} h)")
    n_ok = sum(r.status == "OK" for r in results)
    lines += ["", f"Summary: {n_ok} OK, {sum(r.status == 'INCOMPLETE' for r in results)} incomplete, "
                  f"{sum(r.status == 'NO BARS' for r in results)} no bars, "
                  f"{sum(r.status == 'UNAVAILABLE' for r in results)} unavailable, "
                  f"{sum(r.status == 'NOT FOUND' for r in results)} symbols not found.",
              "If first bars look recent: set MT5 Tools > Options > Charts > Max bars in chart = Unlimited and re-run."]
    return redact("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only MT5 history availability probe")
    ap.add_argument("--symbols", help="comma-separated canonical symbols (default: all configured)")
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES), help="comma-separated, default M1..D1")
    ap.add_argument("--out", help="also write the report to this file (e.g. logs\\history_probe.md)")
    ap.add_argument("--chunk", type=int, default=100_000, help=argparse.SUPPRESS)
    ap.add_argument("--max-bars", type=int, default=20_000_000, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    bad = [t for t in tfs if t not in TIMEFRAMES]
    if bad:
        ap.error(f"unknown timeframe(s): {', '.join(bad)}")
    from config import load_config
    from core.redact import register_secrets, secrets_from_config
    cfg = load_config()
    register_secrets(secrets_from_config(cfg))
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    code, header, results = run_probe(cfg, symbols=symbols, timeframes=tfs, chunk=args.chunk, max_bars=args.max_bars)
    text = format_report(code, header, results)
    try:
        print(text)
    except UnicodeEncodeError:                  # redirected Windows console with a legacy code page
        print(text.encode("ascii", "replace").decode("ascii"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
