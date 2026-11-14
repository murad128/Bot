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
  OK              complete: paged until the terminal had no older bars
  LIMITED         stopped by --max-bars / --max-seconds: NOT the full history
  INTERRUPTED     stopped by Ctrl+C: NOT the full history
  NOT SCANNED     never started (scan interrupted or --max-total-seconds hit)
  INCOMPLETE      paging stopped on an error after some bars were read
  NO BARS         the request succeeded but the terminal has zero bars
  UNAVAILABLE     the first request failed (MT5 error code or the exact
                  TypeError message and argument types are shown)
  NOT FOUND       the symbol does not exist at the broker
Only OK / NO BARS / NOT FOUND rows are complete answers. The report header
says SCAN COMPLETE only if every row is one of those; otherwise it says
SCAN INCOMPLETE and marks every partial first-bar date with "*".

Progress: one line per batch on stderr (symbol, timeframe, bars added,
total, oldest bar, batch time, elapsed). A batch time that grows with depth
means the terminal is slow for deep positions - use the limits below.

MetaTrader5 call contract (package 5.0.x, compiled module):
  copy_rates_from_pos(symbol, timeframe, start_pos, count) - positional only,
  symbol: str, timeframe/start_pos/count: built-in int. Arguments are
  converted to exactly these types before every call. Paging continues until
  the terminal returns no data (a short batch does not end paging), and bars
  are de-duplicated by timestamp, because a new bar forming during paging
  shifts every position by one. If 3 batches in a row add no new bars (a
  terminal that keeps re-serving its oldest bars), paging stops: without this
  guard the scan would never end.

Missing intervals (gaps longer than one bar) are classified using the broker
server clock (Vantage: New York time + 7h, so the trading week runs Monday
00:00 to Friday 24:00 server time):
  weekend         Friday -> Monday gap of at most 3 days          (expected)
  daily break     <= 75 min around server midnight (gold break)    (expected)
  unexpected      everything else - includes market holidays, which
                  you should recognise by their dates in the report

Usage (Windows PowerShell, from the fxbot folder, venv active, MT5 open and
logged in to the DEMO account):
  quick check, one symbol, capped:
    python tools\\history_probe.py --symbols XAUUSD --timeframes M5,H1 --max-bars 50000 --out logs\\probe_quick.md
  incremental, per-timeframe caps and time limits:
    python tools\\history_probe.py --symbols XAUUSD --max-bars-tf M1=200000,M5=200000 --max-seconds 60
  full measurement of one symbol (no bar cap, 10 min per timeframe):
    python tools\\history_probe.py --symbols XAUUSD --max-seconds 600 --out logs\\probe_xauusd.md
Limits: --chunk (bars per request, default 20000), --max-bars (per symbol and
timeframe, 0 = none), --max-bars-tf TF=N,... (overrides per timeframe),
--max-seconds (per symbol and timeframe, default 300, 0 = none),
--max-total-seconds (whole scan, 0 = none), --quiet (no progress lines).
Exit codes: 0 complete, 3 incomplete (limited/interrupted/errors),
2 connection or safety refusal, 130 interrupted by Ctrl+C.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


COMPLETE_STATUSES = ("OK", "NO BARS", "NOT FOUND")
PARTIAL_STATUSES = ("LIMITED", "INTERRUPTED", "INCOMPLETE")     # bars were read, but not all of them


@dataclass
class ProbeResult:
    symbol: str
    timeframe: str
    status: str                       # see module docstring
    broker_symbol: Optional[str] = None
    bars: int = 0
    first: Optional[int] = None       # server epoch seconds of the OLDEST bar scanned
    last: Optional[int] = None
    with_spread: int = 0
    gaps: Dict[str, int] = field(default_factory=lambda: {"weekend": 0, "daily_break": 0, "unexpected": 0})
    largest_unexpected: List[Gap] = field(default_factory=list)
    error: Optional[str] = None       # error text or the reason a scan stopped early
    note: Optional[str] = None
    batches: int = 0
    seconds: float = 0.0

    @property
    def spread_coverage(self) -> Optional[float]:
        return self.with_spread / self.bars if self.bars else None

    @property
    def complete(self) -> bool:
        return self.status in COMPLETE_STATUSES


@dataclass
class Limits:
    chunk: int = 20_000
    max_bars: int = 0                                  # per symbol & timeframe, 0 = no limit
    max_bars_tf: Dict[str, int] = field(default_factory=dict)
    max_seconds: float = 300.0                         # per symbol & timeframe, 0 = no limit
    max_total_seconds: float = 0.0                     # whole scan, 0 = no limit

    def bars_for(self, tf: str) -> int:
        return int(self.max_bars_tf.get(tf, self.max_bars) or 0)


class ScanInterrupted(Exception):
    def __init__(self, partial: "ProbeResult"):
        super().__init__("interrupted")
        self.partial = partial


def stderr_progress(ev: dict) -> None:
    print(f"[{ev['elapsed']:7.1f}s] {ev['symbol']} {ev['timeframe']}: +{ev['added']:,} bars "
          f"(total {ev['total']:,}, oldest {_d(ev['oldest'])}) batch {ev['batch_seconds']:.1f}s",
          file=sys.stderr, flush=True)


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


def probe_one(mt5, broker_symbol: str, symbol: str, tf_name: str, limits: Limits,
              progress: Optional[Callable[[dict], None]] = None, clock=time.monotonic,
              scan_started: Optional[float] = None) -> ProbeResult:
    res = ProbeResult(symbol, tf_name, "OK", broker_symbol=broker_symbol)
    tf = getattr(mt5, f"TIMEFRAME_{tf_name}")
    step = TF_SECONDS[tf_name]
    chunk = max(1, int(limits.chunk))
    max_bars = limits.bars_for(tf_name)
    started = clock()
    scan_started = started if scan_started is None else scan_started
    pos = 0
    newer_first: Optional[int] = None            # oldest bar time already counted (from newer chunks)
    no_progress = 0
    all_gaps: List[Gap] = []
    try:
        while True:
            if limits.max_seconds and clock() - started >= limits.max_seconds:
                res.status, res.error = "LIMITED", f"time limit {limits.max_seconds:g}s per timeframe"
                break
            if limits.max_total_seconds and clock() - scan_started >= limits.max_total_seconds:
                res.status, res.error = "LIMITED", f"total scan time limit {limits.max_total_seconds:g}s"
                break
            want = chunk if not max_bars else min(chunk, max_bars - res.bars)
            t0 = clock()
            try:
                rates = fetch_rates(mt5, broker_symbol, tf, pos, want)
            except _RatesCallError as exc:
                res.error = str(exc)
                res.status = "UNAVAILABLE" if res.bars == 0 else "INCOMPLETE"
                break
            batch_seconds = clock() - t0
            if rates is None:
                code, msg = _last_error(mt5)
                if pos == 0 and code not in MT5_SUCCESS_CODES:
                    res.status, res.error = "UNAVAILABLE", f"MT5 error {code}: {msg}"
                    return res
                break                             # end of history (or empty with success code)
            n = len(rates)
            if n == 0:
                break
            res.batches += 1
            times = np.asarray(rates["time"], dtype=np.int64)
            spreads = np.asarray(rates["spread"]) if "spread" in rates.dtype.names else np.zeros(n)
            if newer_first is not None:
                # a bar formed while paging shifts positions by one: drop bars already counted
                keep = times < newer_first
                times, spreads = times[keep], spreads[keep]
            pos += n                              # advance by what the terminal returned
            if len(times) == 0:
                no_progress += 1
                if no_progress >= 3:              # terminal keeps re-serving bars already counted
                    res.note = "end detected: terminal repeated bars already counted"
                    break
                continue
            no_progress = 0
            res.bars += len(times)
            res.with_spread += int((spreads > 0).sum())
            if res.last is None:
                res.last = int(times[-1])
            res.first = int(times[0])
            all_gaps.extend(classify_gaps(times, step))
            if newer_first is not None:           # gap across the chunk boundary
                all_gaps.extend(classify_gaps(np.array([times[-1], newer_first]), step))
            newer_first = int(times[0])
            if progress:
                progress({"symbol": symbol, "timeframe": tf_name, "added": int(len(times)), "total": res.bars,
                          "oldest": res.first, "batch_seconds": batch_seconds, "elapsed": clock() - scan_started})
            if max_bars and res.bars >= max_bars:
                res.status, res.error = "LIMITED", f"bar limit {max_bars:,}"
                break
    except KeyboardInterrupt:
        res.status, res.error = "INTERRUPTED", "stopped by Ctrl+C"
        _finish(res, all_gaps, started, clock)
        raise ScanInterrupted(res)
    _finish(res, all_gaps, started, clock)
    if res.bars == 0 and res.status == "OK":
        res.status = "NO BARS"
    return res


def _finish(res: ProbeResult, all_gaps: List[Gap], started: float, clock) -> None:
    res.seconds = clock() - started
    for g in all_gaps:
        res.gaps[g.kind] += 1
    res.largest_unexpected = sorted((g for g in all_gaps if g.kind == "unexpected"),
                                    key=lambda g: g.missing_seconds, reverse=True)[:3]


def run_probe(cfg, provider=None, symbols: Optional[List[str]] = None, timeframes=TIMEFRAMES,
              limits: Optional[Limits] = None, progress: Optional[Callable[[dict], None]] = None,
              clock=time.monotonic, chunk: Optional[int] = None,
              max_bars: Optional[int] = None) -> Tuple[int, dict, List[ProbeResult]]:
    """Returns (exit_code, header, results): 0 complete, 3 incomplete, 2 refusal, 130 interrupted."""
    from core.market_data import DataProviderError, MarketDataService
    from integrations.mt5_connector import MT5DataProvider

    limits = limits or Limits()
    if chunk is not None:
        limits.chunk = int(chunk)
    if max_bars is not None:
        limits.max_bars = int(max_bars)
    provider = provider or MT5DataProvider(cfg.mt5, state_path=str(ROOT / "data" / "mt5_server_offset.json"))
    header: dict = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "limits": limits}
    try:
        if not provider.connect():
            header["error"] = "MT5 connection failed (is the terminal running and logged in?)"
            return 2, header, []
    except DataProviderError as exc:              # demo-only guard refusal, package missing, ...
        header["error"] = str(exc)
        return 2, header, []
    results: List[ProbeResult] = []
    plan = [(s, tf) for s in (symbols or cfg.active_symbols) for tf in timeframes]
    scan_started = clock()
    interrupted = False
    try:
        acc = provider.account_summary()
        term = provider.terminal_summary()
        header.update(account=f"{acc.get('trade_mode')} {acc.get('login')}", server=acc.get("server"),
                      terminal_build=term.get("build"))
        md = MarketDataService(provider, cfg)
        mt5 = provider.mt5                        # ReadOnlyMT5 wrapper
        done_symbols = set()
        for sym, tf in plan:
            if interrupted:
                results.append(ProbeResult(sym, tf, "NOT SCANNED", error="scan interrupted"))
                continue
            if limits.max_total_seconds and clock() - scan_started >= limits.max_total_seconds:
                results.append(ProbeResult(sym, tf, "NOT SCANNED", error="total scan time limit reached"))
                continue
            try:
                bs = md.broker_symbol(sym)
                if bs is None:
                    if sym not in done_symbols:
                        results.append(ProbeResult(sym, "-", "NOT FOUND"))
                    done_symbols.add(sym)
                    continue
                done_symbols.add(sym)
                results.append(probe_one(mt5, bs, sym, tf, limits, progress, clock, scan_started))
            except ScanInterrupted as exc:
                results.append(exc.partial)
                interrupted = True
            except KeyboardInterrupt:
                results.append(ProbeResult(sym, tf, "INTERRUPTED", error="stopped by Ctrl+C"))
                interrupted = True
        if not interrupted:
            try:
                off = provider.server_offset(next((r.broker_symbol for r in results if r.broker_symbol), None))
                header["server_time"] = f"UTC{off:+d}h now, mode {provider.time_mode}"
            except Exception:
                header["server_time"] = "offset unknown (market closed?) - dates below are broker server time"
    finally:
        provider.shutdown()
    header["elapsed"] = clock() - scan_started
    header["interrupted"] = interrupted
    if interrupted:
        return 130, header, results
    return (0 if all(r.complete for r in results) else 3), header, results


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
    if code == 2:
        lines += ["", f"**FAILED:** {header.get('error')}"]
        return redact("\n".join(lines) + "\n")
    complete = bool(results) and all(r.complete for r in results) and not header.get("interrupted")
    partial = [r for r in results if not r.complete]
    if complete:
        status_line = "**SCAN COMPLETE** - every symbol/timeframe was paged to the end of the terminal's history."
    else:
        why = ", ".join(f"{sum(r.status == s for r in partial)} {s.lower()}" for s in
                        ("LIMITED", "INTERRUPTED", "NOT SCANNED", "INCOMPLETE", "UNAVAILABLE")
                        if any(r.status == s for r in partial))
        status_line = (f"**SCAN INCOMPLETE** - {len(partial)} of {len(results)} rows not fully measured ({why}). "
                       "First-bar dates marked * are only the oldest bar scanned; older history may exist.")
    lim = header.get("limits")
    lim_txt = ("-" if lim is None else
               f"chunk {lim.chunk:,}, max bars {lim.max_bars or 'none'}"
               + (f" ({', '.join(f'{k}={v:,}' for k, v in lim.max_bars_tf.items())})" if lim.max_bars_tf else "")
               + f", max seconds/TF {lim.max_seconds or 'none'}, max total seconds {lim.max_total_seconds or 'none'}")
    lines += [status_line,
              f"Account: {header.get('account')} on {header.get('server')} | terminal build {header.get('terminal_build')}",
              f"Server time: {header.get('server_time', 'not determined (scan interrupted)')}",
              f"Limits: {lim_txt} | elapsed {header.get('elapsed', 0):,.0f}s",
              "Dates are broker SERVER time. Gaps: weekend / daily break are expected; 'unexpected' includes holidays.",
              "",
              "| Symbol | TF | Status | First bar | Last bar | Bars | Days | Spread data | Gaps wk/brk/unexp | Note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.status == "NOT FOUND":
            lines.append(f"| {r.symbol} | - | NOT FOUND | - | - | - | - | - | - | |")
            continue
        if r.status in ("NO BARS", "UNAVAILABLE", "NOT SCANNED") or (r.status == "INTERRUPTED" and not r.bars):
            detail = r.error or "request succeeded, 0 bars"
            lines.append(f"| {r.symbol} | {r.timeframe} | {r.status} | - | - | 0 | - | - | - | {detail} |")
            continue
        cov = f"{r.spread_coverage:.0%}" if r.spread_coverage is not None else "-"
        g = r.gaps
        star = "" if r.complete else "*"
        note = r.error if not r.complete else (r.note or "")
        lines.append(f"| {r.symbol} | {r.timeframe} | {r.status} | {_d(r.first)}{star} | {_d(r.last)} | {r.bars:,} | "
                     f"{_span_days(r)} | {cov} | {g['weekend']}/{g['daily_break']}/{g['unexpected']} | {note} |")
    unexpected = [(r, gap) for r in results if r.bars for gap in r.largest_unexpected]
    if unexpected:
        lines += ["", "Largest unexpected gaps (check against market holidays):"]
        for r, gap in sorted(unexpected, key=lambda x: x[1].missing_seconds, reverse=True)[:10]:
            lines.append(f"- {r.symbol} {r.timeframe}: {_d(gap.start)} -> {_d(gap.end)} "
                         f"({gap.missing_seconds / 3600:,.1f} h)")
    counts = {s: sum(r.status == s for r in results) for s in
              ("OK", "LIMITED", "INTERRUPTED", "NOT SCANNED", "INCOMPLETE", "NO BARS", "UNAVAILABLE", "NOT FOUND")}
    lines += ["", "Summary: " + ", ".join(f"{v} {k.lower()}" for k, v in counts.items() if v) + ".",
              "If first bars look recent: set MT5 Tools > Options > Charts > Max bars in chart = Unlimited and re-run."]
    return redact("\n".join(lines) + "\n")


def _parse_tf_limits(text: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"expected TF=N, got '{part}'")
        tf, n = part.split("=", 1)
        tf = tf.strip().upper()
        if tf not in TIMEFRAMES:
            raise ValueError(f"unknown timeframe '{tf}'")
        out[tf] = int(n.replace("_", "").strip())
        if out[tf] < 0:
            raise ValueError(f"negative limit for {tf}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only MT5 history availability probe")
    ap.add_argument("--symbols", help="comma-separated canonical symbols (default: all configured)")
    ap.add_argument("--timeframes", default=",".join(TIMEFRAMES), help="comma-separated, default M1..D1")
    ap.add_argument("--out", help="also write the report to this file (e.g. logs\\history_probe.md)")
    ap.add_argument("--chunk", type=int, default=20_000, help="bars per MT5 request (default 20000)")
    ap.add_argument("--max-bars", type=int, default=0, help="max bars per symbol and timeframe (0 = no limit)")
    ap.add_argument("--max-bars-tf", help="per-timeframe bar limits, e.g. M1=200000,M5=500000")
    ap.add_argument("--max-seconds", type=float, default=300.0,
                    help="max seconds per symbol and timeframe (default 300, 0 = no limit)")
    ap.add_argument("--max-total-seconds", type=float, default=0.0, help="max seconds for the whole scan (0 = none)")
    ap.add_argument("--quiet", action="store_true", help="no per-batch progress lines")
    args = ap.parse_args(argv)
    tfs = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    bad = [t for t in tfs if t not in TIMEFRAMES]
    if bad:
        ap.error(f"unknown timeframe(s): {', '.join(bad)}")
    if args.chunk < 1 or args.max_bars < 0 or args.max_seconds < 0 or args.max_total_seconds < 0:
        ap.error("--chunk must be >= 1 and limits must be >= 0")
    try:
        tf_limits = _parse_tf_limits(args.max_bars_tf)
    except ValueError as exc:
        ap.error(f"--max-bars-tf: {exc}")
    from config import load_config
    from core.redact import register_secrets, secrets_from_config
    cfg = load_config()
    register_secrets(secrets_from_config(cfg))
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    limits = Limits(chunk=args.chunk, max_bars=args.max_bars, max_bars_tf=tf_limits,
                    max_seconds=args.max_seconds, max_total_seconds=args.max_total_seconds)
    code, header, results = run_probe(cfg, symbols=symbols, timeframes=tfs, limits=limits,
                                      progress=None if args.quiet else stderr_progress)
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
