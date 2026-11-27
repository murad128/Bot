"""
Offline Time Machine odds report for past live signals.

    python tools/timemachine_report.py --history-dir data\\history_timemachine\\20261127

Reads bot.db READ-ONLY (SQLite mode=ro) and cached M5 CSVs (<SYMBOL>_M5.csv, written by
`main.py download` into a folder you choose). Writes a Markdown and a JSON report into a NEW folder
(default logs/reports/timemachine/<UTC time>/) and refuses to overwrite anything.

It never connects to MT5, never sends Telegram messages, makes no network calls, and never imports the
live engine, the orchestrator, config or .env. It changes no signal, filter, score, Entry, SL, TP or lot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtesting import timemachine_odds as tm  # noqa: E402
from backtesting.historical_data import load_csv  # noqa: E402  (pure pandas reader)

COLS = ("id, symbol, direction, setup_type, score, entry, stop_loss, tp1, tp2, tp3, risk, rr, atr, spread, "
        "status, outcome, tp_hits, realized_r, created_at, closed_at")
LIMITATIONS = [
    "Small samples: a handful of live signals cannot establish an edge; treat every number as a description.",
    "The simulation draws each M5 bar independently from the same time of day, so it removes volatility "
    "clustering and trends that last longer than one bar. It is a model built from past bars, not a forecast.",
    "History replay uses the signal's original SL and TP distances in price. The same distance is a different "
    "risk when volatility differs: compare the SL distance in ATR (from the signal) across days.",
    "M5 bars are coarser than the bot's own M1 tracking: a bar touching both SL and target is counted as the "
    "stop (conservative) and listed as ambiguous; the real order inside that bar is unknown.",
    "Spread comes from each bar's recorded spread (MT5 bar spread), else the signal's spread. Commission and "
    "slippage are not included; a gap through the stop counts as the stop price, a real fill can be worse.",
    "Past bars contain no future news: an event the lookback never saw (central bank, CPI, geopolitics) can "
    "move price in ways no replay shows.",
    "Only the target (TP1 by default) vs the initial SL is measured, not the bot's partial closes, break-even "
    "stop or realised R. 'Neither' means still open when the horizon (clock time) ended.",
    "This report is information only. It does not change any signal, filter, score, Entry, SL, TP or lot size.",
]


def _utc(v):
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def _code_version():
    try:
        head = (ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = ROOT / ".git" / head.split(" ", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()[:12] if ref.exists() else "unknown"
        return head[:12]
    except OSError:
        return "unknown"


def read_signals(db: Path, source: str, ids=None):
    if not db.exists():
        raise SystemExit(f"bot.db not found: {db}")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in con.execute(f"SELECT {COLS} FROM signals WHERE source=? ORDER BY id", (source,))]
    finally:
        con.close()
    if ids:
        rows = [r for r in rows if r["id"] in set(ids)]
    for r in rows:
        r["created_at"] = _utc(r["created_at"])
        r["closed_at"] = _utc(r["closed_at"]) if r["closed_at"] else None
    return rows


def read_history(history_dir: Path, symbol: str):
    path = history_dir / f"{symbol}_M5.csv"
    if not path.exists():
        return None, {"file": str(path), "status": "missing"}
    raw = path.read_bytes()
    bars = tm.prepare_bars(load_csv(str(path)))
    info = {"file": str(path), "status": "ok", "sha256": hashlib.sha256(raw).hexdigest(), "bars": len(bars),
            "first": bars.index[0].isoformat() if len(bars) else None,
            "last": (bars.index[-1] + tm.BAR).isoformat() if len(bars) else None}
    return bars, info


def bot_result(s):
    if s["outcome"] == "OPEN":
        return "open"
    if (s["tp_hits"] or 0) >= 1:
        return "TP1 reached"
    if s["outcome"] == "LOSS":
        return "SL before TP1"
    return s["outcome"].lower()


def analyse(sig, bars, a):
    horizon = timedelta(hours=a.horizon_hours)
    risk = abs(sig["entry"] - sig["stop_loss"])
    reward = abs(sig[a.target] - sig["entry"])
    k = reward / risk if risk else None
    p_be = tm.break_even(k)
    out = {"id": sig["id"], "symbol": sig["symbol"], "direction": sig["direction"], "setup": sig["setup_type"],
           "created_at": sig["created_at"].isoformat(), "entry": sig["entry"], "stop_loss": sig["stop_loss"],
           "target": sig[a.target], "target_r": k, "break_even": p_be,
           "sl_distance": risk, "sl_distance_atr": (risk / sig["atr"]) if sig.get("atr") else None,
           "bot_result": bot_result(sig), "seed": a.seed + sig["id"]}
    if bars is None:
        out["status"] = "INSUFFICIENT DATA: no cached M5 file for this symbol"
        return out
    first, last = bars.index[0], bars.index[-1] + tm.BAR
    need_from = sig["created_at"] - timedelta(days=a.lookback_days)
    notes = []
    if first > need_from:
        notes.append(f"history starts {first.date()}, after the lookback start {need_from.date()}: fewer days used")
    if last < sig["created_at"] + horizon:
        notes.append("data ends before the horizon: the actual path may be incomplete")
    if tm.entry_index(bars, sig["created_at"]) is None or first > sig["created_at"]:
        out["status"] = "INSUFFICIENT DATA: the signal time is outside the cached bars"
        return out
    act = tm.actual_path(bars, sig, horizon, a.target)
    hist = tm.history_replay(bars, sig, horizon, a.lookback_days, a.target)
    sim = tm.simulate(bars, sig, horizon, a.lookback_days, a.replays, a.seed + sig["id"], a.window, a.target)
    out.update(status="ok", notes=notes,
               actual={"result": act.result, "ambiguous": act.ambiguous, "bars": act.bars, "exit_time": act.exit_time,
                       "complete": act.complete},
               history={"target": hist.target, "stop": hist.stop, "neither": hist.neither, "ambiguous": hist.ambiguous,
                        "n": hist.n, "skipped": hist.skipped, "first_day": hist.first_day, "last_day": hist.last_day,
                        "verdict": tm.verdict(hist.target, hist.stop, p_be),
                        "win_share_ci95": tm.wilson(hist.target, hist.target + hist.stop)},
               simulation=None if sim is None else {
                   "replays": sim.replays, "seed": sim.seed, "p_target": sim.p("target"), "p_stop": sim.p("stop"),
                   "p_neither": sim.p("neither"), "p_ambiguous": sim.p("ambiguous"), "pool_bars": sim.pool_bars,
                   "steps": sim.steps, "skipped_steps": sim.skipped_steps,
                   "verdict": tm.verdict(sim.target, sim.stop, p_be)})
    return out


def aggregate(results):
    ok = [r for r in results if r.get("status") == "ok"]
    act = {"target": 0, "stop": 0, "neither": 0, "ambiguous": 0, "incomplete": 0}
    hist = {"target": 0, "stop": 0, "neither": 0, "ambiguous": 0}
    sims, bes = [], []
    for r in ok:
        a = r["actual"]
        if a["result"] in act and (a["complete"] or a["result"] != "neither"):
            act[a["result"]] += 1
            act["ambiguous"] += int(a["ambiguous"])
        else:
            act["incomplete"] += 1
        for k in hist:
            hist[k] += r["history"][k]
        if r["simulation"]:
            sims.append(r["simulation"])
        if r["break_even"] is not None:
            bes.append(r["break_even"])
    mean_be = sum(bes) / len(bes) if bes else None
    hist_sided = hist["target"] + hist["stop"]
    return {
        "signals_total": len(results), "signals_analysed": len(ok),
        "signals_insufficient": len(results) - len(ok),
        "actual": act, "history": dict(hist, win_share=(hist["target"] / hist_sided) if hist_sided else None,
                                       win_share_ci95=tm.wilson(hist["target"], hist_sided)),
        "simulation_mean": {k: (sum(s[k] for s in sims) / len(sims)) if sims else None
                            for k in ("p_target", "p_stop", "p_neither")},
        "mean_break_even": mean_be,
    }


def _pct(x, d=1):
    return "N/A" if x is None else f"{100 * x:.{d}f}%"


def _ci(ci):
    return "N/A" if not ci else f"{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%"


def to_markdown(meta, data_info, results, agg):
    L = ["# Time Machine odds report (offline)", "",
         f"Generated {meta['generated_at']} UTC · code {meta['code_version']} · read-only · information only", "",
         "## Run", "",
         f"- Signals: source `{meta['source']}`, {agg['signals_total']} found, {agg['signals_analysed']} analysed, "
         f"{agg['signals_insufficient']} with insufficient data",
         f"- Signal dates: {meta['signal_first'] or 'N/A'} to {meta['signal_last'] or 'N/A'}",
         f"- Target: {meta['target'].upper()} vs the initial SL · horizon {meta['horizon_hours']} h (clock time) · "
         f"lookback {meta['lookback_days']} days",
         f"- Simulation: {meta['replays']:,} paths per signal, seed {meta['seed']} + signal id, "
         f"time-of-day window ±{meta['window']} bars",
         "- Same-bar SL and target: counted as the STOP (conservative), also listed as ambiguous", "",
         "## Data (cached M5, read-only)", "", "| Symbol | File | Bars | From (UTC) | To (UTC) | SHA-256 |",
         "|---|---|---|---|---|---|"]
    for sym, d in sorted(data_info.items()):
        L.append(f"| {sym} | {Path(d['file']).name} | {d.get('bars', 'missing')} | {d.get('first') or 'N/A'} | "
                 f"{d.get('last') or 'N/A'} | {(d.get('sha256') or 'N/A')[:16]} |")
    a = agg["actual"]
    h = agg["history"]
    s = agg["simulation_mean"]
    L += ["", "## Summary", "",
          f"- **Actual paths** ({agg['signals_analysed']} signals): target first {a['target']}, stop first "
          f"{a['stop']}, neither {a['neither']}, incomplete {a['incomplete']}; ambiguous bars {a['ambiguous']}",
          f"- **History replay** (all signals pooled, {h['target'] + h['stop'] + h['neither']} replays): target "
          f"{h['target']}, stop {h['stop']}, neither {h['neither']} (ambiguous {h['ambiguous']}) · win share "
          f"{_pct(h['win_share'])} (95% CI {_ci(h['win_share_ci95'])})",
          f"- **Simulation** (mean over signals): target first {_pct(s['p_target'])}, stop first "
          f"{_pct(s['p_stop'])}, neither {_pct(s['p_neither'])}",
          f"- **Mean break-even win share**: {_pct(agg['mean_break_even'])}", "",
          "## Per signal", "",
          "| # | Symbol | Dir | Created (UTC) | Entry | SL | Target | k (R) | Break-even | SL in ATR | Bot result "
          "| Actual path | History T/S/N (win share, verdict) | Simulation T/S/N (verdict) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        k_txt = "N/A" if r["target_r"] is None else "%.2f" % r["target_r"]
        atr_txt = "N/A" if r["sl_distance_atr"] is None else "%.2f" % r["sl_distance_atr"]
        base = (f"| {r['id']} | {r['symbol']} | {r['direction']} | {r['created_at'][:16]} | {r['entry']} | "
                f"{r['stop_loss']} | {r['target']} | {k_txt} | {_pct(r['break_even'])} | {atr_txt} "
                f"| {r['bot_result']} |")
        if r.get("status") != "ok":
            L.append(base + f" {r['status']} | | |")
            continue
        act = r["actual"]
        hv, hh = r["history"]["verdict"], r["history"]
        sm = r["simulation"]
        act_txt = act["result"] + (" (ambiguous bar)" if act["ambiguous"] else "") + ("" if act["complete"] else
                                                                                        " (data incomplete)")
        hist_txt = (f"{hh['target']}/{hh['stop']}/{hh['neither']} of {hh['n']} "
                    f"({_pct(hv['win_share'])}, {hv['label']})")
        sim_txt = "N/A" if not sm else (f"{_pct(sm['p_target'])}/{_pct(sm['p_stop'])}/{_pct(sm['p_neither'])} "
                                        f"(seed {sm['seed']}, {sm['verdict']['label']})")
        L.append(base + f" {act_txt} | {hist_txt} | {sim_txt} |")
    notes = [(r["id"], n) for r in results for n in r.get("notes", [])]
    if notes:
        L += ["", "### Data notes", ""] + [f"- #{i}: {n}" for i, n in notes]
    L += ["", "## Limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline Time Machine odds report (read-only, information only)")
    ap.add_argument("--bot", default=str(ROOT), help="bot folder (default: this repository)")
    ap.add_argument("--db", default=None, help="bot.db path (default: <bot>/data/bot.db)")
    ap.add_argument("--history-dir", required=True, help="folder with <SYMBOL>_M5.csv (read-only)")
    ap.add_argument("--target", choices=("tp1", "tp2", "tp3"), default="tp1")
    ap.add_argument("--horizon-hours", type=float, default=24.0)
    ap.add_argument("--lookback-days", type=int, default=60)
    ap.add_argument("--replays", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--signal-id", type=int, action="append", help="only these signals (repeatable)")
    ap.add_argument("--source", default="live")
    ap.add_argument("--out", default=None, help="new report folder (default: <bot>/logs/reports/timemachine/<UTC>)")
    a = ap.parse_args(argv)
    bot = Path(a.bot)
    now = datetime.now(timezone.utc)
    out = Path(a.out) if a.out else bot / "logs" / "reports" / "timemachine" / now.strftime("%Y%m%dT%H%M%SZ")
    if out.exists():
        print(f"Refusing to overwrite: {out} already exists")
        return 2
    signals = read_signals(Path(a.db) if a.db else bot / "data" / "bot.db", a.source, a.signal_id)
    if not signals:
        print("No signals to analyse.")
        return 2
    hist_dir = Path(a.history_dir)
    data, data_info = {}, {}
    for sym in sorted({s["symbol"] for s in signals}):
        data[sym], data_info[sym] = read_history(hist_dir, sym)
    results = [analyse(s, data[s["symbol"]], a) for s in signals]
    agg = aggregate(results)
    meta = {"generated_at": now.strftime("%Y-%m-%d %H:%M:%S"), "code_version": _code_version(), "source": a.source,
            "target": a.target, "horizon_hours": a.horizon_hours, "lookback_days": a.lookback_days,
            "replays": a.replays, "seed": a.seed, "window": a.window, "history_dir": str(hist_dir),
            "signal_first": min(s["created_at"] for s in signals).isoformat(),
            "signal_last": max(s["created_at"] for s in signals).isoformat()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)                                  # never overwrite an existing report
    (out / "report.md").write_text(to_markdown(meta, data_info, results, agg), encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"meta": meta, "data": data_info, "aggregate": agg,
                                                 "signals": results, "limitations": LIMITATIONS},
                                                indent=1, default=str), encoding="utf-8")
    print(f"Report: {out / 'report.md'}  ({agg['signals_analysed']} of {agg['signals_total']} signals analysed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
