"""Telegram message templates (HTML parse mode)."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.domain import SignalCandidate, SymbolSpec
from core.performance_analyzer import Stats, fmt_pct, fmt_pf

DISCLAIMER = ("<i>Signal only - not financial advice. No result is guaranteed; "
              "the score measures confluence, not probability of success.</i>")


def esc(x) -> str:
    return html.escape(str(x), quote=False)


def _icon(symbol: str, direction: str) -> str:
    if symbol.startswith("XAU"):
        return "🟡"
    return "🟢" if direction == "BUY" else "🔴"


def _p(price: float, digits: int) -> str:
    return f"{price:.{digits}f}"


def signal_message(c: SignalCandidate, spec: SymbolSpec, signal_id: Optional[int] = None) -> str:
    p, dg = c.plan, spec.digits
    tps = "\n".join(f"TP{i + 1}: {_p(t, dg)}  ({p.rr_targets[i]:.1f}R)" for i, t in enumerate(p.take_profits))
    reasons = "\n".join(f"• {esc(r)}" for r in c.reasons[:10])
    warn = ""
    if c.warnings:
        warn = "\n\n⚠️ <b>Cautions:</b>\n" + "\n".join(f"• {esc(w)}" for w in c.warnings[:6])
    sl_pips = spec.to_pips(p.risk)
    head = f"{_icon(c.symbol, c.direction.value)} <b>{esc(c.symbol)} – {c.direction.value} SIGNAL</b>"
    if signal_id is not None:
        head += f"  <code>#{signal_id}</code>"
    return (
        f"{head}\n\n"
        f"<b>Direction:</b> {c.direction.value}\n"
        f"<b>Entry:</b> {_p(p.entry_low, dg)} – {_p(p.entry_high, dg)} (ref {_p(p.entry, dg)})\n"
        f"<b>Stop Loss:</b> {_p(p.stop_loss, dg)}  ({sl_pips:.1f} pips)\n\n"
        f"<b>Take Profit:</b>\n{tps}\n\n"
        f"<b>Risk/Reward:</b> 1:{p.primary_rr:.1f}\n"
        f"<b>Timeframe:</b> {c.entry_timeframe} / {c.confirm_timeframe}\n"
        f"<b>Market Trend:</b> {esc(c.trend_label)}\n"
        f"<b>Setup:</b> {esc(c.setup_type.replace('_', ' ').title())}\n"
        f"<b>Signal Score:</b> {c.score:.0f}/100 ({esc(c.grade)})\n\n"
        f"<b>Technical Reasons:</b>\n{reasons}"
        f"{warn}\n\n"
        f"<b>Session:</b> {esc(c.session)}\n"
        f"<b>Signal Status:</b> ACTIVE\n"
        f"<b>Time:</b> {c.created_at:%Y-%m-%d %H:%M} UTC\n\n"
        f"{DISCLAIMER}"
    )


_EVENT_TEXT = {
    "TP1_HIT": ("🎯", "TP1 HIT", "Stop moved to breakeven" ),
    "TP2_HIT": ("🎯", "TP2 HIT", "Stop moved to TP1"),
    "TP3_HIT": ("🏆", "TP3 HIT – target complete", ""),
    "SL_HIT": ("🛑", "STOP LOSS HIT", ""),
    "BE_HIT": ("⚖️", "STOPPED AT BREAKEVEN", ""),
    "EXPIRED": ("⌛", "SIGNAL EXPIRED (time stop)", "Closed at market"),
    "CANCELLED": ("❌", "SIGNAL CANCELLED", "Closed at market"),
}


def event_message(rec, kind: str, price: float, r: float, digits: int, note: str = "") -> str:
    icon, title, extra = _EVENT_TEXT.get(kind, ("ℹ️", kind, ""))
    lines = [f"{icon} <b>{esc(rec.symbol)} {rec.direction} – {title}</b>  <code>#{rec.id}</code>",
             f"Price: {price:.{digits}f}  ({r:+.2f}R on this exit)",
             f"Entry: {rec.entry:.{digits}f}  SL: {rec.current_sl:.{digits}f}"]
    if extra and rec.outcome == "OPEN":
        lines.append(extra)
    if note:
        lines.append(esc(note))
    if rec.outcome != "OPEN":
        lines.append(f"<b>Final result:</b> {rec.outcome}  ({rec.realized_r:+.2f}R incl. partials)")
    return "\n".join(lines)


def news_alert_message(event, minutes: int, symbols: List[str]) -> str:
    details = []
    if event.forecast:
        details.append(f"Forecast: {esc(event.forecast)}")
    if event.previous:
        details.append(f"Previous: {esc(event.previous)}")
    return (
        f"📰 <b>Upcoming {esc(event.impact)}-impact event in {minutes} min</b>\n"
        f"{esc(event.currency)} – {esc(event.title)}\n"
        f"Time: {event.time:%Y-%m-%d %H:%M} UTC\n"
        + ("\n".join(details) + "\n" if details else "")
        + f"Affected: {esc(', '.join(symbols)) or '-'}\n"
        f"New signals are paused around the release."
    )


def headline_message(item) -> str:
    kw = ", ".join(item.risk_keywords)
    return (f"🌐 <b>Market headline</b> ({esc(item.source)}, {item.time:%H:%M} UTC)\n"
            f"{esc(item.headline)}\n"
            f"<i>Keywords: {esc(kw)} – informational, not a trade signal.</i>")


def _row(name: str, s: Stats) -> str:
    return f"{esc(name)}: {s.resolved} res, WR {fmt_pct(s.win_rate)}, {s.net_r:+.2f}R"


def daily_report_message(day: str, created: Stats, closed: Stats, by_symbol: Dict[str, Stats],
                         by_session: Dict[str, Stats], best_symbol: Optional[str], best_session: Optional[str],
                         open_count: int, all_time: Optional[Stats] = None) -> str:
    lines = [
        f"📊 <b>DAILY PERFORMANCE REPORT – {esc(day)}</b>",
        "",
        f"<b>Signals issued today:</b> {created.total}",
        f"<b>Signals closed today:</b> {closed.total - closed.open}",
        f"TP1 Hit: {closed.tp1}",
        f"TP2 Hit: {closed.tp2}",
        f"TP3 Hit: {closed.tp3}",
        f"Stop Loss (before TP1): {closed.sl}",
        f"Expired: {closed.expired}",
        f"Cancelled: {closed.cancelled}",
        f"Still open: {open_count}",
        "",
        f"<b>Wins / Losses / BE:</b> {closed.wins} / {closed.losses} / {closed.breakeven}",
        f"<b>Win Rate:</b> {fmt_pct(closed.win_rate)} (based on {closed.resolved} resolved trades)",
        f"<b>Net result:</b> {closed.net_r:+.2f}R",
        f"<b>Profit factor:</b> {fmt_pf(closed.profit_factor)}",
        f"<b>Max drawdown (day):</b> {closed.max_drawdown_r:.2f}R",
        "",
        f"<b>Best Instrument:</b> {esc(best_symbol or 'n/a')}",
        f"<b>Best Session:</b> {esc(best_session or 'n/a')}",
    ]
    if by_symbol:
        lines += ["", "<b>By instrument:</b>"] + [f"• {_row(k, v)}" for k, v in by_symbol.items()]
    if by_session:
        lines += ["", "<b>By session:</b>"] + [f"• {_row(k, v)}" for k, v in by_session.items()]
    if all_time is not None and all_time.resolved:
        lines += ["", f"<b>All-time:</b> {all_time.resolved} resolved, WR {fmt_pct(all_time.win_rate)}, "
                      f"{all_time.net_r:+.2f}R, PF {fmt_pf(all_time.profit_factor)}, "
                      f"max DD {all_time.max_drawdown_r:.2f}R"]
    lines += [
        "",
        "<i>How it is calculated: each signal is ONE trade with partial exits "
        "(TP1/TP2/TP3 are milestones, not separate wins). A signal is a win if its "
        "realised R incl. partials is &gt; 0. Win rate = wins ÷ resolved trades "
        "(cancelled and open signals excluded). R = multiples of the initial risk.</i>",
    ]
    return "\n".join(lines)


def system_message(level: str, text: str) -> str:
    icon = {"error": "🚨", "warning": "⚠️", "info": "ℹ️", "ok": "✅"}.get(level, "ℹ️")
    return f"{icon} <b>System {esc(level.upper())}</b>\n{esc(text)}\n<i>{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</i>"
