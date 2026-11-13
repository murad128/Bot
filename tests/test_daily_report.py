"""
Daily report: TP milestones and realised R by EVENT time.

Regression for: a TP1 hit on a still-open signal was missing from the daily
report (TP1 Hit 0, Net +0.00R), because only signals CLOSED today were counted.
"""
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from config import load_config
from core.domain import ComponentScore, Direction, RiskPlan, SignalCandidate
from core.orchestrator import SignalBot
from core.performance_analyzer import day_activity, event_contributions, open_trades
from core.signal_tracker import SignalTracker
from database.database import Database
from database.models import SignalRecord

FR = [0.5, 0.3, 0.2]
U = lambda s: pd.Timestamp(s, tz="UTC").to_pydatetime()


def _report(db, cfg, now):
    bot = SignalBot.__new__(SignalBot)     # only the report builder is needed (no MT5, no Telegram)
    bot.db, bot.cfg = db, cfg
    return bot.build_daily_report(now).replace("<b>", "").replace("</b>", "")


def _line(report, prefix):
    return next(l for l in report.splitlines() if l.startswith(prefix))


def _xau_sell(db, cfg, created):
    """The reported XAUUSD SELL: entry 4000.56, TP1 3991.42 (1R = 9.14)."""
    risk = 9.14
    plan = RiskPlan(4000.56, 4000.56, 4000.90, round(4000.56 + risk, 2), [3991.42, 3982.28, 3973.14],
                    risk, [1, 2, 3], 2.0, None)
    c = SignalCandidate("XAUUSD", Direction.SELL, "OB_RETEST", created, "M15", "H1", "Bearish", "London", 82,
                        "Strong", plan, [ComponentScore("smc", 0.8, 20)], ["x"], [], 5.0, 0.25,
                        {"session_bucket": "London"})
    return SignalTracker(db, cfg.risk).register(c)


def _bar(t, o, h, l, c):
    return pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c]},
                        index=pd.DatetimeIndex([pd.Timestamp(t).floor("min")]))


@pytest.fixture
def env(cfg):
    cfg.risk.tp_close_fractions = list(FR)
    return Database(cfg.database_url), cfg


# ----------------------------------------------------------------------------- the reported bug
def test_tp1_on_open_trade_appears_in_report_of_event_day(env):
    db, cfg = env
    rec = _xau_sell(db, cfg, U("2026-11-11 01:30"))
    ev = SignalTracker(db, cfg.risk).update_with_bars(rec, _bar(U("2026-11-11 03:30"), 3995, 3996, 3991, 3992), 0.25)
    assert [e.kind for e in ev] == ["TP1_HIT"] and ev[0].r == pytest.approx(1.0, abs=1e-3)
    r = _report(db, cfg, U("2026-11-11 21:15"))
    assert "Day: 2026-11-11 00:00–24:00 UTC" in r
    assert _line(r, "Signals issued today:").endswith(" 1")
    assert _line(r, "TP1 Hit:") == "TP1 Hit: 1"
    assert "Realised R today (incl. partial exits): +0.50R" in r
    assert _line(r, "Closed:").startswith("Closed: 0")
    assert "Total R of trades closed today: +0.00R" in r
    assert _line(r, "Open trades:") == "Open trades: 1"
    assert "XAUUSD SELL – TP1 hit, stop at breakeven, realised so far +0.50R" in r
    assert "Cumulative realised R on open trades: +0.50R" in r


def test_tp1_one_day_breakeven_close_next_day(env):
    db, cfg = env
    tr = SignalTracker(db, cfg.risk)
    rec = _xau_sell(db, cfg, U("2026-11-10 14:00"))
    tr.update_with_bars(rec, _bar(U("2026-11-10 16:00"), 3995, 3996, 3991, 3992), 0.25)        # TP1 day 1
    rec = db.get_signal(rec.id)
    ev = tr.update_with_bars(rec, _bar(U("2026-11-11 09:00"), 3998, 4001.0, 3997, 4000), 0.25)  # BE day 2
    assert [e.kind for e in ev] == ["BE_HIT"]
    d1 = _report(db, cfg, U("2026-11-10 21:15"))
    d2 = _report(db, cfg, U("2026-11-11 21:15"))
    assert _line(d1, "TP1 Hit:") == "TP1 Hit: 1" and "Realised R today (incl. partial exits): +0.50R" in d1
    assert _line(d2, "TP1 Hit:") == "TP1 Hit: 0"                            # not counted twice
    assert "Stopped at breakeven / trailed stop: 1 / 0" in d2
    assert "Realised R today (incl. partial exits): +0.00R" in d2           # BE exit adds nothing
    assert _line(d2, "Closed:").startswith("Closed: 1")
    assert "Wins / Losses / BE: 1 / 0 / 0" in d2                              # a TP1+BE trade is a win
    assert "Total R of trades closed today: +0.50R" in d2                    # whole trade, incl. day-1 partial
    assert _line(d2, "Open trades:") == "Open trades: 0"


def test_partial_exit_arithmetic_tp1_tp2_then_trailed_stop(env):
    db, cfg = env
    tr = SignalTracker(db, cfg.risk)
    # within the 24h maximum holding time, crossing UTC midnight
    rec = _xau_sell(db, cfg, U("2026-11-10 18:00"))
    tr.update_with_bars(rec, _bar(U("2026-11-10 20:00"), 3995, 3996, 3991, 3992), 0.25)          # TP1 +1R (day 1)
    rec = db.get_signal(rec.id)
    tr.update_with_bars(rec, _bar(U("2026-11-11 06:00"), 3985, 3986, 3982.0, 3983), 0.25)       # TP2 +2R (day 2)
    rec = db.get_signal(rec.id)
    tr.update_with_bars(rec, _bar(U("2026-11-11 07:00"), 3988, 3992.0, 3987, 3991), 0.25)       # trailed SL @ TP1
    final = db.get_signal(rec.id)
    assert final.outcome == "WIN" and final.status == "STOPPED"
    # 0.5*1 + 0.3*2 + 0.2*1 = 1.3R, computed independently by the tracker
    assert final.realized_r == pytest.approx(1.3, abs=1e-3)
    contrib = event_contributions(final, FR)
    assert [c.kind for c in contrib] == ["TP1_HIT", "TP2_HIT", "SL_HIT"]
    assert [round(c.realized_r, 3) for c in contrib] == [0.5, 0.6, 0.2]
    assert sum(c.realized_r for c in contrib) == pytest.approx(final.realized_r)
    d1 = day_activity(db.active_between(U("2026-11-10"), U("2026-11-11")), U("2026-11-10"), U("2026-11-11"), FR)
    d2 = day_activity(db.active_between(U("2026-11-11"), U("2026-11-12")), U("2026-11-11"), U("2026-11-12"), FR)
    assert d1.realized_r == pytest.approx(0.5) and d1.tp1 == 1 and d1.tp2 == 0
    assert d2.realized_r == pytest.approx(0.8) and d2.tp2 == 1 and d2.trailed_stops == 1 and d2.sl_before_tp1 == 0
    assert d1.realized_r + d2.realized_r == pytest.approx(final.realized_r)


def _manual(db, sid, created, events, outcome="OPEN", status="ACTIVE", realized=0.0, closed=None, tp_hits=0,
            entry=100.0, sl=99.0):
    rec = SignalRecord(dedup_key=f"t:{sid}", symbol="XAUUSD", direction="BUY", setup_type="X",
                       entry_timeframe="M15", confirm_timeframe="H1", session="London", score=80, grade="Strong",
                       entry=entry, entry_low=entry, entry_high=entry, stop_loss=99.0, current_sl=sl, tp1=101.0,
                       tp2=102.0, tp3=103.0, risk=1.0, rr=2.0, status=status, outcome=outcome, tp_hits=tp_hits,
                       realized_r=realized, created_at=created, closed_at=closed, events=events)
    return db.add_signal(rec)


# ----------------------------------------------------------------------------- UTC midnight boundaries
@pytest.mark.parametrize("event_time,day1,day2", [
    ("2026-11-10 23:59:59", 1, 0),
    ("2026-11-11 00:00:00", 0, 1),      # midnight belongs to the NEW day ([start, end))
    ("2026-11-11 00:00:01", 0, 1),
])
def test_utc_midnight_boundary(env, event_time, day1, day2):
    db, cfg = env
    _manual(db, 1, U("2026-11-10 20:00"), [{"kind": "TP1_HIT", "time": U(event_time).isoformat(), "price": 101.0, "r": 1.0}],
            status="TP1_HIT", realized=0.5, tp_hits=1, sl=100.0)
    r1 = _report(db, cfg, U("2026-11-10 21:15"))
    r2 = _report(db, cfg, U("2026-11-11 21:15"))
    assert _line(r1, "TP1 Hit:") == f"TP1 Hit: {day1}"
    assert _line(r2, "TP1 Hit:") == f"TP1 Hit: {day2}"
    assert ("+0.50R" in _line(r1, "Realised R today")) == bool(day1)
    assert ("+0.50R" in _line(r2, "Realised R today")) == bool(day2)


def test_signal_created_late_utc_counts_on_its_utc_day(env):
    db, cfg = env
    _xau_sell(db, cfg, U("2026-11-10 23:59"))
    assert _line(_report(db, cfg, U("2026-11-10 23:59:30")), "Signals issued today:").endswith(" 1")
    assert _line(_report(db, cfg, U("2026-11-11 00:00:30")), "Signals issued today:").endswith(" 0")


# ----------------------------------------------------------------------------- duplicates
def test_duplicate_events_are_counted_once(env):
    db, cfg = env
    tp1 = {"kind": "TP1_HIT", "time": U("2026-11-11 10:00").isoformat(), "price": 101.0, "r": 1.0}
    sl = {"kind": "BE_HIT", "time": U("2026-11-11 12:00").isoformat(), "price": 100.0, "r": 0.0}
    sl2 = {"kind": "SL_HIT", "time": U("2026-11-11 12:01").isoformat(), "price": 99.0, "r": -1.0}
    _manual(db, 1, U("2026-11-11 09:00"), [tp1, dict(tp1), sl, sl2], outcome="WIN", status="STOPPED",
            realized=0.5, closed=U("2026-11-11 12:00"), tp_hits=1, sl=100.0)
    act = day_activity(db.active_between(U("2026-11-11"), U("2026-11-12")) * 2,      # same record twice
                       U("2026-11-11"), U("2026-11-12"), FR)
    assert act.tp1 == 1 and act.breakeven_stops == 1 and act.sl_before_tp1 == 0 and act.trailed_stops == 0
    assert act.realized_r == pytest.approx(0.5)


# ----------------------------------------------------------------------------- losses, cancels, costs
def test_full_stop_loss_and_cancelled_trade(env):
    db, cfg = env
    _manual(db, 1, U("2026-11-11 08:00"), [{"kind": "SL_HIT", "time": U("2026-11-11 09:00").isoformat(),
                                            "price": 99.0, "r": -1.0}],
            outcome="LOSS", status="STOPPED", realized=-1.0, closed=U("2026-11-11 09:00"))
    _manual(db, 2, U("2026-11-11 10:00"), [{"kind": "CANCELLED", "time": U("2026-11-11 11:00").isoformat(),
                                            "price": 99.8, "r": -0.2}],
            outcome="CANCELLED", status="CANCELLED", realized=-0.2, closed=U("2026-11-11 11:00"))
    r = _report(db, cfg, U("2026-11-11 21:15"))
    assert "Stop Loss (before TP1): 1" in r and "Cancelled: 1" in r
    assert "Realised R today (incl. partial exits): -1.20R" in r
    assert "Wins / Losses / BE: 0 / 1 / 0" in r
    assert "Win Rate: 0.0% (based on 1 resolved trades)" in r              # cancelled excluded from win rate
    assert "Total R of trades closed today: -1.20R" in r


def test_final_event_reconciles_to_stored_realized_r(env):
    """Costs are only in the stored realized_r; the final event absorbs them so totals never drift."""
    db, cfg = env
    _manual(db, 1, U("2026-11-11 08:00"),
            [{"kind": "TP1_HIT", "time": U("2026-11-11 09:00").isoformat(), "price": 101.0, "r": 1.0},
             {"kind": "BE_HIT", "time": U("2026-11-11 10:00").isoformat(), "price": 100.0, "r": 0.0}],
            outcome="WIN", status="STOPPED", realized=0.43, closed=U("2026-11-11 10:00"), tp_hits=1, sl=100.0)
    c = event_contributions(db.get_signal(1), FR)
    assert c[0].realized_r == pytest.approx(0.5) and c[1].realized_r == pytest.approx(-0.07)
    assert sum(x.realized_r for x in c) == pytest.approx(0.43)


def test_open_trade_without_tp_has_zero_realised(env):
    db, cfg = env
    _manual(db, 1, U("2026-11-11 08:00"), [])
    r = _report(db, cfg, U("2026-11-11 21:15"))
    assert "– no TP yet, realised so far +0.00R" in r and "Realised R today (incl. partial exits): +0.00R" in r
    assert open_trades(db.open_signals())[0].stop_at_entry is False


def test_naive_and_z_timestamps_are_parsed_as_utc(env):
    db, cfg = env
    _manual(db, 1, U("2026-11-11 08:00"), [{"kind": "TP1_HIT", "time": "2026-11-11T09:00:00", "price": 101, "r": 1.0}],
            status="TP1_HIT", realized=0.5, tp_hits=1, sl=100.0)
    _manual(db, 2, U("2026-11-11 08:00"), [{"kind": "TP1_HIT", "time": "2026-11-11T09:00:00Z", "price": 101, "r": 1.0}],
            status="TP1_HIT", realized=0.5, tp_hits=1, sl=100.0)
    act = day_activity(db.active_between(U("2026-11-11"), U("2026-11-12")), U("2026-11-11"), U("2026-11-12"), FR)
    assert act.tp1 == 2 and act.realized_r == pytest.approx(1.0)


# ----------------------------------------------------------------------------- query window
def test_active_between_window(env):
    db, cfg = env
    _manual(db, 1, U("2026-11-09 10:00"), [], outcome="LOSS", status="STOPPED", closed=U("2026-11-09 12:00"))  # before
    _manual(db, 2, U("2026-11-12 01:00"), [])                                                                   # after
    _manual(db, 3, U("2026-11-08 10:00"), [])                                                                   # still open
    _manual(db, 4, U("2026-11-10 10:00"), [], outcome="WIN", status="TP3_HIT", closed=U("2026-11-11 05:00"))   # closed in day
    _manual(db, 5, U("2026-11-10 10:00"), [], outcome="WIN", status="TP3_HIT", closed=U("2026-11-11 00:00"))   # closed at start
    _manual(db, 6, U("2026-11-10 10:00"), [], outcome="WIN", status="TP3_HIT", closed=U("2026-11-10 23:59"))   # closed before
    ids = sorted(r.id for r in db.active_between(U("2026-11-11"), U("2026-11-12")))
    assert ids == [3, 4, 5]


# ----------------------------------------------------------------------------- read-only guarantee
def test_building_the_report_does_not_modify_the_database(env):
    db, cfg = env
    rec = _xau_sell(db, cfg, U("2026-11-11 01:30"))
    SignalTracker(db, cfg.risk).update_with_bars(rec, _bar(U("2026-11-11 03:30"), 3995, 3996, 3991, 3992), 0.25)
    path = cfg.database_url.replace("sqlite:///", "")

    def snapshot():
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            rows = con.execute("select * from signals order by id").fetchall()
            other = [con.execute(f"select count(*) from {t}").fetchone()[0]
                     for t in ("notifications", "outbox", "recommendations", "kv_state")]
        return hashlib.sha256(repr((rows, other)).encode()).hexdigest()

    before = snapshot()
    for _ in range(3):
        _report(db, cfg, U("2026-11-11 21:15"))
    assert snapshot() == before


def test_report_text_is_utf8_safe():
    """Windows: the report contains en dashes and emoji; it must encode as UTF-8 (Telegram) without errors."""
    from core.performance_analyzer import compute_stats
    from integrations.telegram_messages import daily_report_message
    s = compute_stats([])
    text = daily_report_message("2026-11-11", s, s, {}, {}, None, None, 0)
    text.encode("utf-8")
    assert "00:00–24:00 UTC" in text
