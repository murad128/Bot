"""Offline Time Machine odds: price rules, distances, no look-ahead, determinism (pure calculations)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtesting import timemachine_odds as tm

U = lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc)  # noqa: E731


def bars_from(rows):
    """rows: (open time 'YYYY-MM-DD HH:MM', open, high, low, close[, spread])."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz="UTC") for r in rows])
    return tm.prepare_bars(pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                                         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
                                         "spread_price": [r[5] if len(r) > 5 else np.nan for r in rows]}, index=idx))


def flat_bars(start, n, price=100.0, spread=0.0):
    t = pd.Timestamp(start, tz="UTC")
    return [((t + i * tm.BAR).strftime("%Y-%m-%d %H:%M"), price, price, price, price, spread) for i in range(n)]


def sig(direction="BUY", entry=100.0, sl=99.0, tp1=101.5, created="2024-03-06 10:00:05", spread=0.0, sid=1):
    return {"id": sid, "direction": direction, "entry": entry, "stop_loss": sl, "tp1": tp1, "tp2": tp1, "tp3": tp1,
            "spread": spread, "created_at": U(created.replace(" ", "T"))}


# ----------------------------------------------------------------------------- basics
def test_levels_break_even_wilson_and_verdict():
    assert tm.levels("BUY", 100, 1, 2) == (99, 102) and tm.levels("SELL", 100, 1, 2) == (101, 98)
    assert tm.break_even(1.0) == pytest.approx(0.5) and tm.break_even(2.0) == pytest.approx(1 / 3)
    lo, hi = tm.wilson(5, 10)
    assert lo < 0.5 < hi and tm.wilson(0, 0) is None
    assert tm.verdict(52, 48, 0.5)["label"] == "close to break-even"
    assert tm.verdict(60, 40, 0.5)["label"] == "leans toward the target"
    assert tm.verdict(20, 80, 0.5)["label"] == "strongly toward the stop"
    assert tm.verdict(0, 0, 0.5)["win_share"] is None


def test_entry_bar_is_the_last_closed_bar():
    b = bars_from(flat_bars("2024-03-06 09:00", 24))
    i = tm.entry_index(b, U("2024-03-06T10:00:05"))
    assert b.index[i] == pd.Timestamp("2024-03-06 09:55", tz="UTC")       # closed 10:00, evaluated at 10:00:05


# ----------------------------------------------------------------------------- price rules
def test_same_bar_stop_and_target_counts_as_stop_and_ambiguous():
    rows = flat_bars("2024-03-06 09:00", 12) + [("2024-03-06 10:00", 100, 102, 98, 100, 0.0)]
    b = bars_from(rows)
    t = tm.first_touch(b, len(b) - 1, "BUY", 99.0, 101.5, b.index[-1] + tm.BAR, 0.0)
    assert t.result == "stop" and t.ambiguous


def test_sell_is_stopped_by_the_ask_not_the_bid():
    # bid high 100.90 stays under the 101 stop, but ask high = 100.90 + 0.20 spread crosses it
    rows = flat_bars("2024-03-06 09:00", 12) + [("2024-03-06 10:00", 100, 100.9, 99.9, 100, 0.20)]
    b = bars_from(rows)
    t = tm.first_touch(b, len(b) - 1, "SELL", 101.0, 98.5, b.index[-1] + tm.BAR, 0.0)
    assert t.result == "stop" and not t.ambiguous
    t2 = tm.first_touch(b, len(b) - 1, "SELL", 101.5, 98.5, b.index[-1] + tm.BAR, 0.0)
    assert t2.result == "neither"


def test_actual_path_uses_the_signals_own_levels_and_horizon():
    rows = flat_bars("2024-03-06 09:00", 13) + [("2024-03-06 10:05", 100, 101.6, 99.5, 101, 0.0)]
    s = sig()
    assert tm.actual_path(bars_from(rows), s, timedelta(hours=24)).result == "target"
    # a horizon that ends before the 10:05 bar closes sees nothing
    assert tm.actual_path(bars_from(rows), s, timedelta(minutes=5)).result == "neither"


# ----------------------------------------------------------------------------- history replay: distances
def _days_with_level(levels, rise=2.0):
    """One day per level: flat at `level`, then the bar after 10:00 rises by `rise` (open time UTC)."""
    rows = []
    for d, level in enumerate(levels):
        day = pd.Timestamp("2024-03-01", tz="UTC") + pd.Timedelta(days=d)
        for i in range(0, 288):
            t = day + i * tm.BAR
            p = level
            if t.hour == 10 and t.minute == 0:
                rows.append((t.strftime("%Y-%m-%d %H:%M"), p, p + rise, p - 0.1, p + rise * 0.5, 0.0))
            else:
                rows.append((t.strftime("%Y-%m-%d %H:%M"), p, p, p, p, 0.0))
    return bars_from(rows)


def test_history_replay_applies_distances_to_each_days_entry_not_absolute_levels():
    levels = [100 + 10 * d for d in range(6)]                     # each day at a very different price
    b = _days_with_level(levels)
    s = sig(entry=150.0, sl=149.0, tp1=151.5, created="2024-03-06 09:55:05")    # signal on the last day, before the rise
    c = tm.history_replay(b, s, timedelta(hours=1), lookback_days=5)
    # absolute levels (SL 149) would stop out every lower-priced day at once; distances hit the target every day
    assert c.n == 5 and c.target == 5 and c.stop == 0
    assert c.first_day == "2024-03-01" and c.last_day == "2024-03-05"


def test_history_replay_never_uses_bars_from_after_the_signal():
    b = _days_with_level([100] * 6)
    s = sig(entry=100.0, sl=99.0, tp1=101.5, created="2024-03-06 09:55:05")
    full = tm.history_replay(b, s, timedelta(hours=24), lookback_days=5)
    cut = b[b.index + tm.BAR <= pd.Timestamp("2024-03-06 09:55", tz="UTC") + tm.BAR]
    trimmed = tm.history_replay(cut, s, timedelta(hours=24), lookback_days=5)
    assert (full.target, full.stop, full.neither, full.skipped) == \
           (trimmed.target, trimmed.stop, trimmed.neither, trimmed.skipped)
    assert full.skipped.get("window would overlap the signal", 0) == 0


def test_history_replay_counts_skipped_days():
    b = _days_with_level([100] * 3)
    s = sig(entry=100, sl=99, tp1=101.5, created="2024-03-03 09:55:05")
    c = tm.history_replay(b, s, timedelta(hours=1), lookback_days=5)       # only 2 earlier days exist
    assert c.n == 2 and sum(c.skipped.values()) == 3


# ----------------------------------------------------------------------------- simulation
def _random_walk(days=12, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04", periods=288 * days, freq="5min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, len(idx))))
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.03, len(idx)))
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick,
                       "close": close, "spread_price": 0.02}, index=idx)
    return tm.prepare_bars(df[df.index.day_of_week < 5])                  # weekdays only, like FX data


def test_simulation_is_reproducible_with_the_seed():
    b = _random_walk()
    s = sig(entry=float(b["close"].iloc[-300]), created=(b.index[-300] + tm.BAR).strftime("%Y-%m-%d %H:%M:05"))
    s["stop_loss"], s["tp1"] = s["entry"] - 0.5, s["entry"] + 0.75
    a = tm.simulate(b, s, timedelta(hours=24), 7, 2000, seed=20)
    again = tm.simulate(b, s, timedelta(hours=24), 7, 2000, seed=20)
    other = tm.simulate(b, s, timedelta(hours=24), 7, 2000, seed=21)
    assert (a.target, a.stop, a.neither) == (again.target, again.stop, again.neither)
    assert a.target + a.stop + a.neither == 2000 and a.pool_bars > 0
    assert (a.target, a.stop) != (other.target, other.stop)


def test_simulation_has_no_lookahead():
    b = _random_walk()
    t_entry = b.index[-300]
    s = sig(entry=float(b["close"].iloc[-300]), created=(t_entry + tm.BAR).strftime("%Y-%m-%d %H:%M:05"))
    s["stop_loss"], s["tp1"] = s["entry"] - 0.5, s["entry"] + 0.75
    full = tm.simulate(b, s, timedelta(hours=24), 7, 1000, seed=5)
    past = tm.simulate(b[b.index <= t_entry], s, timedelta(hours=24), 7, 1000, seed=5)
    assert (full.target, full.stop, full.neither, full.pool_bars) == (past.target, past.stop, past.neither, past.pool_bars)


def test_pool_excludes_moves_across_gaps():
    rows = flat_bars("2024-03-06 09:00", 3, price=100) + flat_bars("2024-03-06 12:00", 3, price=200)
    pool = tm.build_pool(bars_from(rows), pd.Timestamp("2024-03-06", tz="UTC"), pd.Timestamp("2024-03-07", tz="UTC"))
    ratios = np.concatenate(list(pool.values()))
    assert len(ratios) == 4 and np.allclose(ratios[:, 3], 1.0)             # the 100 -> 200 gap never becomes a move


def test_simulation_skips_the_weekend_when_the_market_is_closed():
    b = _random_walk(days=14)
    fri = b.index[(b.index.day_of_week == 4) & (b.index.hour == 20) & (b.index.minute == 0)][-1]   # Fri 20:00 UTC
    s = sig(entry=float(b.loc[fri, "close"]), created=(fri + tm.BAR).strftime("%Y-%m-%d %H:%M:05"))
    s["stop_loss"], s["tp1"] = s["entry"] - 50, s["entry"] + 50            # never hit: count the steps
    r = tm.simulate(b, s, timedelta(hours=24), 7, 100, seed=1)
    assert r.neither == 100 and r.skipped_steps > 200                      # most of a Fri 20:05 -> Sat 20:05 day is closed
