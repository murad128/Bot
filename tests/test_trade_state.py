from datetime import datetime, timedelta, timezone

import pytest

from core.domain import Direction, Outcome, SignalStatus
from core.trade_state import TradeState

T0 = datetime(2024, 3, 4, 9, 0, tzinfo=timezone.utc)


def buy(**kw):
    args = dict(direction=Direction.BUY, entry=100.0, stop_loss=99.0, take_profits=[101.0, 102.0, 103.0],
                close_fractions=[0.5, 0.3, 0.2], opened_at=T0, max_hold=timedelta(hours=24))
    args.update(kw)
    return TradeState(**args)


def t(m):
    return T0 + timedelta(minutes=m)


def test_full_target_realised_r():
    s = buy()
    s.update_bar(101.2, 100.1, 101.0, t(1))
    assert s.status is SignalStatus.TP1_HIT and s.current_sl == 100.0
    s.update_bar(102.1, 101.05, 102.0, t(2))
    assert s.status is SignalStatus.TP2_HIT and s.current_sl == 101.0
    s.update_bar(103.5, 102.0, 103.2, t(3))
    assert s.status is SignalStatus.TP3_HIT
    assert s.realized_r == pytest.approx(0.5 * 1 + 0.3 * 2 + 0.2 * 3)
    assert s.outcome is Outcome.WIN


def test_tp1_then_breakeven_is_one_win_not_three():
    s = buy()
    s.update_bar(101.1, 100.2, 101.0, t(1))
    ev = s.update_bar(100.5, 99.9, 100.0, t(2))
    assert ev[-1].kind == "BE_HIT"
    assert s.realized_r == pytest.approx(0.5)
    assert s.outcome is Outcome.WIN


def test_ambiguous_bar_resolves_to_stop():
    s = buy()
    s.update_bar(101.5, 98.5, 100.0, t(1))   # touches SL and TP1 in the same bar
    assert s.status is SignalStatus.STOPPED
    assert s.realized_r == pytest.approx(-1.0)


def test_same_bar_tp_then_trailed_stop_is_conservative():
    s = buy()
    s.update_bar(101.2, 100.05, 100.5, t(1))   # TP1 hit, low stays above the new BE stop
    assert s.status is SignalStatus.TP1_HIT
    s2 = buy()
    ev = s2.update_bar(101.2, 99.95, 100.5, t(1))   # TP1 and BE stop both inside the bar: assume the worse order
    assert [e.kind for e in ev] == ["TP1_HIT", "BE_HIT"]
    assert s2.realized_r == pytest.approx(0.5) and not s2.is_open


def test_sell_uses_ask_for_exits():
    s = TradeState(direction=Direction.SELL, entry=100.0, stop_loss=101.0, take_profits=[99.0, 98.0, 97.0],
                   close_fractions=[0.5, 0.3, 0.2], opened_at=T0, max_hold=timedelta(hours=24))
    # bid high 100.85 + spread 0.2 = ask 101.05 >= SL
    s.update_bar(100.85, 99.5, 100.0, t(1), spread=0.2)
    assert s.status is SignalStatus.STOPPED
    s = TradeState(direction=Direction.SELL, entry=100.0, stop_loss=101.0, take_profits=[99.0, 98.0, 97.0],
                   close_fractions=[0.5, 0.3, 0.2], opened_at=T0, max_hold=timedelta(hours=24))
    # bid low 98.9 -> ask 99.1 > TP1 (99.0): not filled yet
    s.update_bar(100.3, 98.9, 99.5, t(1), spread=0.2)
    assert s.tp_hits == 0


def test_expiry_and_cancel():
    s = buy(max_hold=timedelta(minutes=10))
    s.update_bar(100.5, 99.5, 100.4, t(11))
    assert s.status is SignalStatus.EXPIRED and s.realized_r == pytest.approx(0.4)
    c = buy()
    ev = c.cancel(99.8, t(3))
    assert ev.kind == "CANCELLED" and c.outcome is Outcome.CANCELLED and c.realized_r == pytest.approx(-0.2)


def test_costs_reduce_r():
    s = buy(commission=0.1, exit_slippage=0.05)
    s.update_bar(100.2, 98.9, 99.0, t(1))
    assert s.realized_r == pytest.approx(-1.05 - 0.1)


def test_invalid_stop_rejected():
    with pytest.raises(ValueError):
        buy(stop_loss=100.5)
