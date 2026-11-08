"""MT5 connector tested against a fake MetaTrader5 module (the real one is Windows-only)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from config import InstrumentConfig, MT5Config
from core.domain import Timeframe
from core.market_data import MarketDataService
from integrations.mt5_connector import MT5DataProvider

OFFSET_H = 2


class FakeMT5:
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408

    def __init__(self, fail_init=0):
        self.fail_init = fail_init
        self.init_calls = 0
        self.connected = True
        self.orders_sent = 0
        self.symbols = {"XAUUSD+": SimpleNamespace(name="XAUUSD+", point=0.01, digits=2, spread=20),
                        "EURUSD": SimpleNamespace(name="EURUSD", point=0.00001, digits=5, spread=8)}

    def initialize(self, **kw):
        self.init_calls += 1
        if self.init_calls <= self.fail_init:
            return False
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (-1, "fake")

    def terminal_info(self):
        return SimpleNamespace(connected=self.connected, name="fake", build=1)

    def account_info(self):
        return SimpleNamespace(server="Vantage-Demo")

    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbol_select(self, name, enable):
        return True

    def symbols_get(self, pattern):
        p = pattern.rstrip("*")
        return [s for n, s in self.symbols.items() if n.startswith(p)]

    def _server_now(self):
        return datetime.now(timezone.utc).timestamp() + OFFSET_H * 3600

    def symbol_info_tick(self, name):
        return SimpleNamespace(time=int(self._server_now()) - 2, bid=2000.0, ask=2000.25)

    def copy_rates_from_pos(self, name, tf, pos, count):
        step = {1: 60, 5: 300, 15: 900}.get(tf, 3600)
        last_open = int(self._server_now() // step * step)        # forming bar
        times = np.array([last_open - step * i for i in range(count)][::-1])
        price = 2000 + np.arange(count) * 0.1
        dt = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                       ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        arr = np.zeros(count, dtype=dt)
        arr["time"], arr["open"], arr["close"] = times, price, price + 0.05
        arr["high"], arr["low"] = price + 0.2, price - 0.2
        arr["tick_volume"], arr["spread"] = 100, 20
        return arr

    def order_send(self, *a, **k):  # must never be called
        self.orders_sent += 1


def provider(fake, offset="auto"):
    return MT5DataProvider(MT5Config(server_utc_offset=offset, max_reconnect_attempts=3), mt5_module=fake)


def test_symbol_resolution_with_suffix(cfg):
    p = provider(FakeMT5())
    assert p.resolve_symbol(InstrumentConfig("XAUUSD", 0.01, 2, 0.1, 4)) == "XAUUSD+"
    assert p.resolve_symbol(InstrumentConfig("EURUSD", 0.00001, 5, 0.0001, 2)) == "EURUSD"
    assert p.resolve_symbol(InstrumentConfig("USDJPY", 0.001, 3, 0.01, 2)) is None


def test_server_offset_autodetect_and_utc_conversion():
    p = provider(FakeMT5())
    assert p.connect()
    assert p.server_offset("XAUUSD+") == OFFSET_H
    df = p.get_rates("XAUUSD+", Timeframe.M5, 10)
    assert df.index.tz is not None
    # forming bar open time must be within the last 5 minutes of real UTC
    assert datetime.now(timezone.utc) - df.index[-1].to_pydatetime() < timedelta(minutes=5)
    assert df["spread_price"].iloc[-1] == pytest.approx(0.20)
    t = p.get_tick("XAUUSD+")
    assert abs((datetime.now(timezone.utc) - t.time).total_seconds()) < 10


def test_fixed_offset_config():
    p = provider(FakeMT5(), offset="3")
    assert p.server_offset("XAUUSD+") == 3


def test_service_drops_forming_bar_and_specs(cfg):
    fake = FakeMT5()
    p = provider(fake)
    p.connect()
    svc = MarketDataService(p, cfg)
    df = svc.get_candles("XAUUSD", Timeframe.M5, 20)
    assert len(df) == 20
    assert (df.index[-1] + timedelta(minutes=5)).to_pydatetime() <= datetime.now(timezone.utc)
    spec = svc.spec("XAUUSD")
    assert spec.broker_symbol == "XAUUSD+" and spec.digits == 2 and spec.is_gold
    assert svc.is_new_bar("XAUUSD", Timeframe.M5, df) and not svc.is_new_bar("XAUUSD", Timeframe.M5, df)
    assert fake.orders_sent == 0


def test_reconnect_after_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeMT5(fail_init=2)
    p = provider(fake)
    assert p.connect() and fake.init_calls == 3
    fake.connected = False
    assert not p.is_connected()


def test_connector_has_no_trading_calls():
    import inspect
    import integrations.mt5_connector as m
    src = inspect.getsource(m)
    for forbidden in ("order_send(", "order_check(", "positions_get(", "Close("):
        assert forbidden not in src.replace("never calls order_send, order_check", "")
