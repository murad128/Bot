"""
Simulated Vantage MT5 terminal for tests ONLY (never imported by production code).

Mimics the parts of the MetaTrader5 package the bot uses, with broker-like
behaviour: server clock = New York time + 7h (UTC+2 in winter, UTC+3 in
summer, like Vantage), bars built on the SERVER clock (so H4/D1 start at
server midnight = NY close), weekend closure, a one-hour daily gold break,
a suffixed gold symbol, and a DEMO account.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

NY = "America/New_York"


def to_server(idx_utc):
    """UTC DatetimeIndex -> naive server wall clock (NY + 7h)."""
    return (idx_utc.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7))
STEP = {1: 1, 5: 5, 15: 15, 30: 30, 16385: 60, 16388: 240, 16408: 1440}


class FakeVantageMT5:
    __name__ = "FakeVantageMT5"
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408

    def __init__(self, trade_mode=0, days=70, seed=4, end=None):
        self.trade_mode = trade_mode
        self.orders_sent = 0
        rng = np.random.default_rng(seed)
        end = (end or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
        idx = pd.date_range(end - timedelta(days=days), end, freq="1min", tz="UTC")
        server = to_server(idx)
        # market closed: Sat & Sun server time, plus daily break 00:00-01:00 server (gold)
        open_mask = (server.dayofweek < 5) & (server.hour != 0)
        idx = idx[open_mask]
        n = len(idx)
        drift = np.repeat(rng.choice([-1, 1], n // 2000 + 1) * 0.00002, 2000)[:n]
        close = 2650 * np.exp(np.cumsum(drift + rng.normal(0, 0.00025, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wick = np.abs(rng.normal(0, 0.15, n))
        m1 = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                           "low": np.minimum(open_, close) - wick, "close": close,
                           "tick_volume": rng.integers(20, 200, n).astype(float)}, index=idx)
        self.m1 = m1.round(2)
        self.end = end
        self._bars = {}
        self.symbols = {
            "XAUUSD+": SimpleNamespace(name="XAUUSD+", point=0.01, digits=2, spread=25, trade_contract_size=100.0,
                                       path="Metals\\XAUUSD+", visible=True, description="Gold vs US Dollar"),
            "XAUEUR+": SimpleNamespace(name="XAUEUR+", point=0.01, digits=2, spread=40, trade_contract_size=100.0,
                                       path="Metals\\XAUEUR+", visible=False, description="Gold vs Euro"),
            "XAUUSD.fut": SimpleNamespace(name="XAUUSD.fut", point=0.1, digits=1, spread=5, trade_contract_size=100.0,
                                          path="Futures\\XAUUSD.fut", visible=False, description="Gold future"),
            "EURUSD+": SimpleNamespace(name="EURUSD+", point=0.00001, digits=5, spread=12, trade_contract_size=1e5,
                                       path="Forex\\EURUSD+", visible=True, description="Euro vs US Dollar"),
        }

    # ---- broker bars: aggregated on the SERVER clock
    def _tf_bars(self, tf):
        if tf in self._bars:
            return self._bars[tf]
        m = STEP[tf]
        df = self.m1.copy()
        df.index = to_server(df.index)                                # server clock (naive)
        if m > 1:
            df = df.resample(f"{m}min", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}).dropna()
        self._bars[tf] = df
        return df

    def _array(self, df):
        dt = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                       ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        a = np.zeros(len(df), dtype=dt)
        a["time"] = df.index.as_unit("s").asi8          # server wall time encoded as epoch, like MT5
        for c in ("open", "high", "low", "close"):
            a[c] = df[c].to_numpy()
        a["tick_volume"], a["spread"] = df["tick_volume"].to_numpy(), 25
        return a

    # ---- MetaTrader5 API subset
    def initialize(self, **kw):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return (1, "Success")

    def terminal_info(self):
        return SimpleNamespace(connected=True, build=4755, trade_allowed=False, name="Vantage International MT5")

    def account_info(self):
        return SimpleNamespace(login=24681357, server="VantageInternational-Demo", trade_mode=self.trade_mode,
                               company="Vantage International Group Limited", currency="USD",
                               balance=10000.0, equity=10012.5, leverage=500, margin_free=10012.5,
                               name="SHOULD NEVER BE PRINTED")

    def symbol_info(self, name):
        return self.symbols.get(name)

    def symbol_select(self, name, enable):
        return name in self.symbols

    def symbols_get(self, pattern="*"):
        core = pattern.strip("*")
        return [s for n, s in self.symbols.items() if core in n]

    def symbol_info_tick(self, name):
        last = self.m1.iloc[-1]
        t = to_server(self.m1.index[-1:])[0]
        return SimpleNamespace(time=int(t.value // 10**9), bid=float(last["close"]), ask=float(last["close"]) + 0.25)

    def copy_rates_from_pos(self, name, tf, pos, count):
        return self._array(self._tf_bars(tf).tail(count))

    def copy_rates_range(self, name, tf, start, end):
        df = self._tf_bars(tf)
        s, e = pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        return self._array(df[(df.index >= s) & (df.index <= e)])

    def order_send(self, *a, **k):          # present on the real package; must be unreachable
        self.orders_sent += 1
        raise AssertionError("order_send must never be called")
