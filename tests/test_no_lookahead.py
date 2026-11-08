"""
Look-ahead / data-leakage tests.

1. Every engine feature at bar k computed on df[:k+1] must equal the value
   computed on the full series (=> features are causal and the backtester may
   pre-compute them).
2. Swing pivots are never marked without their full confirmation window.
3. The backtester never hands the engine a bar that had not closed yet.
"""
import numpy as np
import pandas as pd
import pytest

from backtesting.backtest_engine import Backtester
from backtesting.historical_data import synthetic_ohlc
from core.domain import Timeframe
from core.price_action import swing_points
from core.signal_engine import prepare_frame


@pytest.fixture(scope="module")
def data():
    return synthetic_ohlc(n=1500, seed=5)


def test_features_are_causal(data):
    full = prepare_frame(data)
    for k in (400, 777, 1200, 1499):
        part = prepare_frame(data.iloc[:k + 1])
        a, b = part.iloc[-1], full.iloc[k]
        for col in full.columns:
            x, y = a[col], b[col]
            if isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
                assert bool(x) == bool(y), (k, col)
            elif pd.isna(x) and pd.isna(y):
                continue
            else:
                assert float(x) == pytest.approx(float(y), rel=1e-9, abs=1e-9), (k, col)


def test_swings_need_confirmation(data):
    h, l = data["high"].to_numpy(), data["low"].to_numpy()
    for s in (2, 3, 5):
        sh, sl = swing_points(h, l, s)
        assert not sh[-s:].any() and not sl[-s:].any()
        # a pivot found on the full series must also be found once its window is complete
        idx = np.flatnonzero(sh)[:20]
        for j in idx:
            sh2, _ = swing_points(h[:j + s + 1], l[:j + s + 1], s)
            assert sh2[j]


def test_backtester_only_passes_closed_bars(cfg, gold_spec, data):
    bt = Backtester(cfg)
    prepared = bt.prepare("XAUUSD", data, gold_spec)
    base_close = prepared.close_ns[Timeframe.M5]
    for i in (300, 801, 1333):
        now_ns = int(base_close[i])
        frames = bt._frames_at(prepared, now_ns)
        for tf, df in frames.items():
            if df.empty:
                continue
            last_close = (df.index[-1] + pd.Timedelta(minutes=tf.minutes)).as_unit("ns").value
            assert last_close <= now_ns, tf
            # and the next bar of that timeframe (if any) has NOT closed yet
            full = prepared.frames[tf]
            pos = full.index.get_loc(df.index[-1])
            if pos + 1 < len(full):
                nxt_close = (full.index[pos + 1] + pd.Timedelta(minutes=tf.minutes)).as_unit("ns").value
                assert nxt_close > now_ns, tf
