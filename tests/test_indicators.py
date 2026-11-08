import numpy as np
import pandas as pd
import pytest

from core import technical_analysis as ta


def _frame(close, spread=0.5):
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="5min", tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                         "low": np.minimum(open_, close) - spread, "close": close,
                         "tick_volume": np.full(len(close), 100.0)}, index=idx)


def test_ema_matches_reference_formula():
    s = pd.Series(np.arange(1, 51, dtype=float))
    e = ta.ema(s, 10)
    alpha = 2 / 11
    ref = s.iloc[0]
    for x in s.iloc[1:]:
        ref = alpha * x + (1 - alpha) * ref
    assert e.iloc[-1] == pytest.approx(ref)
    assert e.iloc[:9].isna().all()


def test_rsi_extremes_and_midpoint():
    up = pd.Series(np.arange(100, 160, dtype=float))
    assert ta.rsi(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(160, 100, -1, dtype=float))
    assert ta.rsi(down).iloc[-1] == pytest.approx(0.0)
    alt = pd.Series([100 + (i % 2) for i in range(200)], dtype=float)
    assert ta.rsi(alt).iloc[-1] == pytest.approx(50.0, abs=3)


def test_atr_constant_range():
    df = _frame(np.full(100, 50.0), spread=1.0)
    assert ta.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_cci_matches_naive_implementation():
    rng = np.random.default_rng(0)
    df = _frame(100 + rng.normal(0, 1, 300).cumsum())
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    naive = (tp - sma) / (0.015 * mad)
    pd.testing.assert_series_equal(ta.cci(df, 20), naive, check_names=False)


def test_supertrend_follows_trend():
    df = _frame(np.linspace(100, 200, 300))
    st = ta.supertrend(df)
    assert st["supertrend_dir"].iloc[-1] == 1
    assert (st["supertrend"].iloc[-50:] < df["close"].iloc[-50:]).all()
    df2 = _frame(np.linspace(200, 100, 300))
    assert ta.supertrend(df2)["supertrend_dir"].iloc[-1] == -1


def test_adx_strong_trend_high():
    df = _frame(np.linspace(100, 200, 300))
    a = ta.adx(df)
    assert a["adx"].iloc[-1] > 40
    assert a["plus_di"].iloc[-1] > a["minus_di"].iloc[-1]


def test_bollinger_and_keltner_shapes():
    rng = np.random.default_rng(1)
    df = _frame(100 + rng.normal(0, 1, 300).cumsum())
    bb = ta.bollinger(df["close"])
    valid = bb.dropna()
    assert (valid["bb_upper"] > valid["bb_mid"]).all() and (valid["bb_lower"] < valid["bb_mid"]).all()
    kc = ta.keltner(df).dropna()
    assert (kc["kc_upper"] > kc["kc_lower"]).all()


def test_percentile_rank_bounds():
    s = pd.Series(np.arange(300, dtype=float))
    r = ta.rolling_percentile_rank(s, 50)
    assert r.iloc[-1] == pytest.approx(1.0)
    assert r.iloc[:49].isna().all()


def test_compute_indicators_columns():
    rng = np.random.default_rng(2)
    df = ta.compute_indicators(_frame(100 + rng.normal(0, 1, 400).cumsum()))
    for col in ("ema20", "ema200", "macd_hist", "adx", "supertrend_dir", "rsi", "stochrsi_k", "cci",
                "atr", "atr_pct", "bb_width", "kc_upper", "bb_squeeze", "vol_spike", "bear_vol_div"):
        assert col in df.columns
    assert not np.isnan(df["ema200"].iloc[-1])
