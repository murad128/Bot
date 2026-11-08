"""
Technical indicators implemented with pandas/numpy only.

Every function is *causal*: the value at bar i depends only on bars <= i.
This is verified in tests/test_no_lookahead.py and is what allows the
backtester to pre-compute indicators once without leaking future data.

Input DataFrames use a UTC DatetimeIndex (bar OPEN time) and the columns
open, high, low, close, tick_volume (spread optional).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (RMA)."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr_s.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Classic Supertrend. direction: +1 bullish, -1 bearish."""
    a = atr(df, n).to_numpy()
    high, low, close = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + mult * a
    lower_basic = hl2 - mult * a
    size = len(df)
    upper = np.full(size, np.nan)
    lower = np.full(size, np.nan)
    line = np.full(size, np.nan)
    direction = np.zeros(size)
    for i in range(size):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i], lower[i] = upper_basic[i], lower_basic[i]
            direction[i] = 1 if close[i] >= hl2[i] else -1
        else:
            upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
            lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < lower[i] else 1
            else:
                direction[i] = 1 if close[i] > upper[i] else -1
        line[i] = lower[i] if direction[i] == 1 else upper[i]
    return pd.DataFrame({"supertrend": line, "supertrend_dir": direction}, index=df.index)


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)
    return out.where(gain.notna())


def stoch_rsi(close: pd.Series, rsi_n: int = 14, stoch_n: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(close, rsi_n)
    lo = r.rolling(stoch_n, min_periods=stoch_n).min()
    hi = r.rolling(stoch_n, min_periods=stoch_n).max()
    raw = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw.rolling(k, min_periods=k).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stochrsi_k": k_line, "stochrsi_d": d_line})


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy()
    out = np.full(len(tp), np.nan)
    if len(tp) >= n:
        win = sliding_window_view(tp, n)
        mean = win.mean(axis=1)
        mad = np.abs(win - mean[:, None]).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[n - 1:] = (tp[n - 1:] - mean) / (0.015 * mad)
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def keltner(df: pd.DataFrame, n: int = 20, atr_n: int = 10, mult: float = 2.0) -> pd.DataFrame:
    mid = ema(df["close"], n)
    a = atr(df, atr_n)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + mult * a, "kc_lower": mid - mult * a})


def rolling_percentile_rank(s: pd.Series, n: int = 200) -> pd.Series:
    """Percentile (0..1) of the current value within the trailing n values (causal)."""
    arr = s.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        win = sliding_window_view(arr, n)
        last = win[:, -1][:, None]
        valid = ~np.isnan(win)
        with np.errstate(invalid="ignore"):
            out[n - 1:] = ((win <= last) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        out[n - 1:][np.isnan(arr[n - 1:])] = np.nan
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------------------
# volume (tick volume)
# ---------------------------------------------------------------------------
def volume_features(df: pd.DataFrame, n: int = 20, spike_mult: float = 2.0, div_lookback: int = 20) -> pd.DataFrame:
    vol = df["tick_volume"].astype(float)
    vol_ma = sma(vol, n)
    ratio = vol / vol_ma.replace(0, np.nan)
    close = df["close"]
    at_high = close >= close.rolling(div_lookback, min_periods=div_lookback).max()
    at_low = close <= close.rolling(div_lookback, min_periods=div_lookback).min()
    short_vol = sma(vol, 5)
    # price making new extremes on fading participation
    weak = short_vol < 0.8 * vol_ma
    return pd.DataFrame({
        "vol_ma": vol_ma,
        "vol_ratio": ratio,
        "vol_spike": ratio >= spike_mult,
        "bear_vol_div": at_high & weak,   # new high on falling volume
        "bull_vol_div": at_low & weak,    # new low on falling volume
    })


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with every indicator column used by the engine."""
    out = df.copy()
    c = out["close"]
    for n in (20, 50, 100, 200):
        out[f"ema{n}"] = ema(c, n)
    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out = out.join(macd(c))
    out = out.join(adx(out))
    out = out.join(supertrend(out))
    out["rsi"] = rsi(c)
    out = out.join(stoch_rsi(c))
    out["cci"] = cci(out)
    out["atr"] = atr(out)
    out["atr_pct"] = rolling_percentile_rank(out["atr"], 200)
    out = out.join(bollinger(c))
    out = out.join(keltner(out))
    out["bb_squeeze"] = (out["bb_upper"] < out["kc_upper"]) & (out["bb_lower"] > out["kc_lower"])
    out = out.join(volume_features(out))
    out["ema50_slope"] = (out["ema50"] - out["ema50"].shift(5)) / out["atr"].replace(0, np.nan)
    return out
