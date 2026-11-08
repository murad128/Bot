"""
Historical data utilities.

Supported CSV formats (auto-detected):
1. MT5 "Export bars" (History Center / Ctrl+U, tab separated):
     <DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
   Dates are in BROKER SERVER TIME -> pass server_utc_offset (Vantage: 2 or 3).
   <SPREAD> is in points -> pass `point` to convert to price.
2. Generic: time,open,high,low,close[,tick_volume][,spread_price|spread_points]
   `time` is ISO-8601 (UTC assumed if no offset given).

Use `python main.py download ...` to fetch history directly from MT5 (Windows).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.domain import Timeframe

log = logging.getLogger(__name__)


def load_csv(path: str, server_utc_offset: float = 0.0, point: Optional[float] = None) -> pd.DataFrame:
    p = Path(path)
    head = p.read_text(encoding="utf-8", errors="ignore")[:500]
    if "<DATE>" in head.upper():
        raw = pd.read_csv(p, sep="\t")
        raw.columns = [c.strip("<>").lower() for c in raw.columns]
        ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str), format="%Y.%m.%d %H:%M:%S")
        idx = ts.dt.tz_localize("UTC") - pd.Timedelta(hours=server_utc_offset)
        df = pd.DataFrame({
            "open": raw["open"].astype(float).to_numpy(), "high": raw["high"].astype(float).to_numpy(),
            "low": raw["low"].astype(float).to_numpy(), "close": raw["close"].astype(float).to_numpy(),
            "tick_volume": raw.get("tickvol", pd.Series(0, index=raw.index)).astype(float).to_numpy(),
        }, index=pd.DatetimeIndex(idx, name="time"))
        if "spread" in raw and point:
            df["spread_price"] = (raw["spread"].astype(float) * point).to_numpy()
        else:
            df["spread_price"] = np.nan
    else:
        raw = pd.read_csv(p)
        raw.columns = [c.strip().lower() for c in raw.columns]
        tcol = next(c for c in raw.columns if c in ("time", "datetime", "date", "timestamp"))
        ts = pd.to_datetime(raw[tcol], utc=True)
        if server_utc_offset:
            ts = ts - pd.Timedelta(hours=server_utc_offset)
        vol = raw["tick_volume"] if "tick_volume" in raw else raw["volume"] if "volume" in raw else pd.Series(0, index=raw.index)
        df = pd.DataFrame({
            "open": raw["open"].astype(float).to_numpy(), "high": raw["high"].astype(float).to_numpy(),
            "low": raw["low"].astype(float).to_numpy(), "close": raw["close"].astype(float).to_numpy(),
            "tick_volume": vol.astype(float).to_numpy(),
        }, index=pd.DatetimeIndex(ts, name="time"))
        if "spread_price" in raw:
            df["spread_price"] = raw["spread_price"].astype(float).to_numpy()
        elif "spread_points" in raw and point:
            df["spread_price"] = (raw["spread_points"].astype(float) * point).to_numpy()
        elif "spread" in raw and point:
            df["spread_price"] = (raw["spread"].astype(float) * point).to_numpy()
        else:
            df["spread_price"] = np.nan
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    out.index.name = "time"
    out.to_csv(path)


def resample(df: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    """Aggregate to a higher timeframe. Bars are labelled by OPEN time (UTC)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
    if "spread_price" in df.columns:
        agg["spread_price"] = "mean"
    out = df.resample(tf.pandas_rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def detect_gaps(df: pd.DataFrame, tf: Timeframe, max_gap_bars: int = 6) -> pd.DataFrame:
    """Report gaps larger than max_gap_bars (weekends are expected and reported too)."""
    d = df.index.to_series().diff()
    gaps = d[d > pd.Timedelta(minutes=tf.minutes * max_gap_bars)]
    return pd.DataFrame({"gap_start": gaps.index - gaps.values, "gap_end": gaps.index, "length": gaps.values})


def download_mt5(market_data, symbol: str, tf: Timeframe, start: datetime, end: datetime, path: str,
                 chunk_days: int = 30) -> pd.DataFrame:
    """Download history from MT5 in chunks and save to CSV (UTC)."""
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        df = market_data.get_range(symbol, tf, cur, nxt)
        if df is not None and not df.empty:
            frames.append(df)
            log.info("%s %s %s -> %s: %d bars", symbol, tf.value, cur.date(), nxt.date(), len(df))
        cur = nxt
    if not frames:
        raise RuntimeError(f"No data downloaded for {symbol} {tf.value}")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    save_csv(out, path)
    return out


def synthetic_ohlc(n: int = 20000, tf: Timeframe = Timeframe.M5, start_price: float = 2000.0,
                   vol: float = 0.0006, seed: int = 7, start: str = "2024-01-01",
                   spread_price: float = 0.25, drift_scale: float = 1.0) -> pd.DataFrame:
    """
    SYNTHETIC random data with trending regimes - for unit tests and pipeline
    smoke tests ONLY. Results on this data say nothing about real performance.
    Weekend bars are excluded. drift_scale=0 gives a driftless random walk
    (used to check the backtester for look-ahead bias: no edge should appear).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n * 1.45), freq=tf.pandas_rule, tz="UTC")
    idx = idx[idx.dayofweek < 5][:n]
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(300, 1500))
        drift[i:i + length] = rng.choice([-1, 0, 1]) * vol * rng.uniform(0.05, 0.25) * drift_scale
        i += length
    sig = vol * np.exp(rng.normal(0, 0.3, n))
    rets = drift + sig * rng.standard_t(4, n) / np.sqrt(2)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = np.maximum(open_, close) + wick * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - wick * rng.uniform(0.2, 1.0, n)
    tv = rng.integers(50, 400, n).astype(float) * (1 + 5 * np.abs(rets) / vol / 10)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "tick_volume": tv,
                         "spread_price": spread_price}, index=pd.DatetimeIndex(idx, name="time"))
