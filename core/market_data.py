"""
Market data abstraction.

`DataProvider` is the interface every source implements (MT5, CSV, ...).
`MarketDataService` adds what the engine relies on:
  * closed bars only (the still-forming bar is dropped)
  * validation: sorted, de-duplicated, OHLC-consistent, no NaNs
  * staleness detection (no new bars / frozen ticks)
  * automatic reconnect on provider errors
  * symbol-spec resolution (live values from the provider, config fallback)

DataFrame convention: UTC DatetimeIndex of bar OPEN times; columns
open, high, low, close, tick_volume, spread_price (spread in price units).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config, InstrumentConfig
from core.domain import SymbolSpec, Tick, Timeframe

log = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]


class DataProviderError(RuntimeError):
    pass


class DataProvider(ABC):
    name = "abstract"

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_rates(self, broker_symbol: str, tf: Timeframe, count: int) -> pd.DataFrame: ...

    @abstractmethod
    def get_rates_range(self, broker_symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame: ...

    @abstractmethod
    def get_tick(self, broker_symbol: str) -> Optional[Tick]: ...

    @abstractmethod
    def resolve_symbol(self, inst: InstrumentConfig) -> Optional[str]: ...

    def symbol_info(self, broker_symbol: str) -> Optional[dict]:
        return None

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def validate_ohlc(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Clean a raw OHLC frame. Rows that violate OHLC invariants are dropped (and logged)."""
    if df is None or df.empty:
        return df
    out = df[~df.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=OHLC)
    bad = (out["high"] < out[["open", "close"]].max(axis=1)) | (out["low"] > out[["open", "close"]].min(axis=1)) \
        | (out["high"] < out["low"]) | (out[OHLC] <= 0).any(axis=1)
    if bad.any():
        log.warning("Dropped %d invalid OHLC rows %s", int(bad.sum()), context)
        out = out[~bad]
    if "tick_volume" not in out.columns:
        out["tick_volume"] = 0.0
    if "spread_price" not in out.columns:
        out["spread_price"] = np.nan
    return out


def closed_bars(df: pd.DataFrame, tf: Timeframe, now: datetime) -> pd.DataFrame:
    """Keep bars whose close time (open + tf) is <= now."""
    if df is None or df.empty:
        return df
    close_times = df.index + pd.Timedelta(minutes=tf.minutes)
    return df[close_times <= pd.Timestamp(now)]


class MarketDataService:
    def __init__(self, provider: DataProvider, cfg: Config):
        self.provider = provider
        self.cfg = cfg
        self._symbols: Dict[str, str] = {}
        self._specs: Dict[str, SymbolSpec] = {}
        self._last_bar: Dict[Tuple[str, Timeframe], pd.Timestamp] = {}
        self.consecutive_failures = 0

    # ------------------------------------------------------------------ connection
    def ensure_connected(self) -> bool:
        if self.provider.is_connected():
            return True
        log.warning("Data provider %s disconnected - reconnecting", self.provider.name)
        ok = self.provider.connect()
        if ok:
            self._symbols.clear()
        return ok

    def _call(self, fn, *args):
        try:
            out = fn(*args)
            self.consecutive_failures = 0
            return out
        except Exception as exc:
            self.consecutive_failures += 1
            log.warning("Provider call %s failed (%s); attempting reconnect", getattr(fn, "__name__", fn), exc)
            if self.ensure_connected():
                return fn(*args)
            raise DataProviderError(str(exc)) from exc

    # ------------------------------------------------------------------ symbols
    def broker_symbol(self, symbol: str) -> Optional[str]:
        if symbol not in self._symbols:
            bs = self.provider.resolve_symbol(self.cfg.instrument(symbol))
            if bs is None:
                return None
            self._symbols[symbol] = bs
        return self._symbols[symbol]

    def spec(self, symbol: str) -> SymbolSpec:
        if symbol in self._specs:
            return self._specs[symbol]
        inst = self.cfg.instrument(symbol)
        bs = self.broker_symbol(symbol) or symbol
        info = self.provider.symbol_info(bs) or {}
        point = float(info.get("point") or inst.point)
        digits = int(info.get("digits") if info.get("digits") is not None else inst.digits)
        spec = SymbolSpec(symbol=symbol, broker_symbol=bs, point=point, digits=digits, pip_size=inst.pip_size,
                          max_spread_pips=inst.max_spread_pips, is_gold=inst.is_gold,
                          currencies=inst.currencies, commission_pips=inst.commission_pips,
                          slippage_pips=inst.slippage_pips)
        self._specs[symbol] = spec
        return spec

    # ------------------------------------------------------------------ data
    def get_candles(self, symbol: str, tf: Timeframe, count: int, raw_symbol: bool = False,
                    include_forming: bool = False) -> Optional[pd.DataFrame]:
        bs = symbol if raw_symbol else self.broker_symbol(symbol)
        if bs is None:
            log.error("Symbol %s not available at the data provider", symbol)
            return None
        df = self._call(self.provider.get_rates, bs, tf, count + 1)
        df = validate_ohlc(df, f"{symbol} {tf.value}")
        if df is None or df.empty:
            return None
        if not include_forming:
            df = closed_bars(df, tf, self.provider.now())
        return df.tail(count)

    def get_range(self, symbol: str, tf: Timeframe, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        bs = self.broker_symbol(symbol)
        if bs is None:
            return None
        df = validate_ohlc(self._call(self.provider.get_rates_range, bs, tf, start, end), f"{symbol} {tf.value}")
        return closed_bars(df, tf, self.provider.now()) if df is not None else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        bs = self.broker_symbol(symbol)
        if bs is None:
            return None
        t = self._call(self.provider.get_tick, bs)
        if t is None or t.bid <= 0 or t.ask <= 0 or t.ask < t.bid:
            return None
        return Tick(symbol, t.bid, t.ask, t.time)

    def is_new_bar(self, symbol: str, tf: Timeframe, df: pd.DataFrame) -> bool:
        """True exactly once per newly closed bar (duplicate-processing guard)."""
        if df is None or df.empty:
            return False
        last = df.index[-1]
        key = (symbol, tf)
        if self._last_bar.get(key) == last:
            return False
        self._last_bar[key] = last
        return True

    def tick_is_stale(self, tick: Optional[Tick], max_age_s: int) -> bool:
        if tick is None:
            return True
        return (self.provider.now() - tick.time).total_seconds() > max_age_s


# ---------------------------------------------------------------------------
# CSV provider (offline / testing / alternative data source)
# ---------------------------------------------------------------------------
class CSVDataProvider(DataProvider):
    """
    Reads <dir>/<SYMBOL>_<TF>.csv (see backtesting/historical_data.py for formats).
    Missing timeframes are resampled from the lowest available one.
    `now()` is the end of the data, so this provider replays the LAST state
    of the files; it is meant for offline checks, not live trading.
    """
    name = "csv"

    def __init__(self, data_dir: str, default_spread_points: float = 0.0):
        self.dir = Path(data_dir)
        self._cache: Dict[Tuple[str, Timeframe], pd.DataFrame] = {}
        self.default_spread_points = default_spread_points
        self._now: Optional[datetime] = None

    def connect(self) -> bool:
        return self.dir.exists()

    def shutdown(self) -> None:
        self._cache.clear()

    def is_connected(self) -> bool:
        return self.dir.exists()

    def _load(self, symbol: str, tf: Timeframe) -> pd.DataFrame:
        key = (symbol, tf)
        if key in self._cache:
            return self._cache[key]
        from backtesting.historical_data import load_csv, resample
        path = self.dir / f"{symbol}_{tf.value}.csv"
        if path.exists():
            df = load_csv(str(path))
        else:
            base = None
            for cand in Timeframe:
                p = self.dir / f"{symbol}_{cand.value}.csv"
                if p.exists() and cand.minutes <= tf.minutes:
                    base = (cand, p)
            if base is None:
                raise DataProviderError(f"No CSV data for {symbol} {tf.value} in {self.dir}")
            df = resample(load_csv(str(base[1])), tf)
        self._cache[key] = df
        return df

    def get_rates(self, broker_symbol, tf, count):
        return self._load(broker_symbol, tf).tail(count)

    def get_rates_range(self, broker_symbol, tf, start, end):
        df = self._load(broker_symbol, tf)
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

    def get_tick(self, broker_symbol):
        df = self._load(broker_symbol, Timeframe.M5) if (self.dir / f"{broker_symbol}_M5.csv").exists() \
            else self._load(broker_symbol, Timeframe.M15)
        last = df.iloc[-1]
        sp = float(last.get("spread_price", np.nan))
        if np.isnan(sp):
            sp = 0.0
        return Tick(broker_symbol, float(last["close"]), float(last["close"]) + sp, self.now())

    def resolve_symbol(self, inst):
        sym = inst.broker_symbol or inst.symbol
        return sym if any(self.dir.glob(f"{sym}_*.csv")) else None

    def now(self) -> datetime:
        if self._now is None:
            latest = None
            for p in self.dir.glob("*_*.csv"):
                sym, tf = p.stem.rsplit("_", 1)
                try:
                    t = Timeframe(tf)
                except ValueError:
                    continue
                df = self._load(sym, t)
                end = df.index[-1] + pd.Timedelta(minutes=t.minutes)
                latest = end if latest is None else max(latest, end)
            self._now = latest.to_pydatetime() if latest is not None else datetime.now(timezone.utc)
        return self._now
