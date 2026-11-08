"""
MetaTrader 5 data provider (official `MetaTrader5` Python package).

READ-ONLY by design: this module only calls initialize / copy_rates_* /
symbol_info* / symbol_select / terminal_info / shutdown. It never calls
order_send, order_check or any trading function, so the bot cannot place
orders on the account.

Requirements
* Windows (or Wine) with the MT5 terminal installed and logged in to Vantage.
  The `MetaTrader5` package is only published for Windows.
* "Algo trading" does NOT need to be enabled because no orders are sent.

Broker time
* MT5 returns bar/tick timestamps in TRADE SERVER time encoded as if it
  were UTC. Vantage servers run on GMT+2 / GMT+3 (New-York-close aligned,
  with DST). We convert to real UTC using MT5_SERVER_UTC_OFFSET, or detect
  the offset automatically from a fresh tick ("auto", default).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import InstrumentConfig, MT5Config
from core.domain import Tick, Timeframe
from core.market_data import DataProvider, DataProviderError

log = logging.getLogger(__name__)


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise DataProviderError(
            "MetaTrader5 package not available. It only installs on Windows: "
            "`pip install MetaTrader5` on the VPS running the MT5 terminal, "
            "or set DATA_PROVIDER=csv for offline use.") from exc


class MT5DataProvider(DataProvider):
    name = "mt5"

    def __init__(self, cfg: MT5Config, mt5_module=None):
        self.cfg = cfg
        self._mt5 = mt5_module
        self._connected = False
        self._offset_hours: Optional[int] = None
        self._offset_checked_at = 0.0
        if cfg.server_utc_offset not in ("auto", ""):
            self._offset_hours = int(cfg.server_utc_offset)

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = _import_mt5()
        return self._mt5

    # ------------------------------------------------------------------ connection
    def connect(self) -> bool:
        mt5 = self.mt5
        delay = 2.0
        for attempt in range(1, self.cfg.max_reconnect_attempts + 1):
            try:
                mt5.shutdown()
            except Exception:
                pass
            kwargs = {"timeout": self.cfg.timeout_ms}
            if self.cfg.terminal_path:
                kwargs["path"] = self.cfg.terminal_path
            if self.cfg.login:
                kwargs.update(login=self.cfg.login, password=self.cfg.password or "", server=self.cfg.server or "")
            ok = mt5.initialize(**kwargs)
            if ok:
                info = mt5.terminal_info()
                acc = mt5.account_info()
                self._connected = True
                log.info("MT5 connected: terminal=%s build=%s server=%s",
                         getattr(info, "name", "?"), getattr(info, "build", "?"), getattr(acc, "server", "?"))
                return True
            log.warning("MT5 initialize failed (attempt %d/%d): %s", attempt,
                        self.cfg.max_reconnect_attempts, mt5.last_error())
            time.sleep(delay)
            delay = min(delay * 2, 60)
        self._connected = False
        return False

    def shutdown(self) -> None:
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        try:
            info = self.mt5.terminal_info()
            return bool(info is not None and getattr(info, "connected", True))
        except Exception:
            return False

    # ------------------------------------------------------------------ time
    def server_offset(self, probe_symbol: Optional[str] = None) -> int:
        """Hours to subtract from MT5 timestamps to obtain UTC."""
        if self.cfg.server_utc_offset not in ("auto", ""):
            return int(self.cfg.server_utc_offset)
        if self._offset_hours is not None and time.monotonic() - self._offset_checked_at < 3600:
            return self._offset_hours
        if probe_symbol:
            t = self.mt5.symbol_info_tick(probe_symbol)
            if t is not None and t.time:
                diff = t.time - datetime.now(timezone.utc).timestamp()
                hours = int(round(diff / 3600.0))
                # only trust a fresh tick: offset must be integral and within +-14h
                if abs(diff - hours * 3600) < 600 and -14 <= hours <= 14:
                    if self._offset_hours not in (None, hours):
                        log.info("MT5 server UTC offset changed %s -> %s", self._offset_hours, hours)
                    self._offset_hours = hours
                    self._offset_checked_at = time.monotonic()
        if self._offset_hours is None:
            log.warning("Could not auto-detect MT5 server offset (market closed?) - assuming 0. "
                        "Set MT5_SERVER_UTC_OFFSET explicitly to avoid this.")
            return 0
        return self._offset_hours

    # ------------------------------------------------------------------ symbols
    def resolve_symbol(self, inst: InstrumentConfig) -> Optional[str]:
        mt5 = self.mt5
        candidates = []
        if inst.broker_symbol:
            candidates.append(inst.broker_symbol)
        candidates.append(inst.symbol + (self.cfg.symbol_suffix or ""))
        candidates.append(inst.symbol)
        for c in candidates:
            if mt5.symbol_info(c) is not None:
                mt5.symbol_select(c, True)
                return c
        found = mt5.symbols_get(f"{inst.symbol}*") or ()
        names = sorted((s.name for s in found), key=len)
        if names:
            mt5.symbol_select(names[0], True)
            log.info("Resolved %s -> %s", inst.symbol, names[0])
            return names[0]
        log.error("Symbol %s not found at broker (tried %s)", inst.symbol, candidates)
        return None

    def symbol_info(self, broker_symbol: str) -> Optional[dict]:
        info = self.mt5.symbol_info(broker_symbol)
        if info is None:
            return None
        return {"point": info.point, "digits": info.digits,
                "contract_size": getattr(info, "trade_contract_size", None),
                "spread_points": getattr(info, "spread", None)}

    # ------------------------------------------------------------------ data
    def _tf(self, tf: Timeframe):
        return getattr(self.mt5, f"TIMEFRAME_{tf.value}")

    def _to_frame(self, rates, broker_symbol: str) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            err = self.mt5.last_error()
            raise DataProviderError(f"No rates for {broker_symbol}: {err}")
        df = pd.DataFrame(rates)
        offset = self.server_offset(broker_symbol)
        idx = pd.to_datetime(df["time"], unit="s", utc=True) - pd.Timedelta(hours=offset)
        info = self.mt5.symbol_info(broker_symbol)
        point = info.point if info is not None else 0.0
        out = pd.DataFrame({
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
            "tick_volume": df["tick_volume"].astype(float).to_numpy(),
            "spread_price": (df["spread"].astype(float) * point).to_numpy() if "spread" in df else np.nan,
        }, index=pd.DatetimeIndex(idx, name="time"))
        return out

    def get_rates(self, broker_symbol: str, tf: Timeframe, count: int) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(broker_symbol, self._tf(tf), 0, int(count))
        return self._to_frame(rates, broker_symbol)

    def get_rates_range(self, broker_symbol: str, tf: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        offset = timedelta(hours=self.server_offset(broker_symbol))
        # MT5 expects server-time datetimes in "UTC" encoding
        s = (start + offset).replace(tzinfo=timezone.utc)
        e = (end + offset).replace(tzinfo=timezone.utc)
        rates = self.mt5.copy_rates_range(broker_symbol, self._tf(tf), s, e)
        return self._to_frame(rates, broker_symbol)

    def get_tick(self, broker_symbol: str) -> Optional[Tick]:
        t = self.mt5.symbol_info_tick(broker_symbol)
        if t is None:
            return None
        offset = self.server_offset(broker_symbol)
        ts = datetime.fromtimestamp(t.time, tz=timezone.utc) - timedelta(hours=offset)
        return Tick(broker_symbol, float(t.bid), float(t.ask), ts)
