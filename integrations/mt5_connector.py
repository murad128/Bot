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

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import InstrumentConfig, MT5Config
from core.domain import Tick, Timeframe
from core.market_data import DataProvider, DataProviderError

log = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")
_NY_PLUS = pd.Timedelta(hours=7)        # GMT+2/+3 "New York close" servers: server clock = NY + 7h


def ny_close_offset(at: datetime) -> int:
    """UTC offset (hours) of a NY-close server at a given moment: +2 in NY winter, +3 in NY summer."""
    return int(at.astimezone(_NY).utcoffset().total_seconds() // 3600) + 7

# Everything the bot is allowed to touch in the MetaTrader5 package. Any other
# attribute (order_send, order_check, positions_get, orders_get, history_*,
# order_calc_*, ...) raises PermissionError, so trading is impossible even by
# mistake in future code.
READ_ONLY_API = frozenset({
    "initialize", "shutdown", "last_error", "version", "terminal_info", "account_info",
    "symbol_info", "symbol_info_tick", "symbol_select", "symbols_get", "symbols_total",
    "copy_rates_from_pos", "copy_rates_range", "copy_rates_from",
})
ACCOUNT_TRADE_MODE = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


class ReadOnlyMT5:
    """Proxy around the MetaTrader5 module that only exposes data functions."""

    def __init__(self, module):
        object.__setattr__(self, "_m", module)

    def __getattr__(self, name):
        if name.startswith("TIMEFRAME_") or name.startswith("ACCOUNT_") or name in READ_ONLY_API \
                or name in ("__name__", "__file__", "__version__"):
            return getattr(self._m, name)
        raise PermissionError(f"MetaTrader5.{name} is blocked: this bot is signal-only and read-only")

    def __setattr__(self, name, value):
        raise PermissionError("ReadOnlyMT5 is immutable")

    @property
    def module_file(self) -> str:
        return str(getattr(self._m, "__file__", "") or "")


def mask_login(login) -> str:
    s = str(login or "")
    return ("*" * max(0, len(s) - 3) + s[-3:]) if s else "?"


def _import_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
        return mt5
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise DataProviderError(
            "MetaTrader5 package not available. It only installs on Windows: "
            "`pip install MetaTrader5` on the VPS running the MT5 terminal, "
            "(the live bot requires it; CSV data is only for offline backtests).") from exc


class MT5DataProvider(DataProvider):
    name = "mt5"

    def __init__(self, cfg: MT5Config, mt5_module=None, state_path: Optional[str] = None):
        self.cfg = cfg
        self._mt5 = ReadOnlyMT5(mt5_module) if mt5_module is not None else None
        self._connected = False
        self._offset_hours: Optional[int] = None
        self._offset_checked_at = 0.0
        self._offset_fresh = False
        self.state_path = Path(state_path) if state_path else None
        # "ny_close": DST-aware conversion (server = NY + 7h, e.g. Vantage);
        # "constant": one fixed offset for every timestamp.
        mode = (getattr(cfg, "server_time_mode", "auto") or "auto").lower()
        self._mode = mode if mode in ("ny_close", "constant") else None
        if cfg.server_utc_offset not in ("auto", "") and self._mode is None:
            self._mode = "constant"
        if cfg.server_utc_offset not in ("auto", ""):
            self._offset_hours = int(cfg.server_utc_offset)
            self._offset_fresh = True
        else:
            self._offset_hours = self._load_offset()

    @property
    def mt5(self):
        if self._mt5 is None:
            self._mt5 = ReadOnlyMT5(_import_mt5())
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
                if acc is None:
                    log.warning("MT5 terminal is running but not logged in to any account: %s", mt5.last_error())
                    mt5.shutdown()
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                mode = ACCOUNT_TRADE_MODE.get(getattr(acc, "trade_mode", None), "UNKNOWN")
                if self.cfg.require_demo and mode != "DEMO":
                    mt5.shutdown()
                    self._connected = False
                    raise DataProviderError(
                        f"Account {mask_login(getattr(acc, 'login', None))} on {getattr(acc, 'server', '?')} "
                        f"is a {mode} account. "
                        "MT5_REQUIRE_DEMO=true: refusing to run on anything but a DEMO account.")
                self._connected = True
                log.info("MT5 connected: terminal build=%s server=%s account=%s mode=%s",
                         getattr(info, "build", "?"), getattr(acc, "server", "?"),
                         mask_login(getattr(acc, "login", None)), mode)
                return True
            log.warning("MT5 initialize failed (attempt %d/%d): %s", attempt,
                        self.cfg.max_reconnect_attempts, mt5.last_error())
            time.sleep(delay)
            delay = min(delay * 2, 60)
        self._connected = False
        return False

    def account_summary(self) -> dict:
        """Account facts with the login masked and the holder's name omitted."""
        acc = self.mt5.account_info()
        if acc is None:
            return {}
        return {"login": mask_login(acc.login), "server": acc.server, "company": getattr(acc, "company", ""),
                "trade_mode": ACCOUNT_TRADE_MODE.get(acc.trade_mode, "UNKNOWN"), "currency": acc.currency,
                "balance": acc.balance, "equity": acc.equity, "leverage": acc.leverage,
                "margin_free": getattr(acc, "margin_free", None)}

    def terminal_summary(self) -> dict:
        t = self.mt5.terminal_info()
        if t is None:
            return {}
        return {"build": getattr(t, "build", None), "connected": getattr(t, "connected", None),
                "algo_trading_enabled": getattr(t, "trade_allowed", None),
                "package_version": getattr(self.mt5, "__version__", None),
                "package_file": getattr(self.mt5, "module_file", "")}

    def gold_candidates(self) -> list:
        """Every broker symbol that looks like spot gold vs USD."""
        out = []
        for s in self.mt5.symbols_get("*XAU*") or ():
            name = s.name
            if "USD" not in name.upper():
                continue
            out.append({"name": name, "path": getattr(s, "path", ""), "digits": s.digits, "point": s.point,
                        "contract_size": getattr(s, "trade_contract_size", None),
                        "visible": getattr(s, "visible", None), "description": getattr(s, "description", "")})
        out.sort(key=lambda d: (not d["name"].upper().startswith("XAUUSD"), len(d["name"])))
        return out

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
        """
        Hours to subtract from MT5 timestamps to obtain UTC.

        Order of preference: explicit MT5_SERVER_UTC_OFFSET -> detected from a
        fresh tick (re-checked hourly, catches DST changes) -> last detected
        value persisted on disk. If none is known, raise instead of guessing:
        a wrong offset silently shifts every bar by hours.
        """
        if self.cfg.server_utc_offset not in ("auto", ""):
            return int(self.cfg.server_utc_offset)
        if self._offset_fresh and time.monotonic() - self._offset_checked_at < 3600:
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
                    self._offset_fresh = True
                    self._offset_checked_at = time.monotonic()
                    if (getattr(self.cfg, "server_time_mode", "auto") or "auto").lower() == "auto":
                        new_mode = "ny_close" if hours == ny_close_offset(datetime.now(timezone.utc)) else "constant"
                        if new_mode != self._mode:
                            log.info("MT5 server time mode: %s (current offset UTC%+d)", new_mode, hours)
                        self._mode = new_mode
                    self._save_offset(hours)
        if self._offset_hours is None:
            raise DataProviderError(
                "MT5 server UTC offset unknown (market closed, no fresh tick and no saved value). "
                "Set MT5_SERVER_UTC_OFFSET (Vantage: 2 in winter, 3 in summer) or retry when the market is open.")
        return self._offset_hours

    def _load_offset(self) -> Optional[int]:
        if not self.state_path or not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text())
            v = int(data["offset_hours"])
            if self._mode is None and data.get("mode") in ("ny_close", "constant"):
                self._mode = data["mode"]
            log.info("Using last detected MT5 server offset %+d h, mode %s (will re-detect from live ticks)",
                     v, self._mode)
            return v
        except Exception:
            return None

    def _save_offset(self, hours: int) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps({"offset_hours": hours, "mode": self._mode,
                                                   "detected_at": datetime.now(timezone.utc).isoformat()}))
        except Exception as exc:
            log.debug("Could not persist MT5 offset: %s", exc)

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

    @property
    def time_mode(self) -> str:
        return self._mode or "constant"

    def server_to_utc(self, epoch_seconds, probe_symbol: Optional[str] = None) -> pd.DatetimeIndex:
        """MT5 timestamps (server wall clock encoded as epoch) -> UTC, DST-correct in ny_close mode."""
        offset = self.server_offset(probe_symbol)
        naive = pd.to_datetime(np.asarray(epoch_seconds, dtype="int64"), unit="s")
        if self.time_mode == "ny_close":
            return (naive - _NY_PLUS).tz_localize(_NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        return naive.tz_localize("UTC") - pd.Timedelta(hours=offset)

    def utc_to_server(self, dt: datetime, probe_symbol: Optional[str] = None) -> datetime:
        """UTC -> server wall clock as a UTC-tagged datetime (the encoding MT5 expects)."""
        offset = self.server_offset(probe_symbol)
        if self.time_mode == "ny_close":
            wall = dt.astimezone(_NY).replace(tzinfo=None) + timedelta(hours=7)
        else:
            wall = dt.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=offset)
        return wall.replace(tzinfo=timezone.utc)

    def _to_frame(self, rates, broker_symbol: str) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            err = self.mt5.last_error()
            raise DataProviderError(f"No rates for {broker_symbol}: {err}")
        df = pd.DataFrame(rates)
        idx = self.server_to_utc(df["time"].to_numpy(), broker_symbol)
        keep = ~idx.isna()                       # the repeated NY fall-back hour (market closed) is dropped
        df, idx = df[keep], idx[keep]
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
        # MT5 expects server-time datetimes in "UTC" encoding
        s = self.utc_to_server(start, broker_symbol)
        e = self.utc_to_server(end, broker_symbol)
        rates = self.mt5.copy_rates_range(broker_symbol, self._tf(tf), s, e)
        return self._to_frame(rates, broker_symbol)

    def get_tick(self, broker_symbol: str) -> Optional[Tick]:
        t = self.mt5.symbol_info_tick(broker_symbol)
        if t is None:
            return None
        ts = self.server_to_utc([int(t.time)], broker_symbol)[0].to_pydatetime()
        return Tick(broker_symbol, float(t.bid), float(t.ask), ts)
