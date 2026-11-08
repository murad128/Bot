"""
Macro inputs for gold analysis.

* US Treasury yields: FRED (St. Louis Fed) - free API key required (FRED_API_KEY).
    DGS10  = 10-year nominal yield (daily)
    DFII10 = 10-year TIPS real yield (daily) - the series most relevant for gold
* DXY: read from the MT5 data provider if the broker offers a dollar-index
  symbol (set DXY_SYMBOL). If not available, this input is reported as
  unavailable - it is never approximated with made-up values.

Both are daily-resolution inputs; they describe the backdrop, not entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)


@dataclass
class YieldReading:
    series: str
    latest: float
    change_5d_bp: float   # basis points over ~5 observations
    as_of: str


class FredClient:
    URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str], session: Optional[requests.Session] = None,
                 cache_minutes: int = 180):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.cache: Dict[str, Tuple[datetime, YieldReading]] = {}
        self.cache_ttl = timedelta(minutes=cache_minutes)
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def reading(self, series: str) -> Optional[YieldReading]:
        if not self.api_key:
            return None
        now = datetime.now(timezone.utc)
        hit = self.cache.get(series)
        if hit and now - hit[0] < self.cache_ttl:
            return hit[1]
        try:
            resp = self.session.get(self.URL, timeout=15, params={
                "series_id": series, "api_key": self.api_key, "file_type": "json",
                "sort_order": "desc", "limit": 15})
            resp.raise_for_status()
            obs: List[dict] = [o for o in resp.json().get("observations", []) if o.get("value") not in (".", "", None)]
            if len(obs) < 6:
                raise ValueError(f"not enough observations for {series}")
            latest = float(obs[0]["value"])
            prior = float(obs[5]["value"])
            r = YieldReading(series, latest, round((latest - prior) * 100, 1), obs[0]["date"])
            self.cache[series] = (now, r)
            self.last_error = None
            return r
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("FRED %s fetch failed: %s", series, exc)
            return hit[1] if hit else None


@dataclass
class DxyReading:
    symbol: str
    trend: int            # +1 rising (USD strength), -1 falling, 0 flat
    change_pct_5d: float
    as_of: datetime


class MacroDataService:
    def __init__(self, fred: FredClient, market_data=None, dxy_symbol: Optional[str] = None):
        self.fred = fred
        self.market_data = market_data
        self.dxy_symbol = dxy_symbol
        self._dxy_cache: Optional[Tuple[datetime, DxyReading]] = None

    def yields(self) -> Dict[str, Optional[YieldReading]]:
        return {"DGS10": self.fred.reading("DGS10"), "DFII10": self.fred.reading("DFII10")}

    def dxy(self) -> Optional[DxyReading]:
        if not self.dxy_symbol or self.market_data is None:
            return None
        now = datetime.now(timezone.utc)
        if self._dxy_cache and now - self._dxy_cache[0] < timedelta(minutes=15):
            return self._dxy_cache[1]
        try:
            from core.domain import Timeframe
            df = self.market_data.get_candles(self.dxy_symbol, Timeframe.H4, 60, raw_symbol=True)
            if df is None or len(df) < 40:
                return None
            close = df["close"]
            ema20 = close.ewm(span=20, adjust=False).mean()
            slope = (ema20.iloc[-1] - ema20.iloc[-6]) / ema20.iloc[-6]
            trend = 1 if slope > 0.0005 else -1 if slope < -0.0005 else 0
            chg = (close.iloc[-1] / close.iloc[-31] - 1) * 100 if len(close) > 31 else 0.0
            r = DxyReading(self.dxy_symbol, trend, round(float(chg), 2), df.index[-1].to_pydatetime())
            self._dxy_cache = (now, r)
            return r
        except Exception as exc:
            log.warning("DXY read failed (%s): %s", self.dxy_symbol, exc)
            return None
