"""
Economic calendar providers.

* ForexFactoryCalendar - the public weekly JSON export from ForexFactory
  (https://nfs.faireconomy.media/ff_calendar_thisweek.json). It is free and
  widely used but UNOFFICIAL: no SLA, rate-limited (cache it; do not poll
  more than every few minutes) and contains no "actual" values.
* CSVCalendar - a local CSV (time_utc,currency,impact,title[,forecast,previous,actual]);
  used for backtests with a historical calendar export or as a manual fallback.
* NullCalendar - explicitly "no calendar"; `available` is False so the
  fundamental module can block or warn instead of silently assuming no news.

Event times are always normalised to timezone-aware UTC.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

TIER1_KEYWORDS = (
    "non-farm", "nonfarm", "nfp", "cpi", "consumer price", "pce", "fomc", "federal funds",
    "fed chair", "powell", "interest rate decision", "rate statement", "unemployment rate",
    "gdp", "core retail sales", "monetary policy",
)


@dataclass
class CalendarEvent:
    time: datetime          # UTC
    currency: str
    impact: str             # High | Medium | Low | Holiday
    title: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    source: str = ""

    @property
    def is_high(self) -> bool:
        return self.impact.lower() == "high"

    @property
    def is_medium(self) -> bool:
        return self.impact.lower() == "medium"

    @property
    def is_tier1(self) -> bool:
        t = self.title.lower()
        return self.is_high and any(k in t for k in TIER1_KEYWORDS)

    @property
    def key(self) -> str:
        return f"{self.currency}|{self.title}|{self.time.isoformat()}"

    def to_json(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.isoformat()
        return d

    @staticmethod
    def from_json(d: dict) -> "CalendarEvent":
        d = dict(d)
        d["time"] = datetime.fromisoformat(d["time"])
        return CalendarEvent(**d)


def _parse_time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class CalendarProvider(ABC):
    name = "abstract"

    @abstractmethod
    def fetch(self) -> List[CalendarEvent]:
        """Return events; raise on failure."""


class ForexFactoryCalendar(CalendarProvider):
    name = "forexfactory"
    URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 15):
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch(self) -> List[CalendarEvent]:
        resp = self.session.get(self.URL, timeout=self.timeout,
                                headers={"User-Agent": "fx-signal-bot/1.0"})
        if resp.status_code == 429:
            raise RuntimeError("ForexFactory calendar rate limit (HTTP 429)")
        resp.raise_for_status()
        out = []
        for row in resp.json():
            try:
                out.append(CalendarEvent(
                    time=_parse_time(row["date"]), currency=row.get("country", "").upper(),
                    impact=row.get("impact", ""), title=row.get("title", ""),
                    forecast=row.get("forecast") or None, previous=row.get("previous") or None,
                    actual=row.get("actual") or None, source=self.name,
                ))
            except Exception as exc:  # skip malformed rows, never invent data
                log.debug("Skipping malformed calendar row %s: %s", row, exc)
        return out


class CSVCalendar(CalendarProvider):
    name = "csv"

    def __init__(self, path: str):
        self.path = Path(path)

    def fetch(self) -> List[CalendarEvent]:
        if not self.path.exists():
            raise FileNotFoundError(f"Calendar CSV not found: {self.path}")
        out = []
        with self.path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.append(CalendarEvent(
                    time=_parse_time(row["time_utc"]), currency=row["currency"].upper().strip(),
                    impact=row["impact"].strip().capitalize(), title=row["title"].strip(),
                    forecast=row.get("forecast") or None, previous=row.get("previous") or None,
                    actual=row.get("actual") or None, source=self.name,
                ))
        return out


class NullCalendar(CalendarProvider):
    name = "none"

    def fetch(self) -> List[CalendarEvent]:
        return []


class EconomicCalendar:
    """Caching facade with failure tracking. `available` reflects data freshness."""

    def __init__(self, provider: CalendarProvider, refresh_minutes: int = 60,
                 cache_path: Optional[str] = None, max_staleness_hours: float = 12.0):
        self.provider = provider
        self.refresh = timedelta(minutes=refresh_minutes)
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_staleness = timedelta(hours=max_staleness_hours)
        self.events: List[CalendarEvent] = []
        self.last_success: Optional[datetime] = None
        self.last_attempt: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self._backoff_until = 0.0
        self._load_cache()

    @property
    def available(self) -> bool:
        if isinstance(self.provider, NullCalendar):
            return False
        if self.last_success is None:
            return False
        return datetime.now(timezone.utc) - self.last_success <= self.max_staleness

    def refresh_if_needed(self, now: Optional[datetime] = None, force: bool = False) -> bool:
        now = now or datetime.now(timezone.utc)
        if not force and self.last_attempt and now - self.last_attempt < self.refresh:
            return False
        if time.monotonic() < self._backoff_until:
            return False
        self.last_attempt = now
        try:
            events = self.provider.fetch()
            self.events = sorted(events, key=lambda e: e.time)
            self.last_success = now
            self.last_error = None
            self._save_cache()
            log.info("Calendar refreshed from %s: %d events", self.provider.name, len(events))
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._backoff_until = time.monotonic() + 300
            log.warning("Calendar refresh failed (%s): %s", self.provider.name, exc)
            return False

    def events_between(self, start: datetime, end: datetime, currencies: Optional[List[str]] = None) -> List[CalendarEvent]:
        cur = set(c.upper() for c in currencies) if currencies else None
        return [e for e in self.events if start <= e.time <= end and (cur is None or e.currency in cur)]

    # --------------------------------------------------------------- cache
    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"fetched_at": self.last_success.isoformat() if self.last_success else None,
                       "provider": self.provider.name,
                       "events": [e.to_json() for e in self.events]}
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.cache_path)
        except Exception as exc:  # cache is best-effort
            log.debug("Calendar cache write failed: %s", exc)

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text())
            if payload.get("provider") != self.provider.name:
                return
            self.events = [CalendarEvent.from_json(e) for e in payload.get("events", [])]
            if payload.get("fetched_at"):
                self.last_success = datetime.fromisoformat(payload["fetched_at"])
        except Exception as exc:
            log.debug("Calendar cache read failed: %s", exc)


def build_calendar(cfg, cache_dir: str) -> EconomicCalendar:
    name = (cfg.calendar_provider or "none").lower()
    if name == "forexfactory":
        provider: CalendarProvider = ForexFactoryCalendar()
    elif name == "csv":
        provider = CSVCalendar(cfg.calendar_csv_path)
    else:
        provider = NullCalendar()
    return EconomicCalendar(provider, cfg.calendar_refresh_minutes,
                            cache_path=str(Path(cache_dir) / "calendar_cache.json"))
