"""
Financial news headlines.

* FinnhubNewsProvider - https://finnhub.io (free API key, FINNHUB_API_KEY).
* NullNewsProvider    - used when no key is configured.

Headlines are used ONLY for (a) Telegram alerts about potentially market-moving
events and (b) a "headline risk" caution flag. The bot does not derive a
trade direction from headlines: keyword sentiment is too unreliable to be used
as a directional signal, and inventing a fundamental bias would violate the
"never fabricate news" rule.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

RISK_KEYWORDS = (
    "war", "missile", "invasion", "attack", "sanction", "emergency", "default", "bank failure",
    "nuclear", "ceasefire", "escalat", "terror", "coup", "intervention", "surprise rate",
    "emergency meeting", "tariff",
)


@dataclass
class NewsItem:
    id: str
    time: datetime
    headline: str
    source: str
    url: str = ""
    summary: str = ""

    @property
    def risk_keywords(self) -> List[str]:
        text = f"{self.headline} {self.summary}".lower()
        return [k for k in RISK_KEYWORDS if k in text]


class NewsProvider(ABC):
    name = "abstract"
    available = False

    @abstractmethod
    def latest(self, since: Optional[datetime] = None) -> List[NewsItem]:
        ...


class NullNewsProvider(NewsProvider):
    name = "none"
    available = False

    def latest(self, since=None):
        return []


class FinnhubNewsProvider(NewsProvider):
    name = "finnhub"
    URL = "https://finnhub.io/api/v1/news"

    def __init__(self, api_key: str, categories=("forex", "general"), session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.categories = categories
        self.session = session or requests.Session()
        self.available = True
        self._next_allowed = 0.0

    def latest(self, since: Optional[datetime] = None) -> List[NewsItem]:
        if time.monotonic() < self._next_allowed:
            return []
        items: List[NewsItem] = []
        for cat in self.categories:
            resp = self.session.get(self.URL, params={"category": cat, "token": self.api_key}, timeout=15)
            if resp.status_code == 429:
                self._next_allowed = time.monotonic() + 120
                log.warning("Finnhub rate limit hit; backing off 120s")
                break
            resp.raise_for_status()
            for row in resp.json():
                ts = datetime.fromtimestamp(int(row.get("datetime", 0)), tz=timezone.utc)
                if since and ts <= since:
                    continue
                items.append(NewsItem(id=f"finnhub:{row.get('id')}", time=ts,
                                      headline=row.get("headline", ""), source=row.get("source", ""),
                                      url=row.get("url", ""), summary=row.get("summary", "")))
        items.sort(key=lambda x: x.time)
        return items


def build_news_provider(cfg) -> NewsProvider:
    if cfg.finnhub_api_key:
        return FinnhubNewsProvider(cfg.finnhub_api_key)
    return NullNewsProvider()
