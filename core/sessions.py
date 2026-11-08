"""
Trading-session detection based on local exchange hours (DST-correct via zoneinfo).

Sessions (local time):
  Sydney   07:00-16:00 Australia/Sydney
  Tokyo    09:00-18:00 Asia/Tokyo
  London   08:00-17:00 Europe/London
  New York 08:00-17:00 America/New_York
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import List
from zoneinfo import ZoneInfo

SESSIONS = {
    "Sydney": (ZoneInfo("Australia/Sydney"), time(7, 0), time(16, 0)),
    "Tokyo": (ZoneInfo("Asia/Tokyo"), time(9, 0), time(18, 0)),
    "London": (ZoneInfo("Europe/London"), time(8, 0), time(17, 0)),
    "New York": (ZoneInfo("America/New_York"), time(8, 0), time(17, 0)),
}

_NY = ZoneInfo("America/New_York")


def _as_utc(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def active_sessions(now: datetime) -> List[str]:
    now = _as_utc(now)
    out = []
    for name, (tz, start, end) in SESSIONS.items():
        local = now.astimezone(tz)
        if local.weekday() >= 5:
            continue
        if start <= local.time() < end:
            out.append(name)
    return out


def session_label(now: datetime) -> str:
    """Human label, e.g. 'London / New York overlap'."""
    s = active_sessions(now)
    if not s:
        return "Off-session"
    if "London" in s and "New York" in s:
        return "London / New York overlap"
    priority = ["New York", "London", "Tokyo", "Sydney"]
    for p in priority:
        if p in s:
            return p
    return s[0]


def primary_session(now: datetime) -> str:
    """Single bucket used for statistics."""
    s = active_sessions(now)
    for p in ["New York", "London", "Tokyo", "Sydney"]:
        if p in s:
            return p
    return "Off-session"


def is_fx_market_open(now: datetime) -> bool:
    """FX trades from Sunday 17:00 to Friday 17:00 New York time."""
    local = _as_utc(now).astimezone(_NY)
    wd = local.weekday()  # Mon=0 .. Sun=6
    if wd == 5:
        return False
    if wd == 6:
        return local.time() >= time(17, 0)
    if wd == 4:
        return local.time() < time(17, 0)
    return True


def is_session_allowed(now: datetime, allowed: List[str]) -> bool:
    if not allowed:
        return True
    return any(s in allowed for s in active_sessions(now))


def minutes_since_week_open(now: datetime) -> float:
    """Minutes since the Sunday 17:00 New York open (inf if the market is closed)."""
    local = _as_utc(now).astimezone(_NY)
    if not is_fx_market_open(now):
        return float("inf")
    days_back = (local.weekday() - 6) % 7          # back to Sunday
    opened = (local - timedelta(days=days_back)).replace(hour=17, minute=0, second=0, microsecond=0)
    if opened > local:
        opened -= timedelta(days=7)
    return (local - opened).total_seconds() / 60.0


def minutes_to_week_close(now: datetime) -> float:
    """Minutes until the Friday 17:00 New York close (inf if the market is closed)."""
    local = _as_utc(now).astimezone(_NY)
    if not is_fx_market_open(now):
        return float("inf")
    days_ahead = (4 - local.weekday()) % 7
    close = (local + timedelta(days=days_ahead)).replace(hour=17, minute=0, second=0, microsecond=0)
    return (close - local).total_seconds() / 60.0
