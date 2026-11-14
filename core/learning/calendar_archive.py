"""
Economic-calendar archive.

Called only after EconomicCalendar.refresh_if_needed() returned True (a
successful fetch), with the events that fetch produced. Events are upserted
by CalendarEvent.key (currency | title | time):
  new key          -> inserted (first_seen = last_seen = last_changed = now)
  known key        -> last_seen updated
  changed values   -> impact/forecast/previous/actual/source updated,
                      last_changed = now, revisions += 1
Nothing is ever deleted: an event that disappears from the feed (e.g. a
rescheduled release, which gets a new key) stays with its last_seen time.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

from sqlalchemy import select

log = logging.getLogger(__name__)

MUTABLE_FIELDS = ("impact", "forecast", "previous", "actual", "source")


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def archive_events(db, events: Iterable, now: datetime) -> Dict[str, int]:
    """Upsert events in one transaction. Returns {'inserted', 'updated', 'unchanged'}."""
    from database.learning_models import CalendarEventRecord
    now = _aware(now)
    batch: Dict[str, object] = {}
    for e in events:
        batch[e.key] = e                                    # same key twice in one feed -> last wins, once
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    if not batch:
        return stats
    with db.session() as s:
        keys = list(batch)
        existing: Dict[str, CalendarEventRecord] = {}
        for i in range(0, len(keys), 500):                  # stay below SQLite's parameter limit
            chunk = keys[i:i + 500]
            for row in s.scalars(select(CalendarEventRecord).where(CalendarEventRecord.event_key.in_(chunk))):
                existing[row.event_key] = row
        for key, e in batch.items():
            values = {"impact": e.impact, "forecast": e.forecast, "previous": e.previous,
                      "actual": e.actual, "source": e.source}
            row = existing.get(key)
            if row is None:
                s.add(CalendarEventRecord(event_key=key, event_time=_aware(e.time), currency=e.currency,
                                          title=e.title, first_seen=now, last_seen=now, last_changed=now,
                                          revisions=0, **values))
                stats["inserted"] += 1
                continue
            row.last_seen = now
            changed = {k: v for k, v in values.items() if getattr(row, k) != v}
            if changed:
                for k, v in changed.items():
                    setattr(row, k, v)
                row.last_changed = now
                row.revisions = (row.revisions or 0) + 1
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
    log.info("Calendar archive: %(inserted)d new, %(updated)d changed, %(unchanged)d unchanged", stats)
    return stats
