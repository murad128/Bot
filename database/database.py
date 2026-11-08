"""
Database access layer.

SQLite is the default (zero setup on a VPS). To migrate to PostgreSQL set
DATABASE_URL=postgresql+psycopg://user:pass@host/db and install `psycopg`;
the models use only portable column types.
"""
from __future__ import annotations

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, List, Optional

from sqlalchemy import create_engine, event, select, delete, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from database.models import (Base, KeyValue, NotificationLog, OutboxMessage, Recommendation,
                             SignalRecord)

log = logging.getLogger(__name__)


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite drops tzinfo; normalise everything to aware UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class Database:
    def __init__(self, url: str):
        self.url = url
        if url.startswith("sqlite:///"):
            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _pragma(dbapi_conn, _):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self.Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ------------------------------------------------------------------ signals
    def add_signal(self, rec: SignalRecord) -> Optional[SignalRecord]:
        """Insert; returns None if a signal with the same dedup key exists."""
        try:
            with self.session() as s:
                s.add(rec)
                s.flush()
                return rec
        except IntegrityError:
            log.info("Duplicate signal ignored: %s", rec.dedup_key)
            return None

    def save_signal(self, rec: SignalRecord) -> None:
        with self.session() as s:
            s.merge(rec)

    def get_signal(self, signal_id: int) -> Optional[SignalRecord]:
        with self.session() as s:
            return s.get(SignalRecord, signal_id)

    def open_signals(self, source: str = "live") -> List[SignalRecord]:
        with self.session() as s:
            q = select(SignalRecord).where(SignalRecord.source == source, SignalRecord.outcome == "OPEN")
            return list(s.scalars(q))

    def signals_between(self, start: datetime, end: datetime, source: str = "live") -> List[SignalRecord]:
        with self.session() as s:
            q = (select(SignalRecord)
                 .where(SignalRecord.source == source, SignalRecord.created_at >= start, SignalRecord.created_at < end)
                 .order_by(SignalRecord.created_at))
            return list(s.scalars(q))

    def closed_between(self, start: datetime, end: datetime, source: str = "live") -> List[SignalRecord]:
        with self.session() as s:
            q = (select(SignalRecord)
                 .where(SignalRecord.source == source, SignalRecord.closed_at >= start, SignalRecord.closed_at < end)
                 .order_by(SignalRecord.closed_at))
            return list(s.scalars(q))

    def all_closed(self, source: str = "live", limit: int = 5000) -> List[SignalRecord]:
        with self.session() as s:
            q = (select(SignalRecord).where(SignalRecord.source == source, SignalRecord.outcome != "OPEN")
                 .order_by(SignalRecord.closed_at.desc()).limit(limit))
            return list(reversed(list(s.scalars(q))))

    def last_signal_time(self, symbol: str, source: str = "live") -> Optional[datetime]:
        with self.session() as s:
            v = s.scalar(select(func.max(SignalRecord.created_at))
                         .where(SignalRecord.symbol == symbol, SignalRecord.source == source))
            return _aware(v)

    def consecutive_losses(self, source: str = "live", since: Optional[datetime] = None) -> int:
        """Count of most recent resolved signals that were losses (cancelled ignored)."""
        with self.session() as s:
            q = (select(SignalRecord.outcome)
                 .where(SignalRecord.source == source, SignalRecord.outcome.in_(["WIN", "LOSS", "BREAKEVEN"]))
                 .order_by(SignalRecord.closed_at.desc()).limit(50))
            if since is not None:
                q = q.where(SignalRecord.closed_at >= since)
            n = 0
            for outcome in s.scalars(q):
                if outcome != "LOSS":
                    break
                n += 1
            return n

    # ------------------------------------------------------------------ notifications
    def notification_sent(self, key: str) -> bool:
        with self.session() as s:
            return s.scalar(select(NotificationLog.id).where(NotificationLog.key == key)) is not None

    def mark_notification(self, key: str, kind: str) -> bool:
        """Returns False if the key already existed."""
        try:
            with self.session() as s:
                s.add(NotificationLog(key=key, kind=kind))
            return True
        except IntegrityError:
            return False

    def outbox_add(self, kind: str, text: str, key: Optional[str], error: str) -> None:
        with self.session() as s:
            s.add(OutboxMessage(kind=kind, text=text, key=key, attempts=1, last_error=error))

    def outbox_pending(self, max_attempts: int = 20) -> List[OutboxMessage]:
        with self.session() as s:
            return list(s.scalars(select(OutboxMessage).where(OutboxMessage.attempts < max_attempts)
                                  .order_by(OutboxMessage.id)))

    def outbox_done(self, msg_id: int) -> None:
        with self.session() as s:
            s.execute(delete(OutboxMessage).where(OutboxMessage.id == msg_id))

    def outbox_failed(self, msg_id: int, error: str) -> None:
        with self.session() as s:
            m = s.get(OutboxMessage, msg_id)
            if m:
                m.attempts += 1
                m.last_error = error

    # ------------------------------------------------------------------ kv + recommendations
    def kv_get(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            row = s.get(KeyValue, key)
            return row.value.get("v", default) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        with self.session() as s:
            s.merge(KeyValue(key=key, value={"v": value}))

    def add_recommendation(self, category: str, text: str, evidence: dict) -> bool:
        try:
            with self.session() as s:
                s.add(Recommendation(category=category, text=text, evidence=evidence))
            return True
        except IntegrityError:
            return False

    def recommendations(self, status: Optional[str] = None) -> List[Recommendation]:
        with self.session() as s:
            q = select(Recommendation).order_by(Recommendation.created_at.desc())
            if status:
                q = q.where(Recommendation.status == status)
            return list(s.scalars(q))

    def set_recommendation_status(self, reco_id: int, status: str) -> bool:
        with self.session() as s:
            r = s.get(Recommendation, reco_id)
            if not r:
                return False
            r.status = status
            return True

    # ------------------------------------------------------------------ maintenance
    def backup(self, backup_dir: str, keep: int = 14) -> Optional[Path]:
        """Online, consistent SQLite backup with rotation. No-op for other engines."""
        if not self.url.startswith("sqlite:///"):
            log.info("Backup skipped: use pg_dump for non-SQLite databases")
            return None
        src = Path(self.url.replace("sqlite:///", ""))
        if not src.exists():
            return None
        out_dir = Path(backup_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{src.stem}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.db"
        with sqlite3.connect(src) as con, sqlite3.connect(dest) as bak:
            con.backup(bak)
        backups = sorted(out_dir.glob(f"{src.stem}_*.db"))
        for old in backups[:-keep]:
            old.unlink(missing_ok=True)
        log.info("Database backup written to %s", dest)
        return dest
