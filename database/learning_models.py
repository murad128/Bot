"""
Learning-engine tables (L1.1a). NEW tables only: no existing table or column
is changed. They are created by the existing Base.metadata.create_all call,
which only adds missing tables.

  schema_versions     version of the learning schema (migration guard)
  settings_snapshots  strategy-relevant settings (no secrets, no paths) by fingerprint
  calendar_events     archive of every economic-calendar event seen
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database.models import Base, utcnow


class SchemaVersion(Base):
    __tablename__ = "schema_versions"

    component: Mapped[str] = mapped_column(String(50), primary_key=True)
    version: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SettingsSnapshot(Base):
    __tablename__ = "settings_snapshots"

    fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)   # sha256 hex
    fingerprint_version: Mapped[int] = mapped_column(Integer)
    code_version: Mapped[str] = mapped_column(String(64))
    settings: Mapped[dict] = mapped_column(JSON)                            # allowlisted values only
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CalendarEventRecord(Base):
    __tablename__ = "calendar_events"

    event_key: Mapped[str] = mapped_column(String(300), primary_key=True)   # CalendarEvent.key
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    currency: Mapped[str] = mapped_column(String(10), index=True)
    impact: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(Text)
    forecast: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    previous: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    actual: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(30))
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_changed: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revisions: Mapped[int] = mapped_column(Integer, default=0)              # times a value changed
