"""
Learning-engine tables (L1.1a). NEW tables only: no existing table or column
is changed. They are created by the existing Base.metadata.create_all call,
which only adds missing tables.

  schema_versions     version of the learning schema (migration guard)
  settings_snapshots  strategy-relevant settings (no secrets, no paths) by fingerprint
  calendar_events     archive of every economic-calendar event seen
  signal_snapshots            (L1.1b) market / SMC / news context per live signal
  signal_indicator_snapshots  (L1.1b) last closed bar + indicators per signal and timeframe
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
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


# ---------------------------------------------------------------------------- L1.1b
class SignalSnapshot(Base):
    """Market / SMC / news context at the moment a live signal was created (one row per signal)."""
    __tablename__ = "signal_snapshots"

    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("signals.id"), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    broker_symbol: Mapped[str] = mapped_column(String(40))
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    feature_version: Mapped[int] = mapped_column(Integer)
    settings_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    capture_status: Mapped[str] = mapped_column(String(20))                # complete | partial
    missing: Mapped[list] = mapped_column(JSON, default=list)              # what could not be captured
    market: Mapped[dict] = mapped_column(JSON, default=dict)               # prices, spread, bar times
    smc: Mapped[dict] = mapped_column(JSON, default=dict)                  # per-timeframe SMC summary
    news: Mapped[dict] = mapped_column(JSON, default=dict)                 # calendar events +-24h
    capture_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class SignalIndicatorSnapshot(Base):
    """Last closed bar (OHLC + every existing indicator column) per timeframe, as the engine saw it."""
    __tablename__ = "signal_indicator_snapshots"
    __table_args__ = (UniqueConstraint("signal_id", "timeframe", name="uq_indicator_snapshot"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[int] = mapped_column(Integer, ForeignKey("signals.id"), index=True)
    timeframe: Mapped[str] = mapped_column(String(5))
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))    # open time of the last closed bar
    values: Mapped[dict] = mapped_column(JSON, default=dict)
