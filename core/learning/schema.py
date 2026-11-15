"""
Schema version for the learning tables.

The project has no migration tool: Base.metadata.create_all only creates
missing tables and never alters existing ones. The rule is therefore:
learning tables are added, never altered. A row in `schema_versions`
records which version of those tables a database has, so a future change
that needs a real migration can detect where it starts.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

log = logging.getLogger(__name__)

LEARNING_COMPONENT = "learning"
LEARNING_SCHEMA_VERSION = 2          # 1: L1.1a tables, 2: + L1.1b snapshot tables (added, not altered)


def ensure_learning_schema(session_factory) -> int:
    """Record the learning schema version; returns the version stored in the database."""
    from database.learning_models import SchemaVersion
    from database.models import utcnow
    s = session_factory()
    try:
        row = s.scalar(select(SchemaVersion).where(SchemaVersion.component == LEARNING_COMPONENT))
        if row is None:
            s.add(SchemaVersion(component=LEARNING_COMPONENT, version=LEARNING_SCHEMA_VERSION, updated_at=utcnow()))
            s.commit()
            return LEARNING_SCHEMA_VERSION
        if row.version > LEARNING_SCHEMA_VERSION:
            log.warning("Database learning schema v%s is newer than this code (v%s); learning tables are "
                        "additive, continuing", row.version, LEARNING_SCHEMA_VERSION)
        elif row.version < LEARNING_SCHEMA_VERSION:
            row.version, row.updated_at = LEARNING_SCHEMA_VERSION, utcnow()
            s.commit()
        return row.version
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
