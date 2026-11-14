"""
Settings fingerprint: a SHA-256 of the strategy-relevant settings, so every
captured signal can later be tied to the exact settings that produced it.

ALLOWLIST approach: only the fields listed below are included. Everything
else is excluded by default - secrets (passwords, tokens, API keys), account
identifiers (MT5 login/server, Telegram chat id), the database URL and all
filesystem paths. A new config field is therefore never included unless it
is added here on purpose.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import BASE_DIR

log = logging.getLogger(__name__)

FINGERPRINT_VERSION = 1

# section -> allowed keys (None = the whole section, which contains no secrets or paths)
ALLOWLIST: Dict[str, Optional[Tuple[str, ...]]] = {
    "instruments": None,
    "scoring": None,
    "risk": None,
    "analysis": None,
    "news": ("calendar_provider", "calendar_refresh_minutes", "high_impact_before_min", "high_impact_after_min",
             "medium_impact_before_min", "medium_impact_after_min", "block_on_medium", "gold_tier1_before_min",
             "gold_tier1_after_min", "alert_lead_minutes", "block_if_calendar_unavailable",
             "headline_poll_minutes"),
    "mt5": ("server_utc_offset", "server_time_mode", "require_demo", "symbol_suffix", "dxy_symbol"),
    "runtime": ("scan_interval_s", "tracker_interval_s", "stale_tick_seconds"),
    "data_provider": None,
}


def strategy_settings(cfg) -> Dict[str, Any]:
    """Allowlisted, JSON-serialisable settings. Built from Config.to_dict() (secrets already masked there)."""
    full = cfg.to_dict()
    out: Dict[str, Any] = {}
    for section, keys in ALLOWLIST.items():
        if section not in full:
            continue
        value = full[section]
        if keys is None or not isinstance(value, dict):
            out[section] = value
        else:
            out[section] = {k: value[k] for k in keys if k in value}
    return json.loads(json.dumps(out, sort_keys=True, default=str))     # canonical, plain types


def code_version(base_dir: Path = BASE_DIR) -> str:
    """Current git commit read from .git (no git process is started). 'unknown' if unavailable."""
    git = Path(base_dir) / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:40]
        ref = head.split(" ", 1)[1].strip()
        ref_file = git / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip()[:40]
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ", 1)[0][:40]
    except OSError:
        pass
    return "unknown"


def settings_fingerprint(cfg, base_dir: Path = BASE_DIR) -> Tuple[str, Dict[str, Any], str]:
    """(sha256 hex, allowlisted settings, code version). The hash covers settings + code version."""
    settings = strategy_settings(cfg)
    version = code_version(base_dir)
    payload = json.dumps({"fingerprint_version": FINGERPRINT_VERSION, "code_version": version,
                          "settings": settings}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), settings, version


def record_settings(db, cfg, base_dir: Path = BASE_DIR) -> str:
    """Store the current settings snapshot once per fingerprint (upsert of last_seen). Returns the fingerprint."""
    from database.learning_models import SettingsSnapshot
    from database.models import utcnow
    fp, settings, version = settings_fingerprint(cfg, base_dir)
    now = utcnow()
    with db.session() as s:
        row = s.get(SettingsSnapshot, fp)
        if row is None:
            s.add(SettingsSnapshot(fingerprint=fp, fingerprint_version=FINGERPRINT_VERSION, code_version=version,
                                   settings=settings, first_seen=now, last_seen=now))
            log.info("Settings fingerprint %s recorded (code %s)", fp[:12], version[:12])
        else:
            row.last_seen = now
    return fp
