"""
SQLite backup, compression, verification and restore.

Backup (create_backup)
1. Consistent online copy with the SQLite backup API. The live database is
   opened READ-ONLY, so taking a backup never writes to it (a read-only
   connection never checkpoints the WAL into the main file). Committed data
   still in the WAL is included. If read-only access is impossible, a normal
   connection is used and a warning is logged and returned.
2. The copy is written to a temporary file, switched to rollback-journal mode
   (no -wal/-shm side files), checked with PRAGMA integrity_check and
   per-table record counts, then renamed into place.
3. Compression: gzip to a temporary file, decompressed again and compared
   (SHA-256) with the verified copy. Only then is the .db.gz kept and the
   uncompressed copy removed; otherwise the verified .db is kept.
4. Every backup file gets its OWN manifest: <backup file>.json (e.g.
   bot_20261114_210000.db.gz.json). Older manifests named <stem>_<ts>.json are
   still read, but only for the file they name.
5. Maintenance at each backup: hidden temporary files (".*.partial") older
   than 24 hours are removed; a stale uncompressed duplicate of a verified
   .db.gz is removed (retried later if Windows refuses the deletion).
6. Rotation keeps the newest `keep` backups. A backup and its uncompressed
   duplicate count as one; the backup just written is never deleted.

Restore (restore_backup)
* Manifest required (checksum, integrity, per-table counts) unless
  allow_no_manifest=True (old backups without a manifest).
* Decompresses to a temporary file next to the target and verifies it first.
* A file that contains a bot database (a 'signals' table) is never
  overwritten by a plain restore, whatever DATABASE_URL says. Only
  replace_live may replace the live database.
* replace_live (the configured live database):
    1. bot instance lock taken (refused if the bot runs);
    2. Linux: refused if another process has the file open (best effort);
    3. exclusive SQLite lock with no waiting - refused if any other SQLite
       connection is open; NOTHING is changed when refused;
    4. WAL checkpointed into the main file under that lock (must not be busy);
    5. verified safety backup taken from the locked connection;
    6. connection closed; the WAL must now be empty;
    7. only then: live file -> .pre-restore-<time>, its -wal/-shm -> aside,
       restored file -> live name; restored database verified;
    8. any failure in 7 rolls back: the restored file is moved away and the
       original database and its -wal/-shm are moved back.
  The pre-restore copy and the safety backup are kept (never deleted).

Standard library only. Windows 11 / Windows Server 2022, Python 3.12+.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

BACKUP_RE = re.compile(r"^(?P<stem>.+)_(?P<ts>\d{8}_\d{6}(?:_\d+)?)\.db(?P<gz>\.gz)?$")
PARTIAL_RE = re.compile(r"^\..+\.partial(?:-wal|-shm|-journal)?$")
STALE_PARTIAL_SECONDS = 24 * 3600
CHUNK = 1024 * 1024
SQLITE_HEADER = b"SQLite format 3\x00"


class BackupError(RuntimeError):
    pass


@dataclass
class BackupResult:
    path: Path                       # the kept backup (.db.gz, or .db if compression failed)
    manifest: Path
    compressed: bool
    db_bytes: int
    stored_bytes: int
    sha256: str
    counts: Dict[str, int]
    integrity: str
    warnings: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.db_bytes / self.stored_bytes if self.stored_bytes else 0.0


@dataclass
class RestoreResult:
    target: Path
    source: Path
    sha256: str
    counts: Dict[str, int]
    integrity: str
    safety_backup: Optional[Path] = None
    pre_restore_copy: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_gz(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _ro_uri(path: Path) -> str:
    """Read-only SQLite URI; percent-encodes the path so spaces, '#', '%' or '?' in Windows folder names work."""
    from urllib.parse import quote
    p = Path(path).resolve().as_posix()                     # C:/Users/... on Windows
    if not p.startswith("/"):
        p = "/" + p                                         # file:///C:/... form
    return f"file://{quote(p, safe='/:')}?mode=ro"


def inspect_db(path: Path) -> Dict[str, object]:
    """integrity_check result and record count per table, opened read-only."""
    with closing(sqlite3.connect(_ro_uri(path), uri=True, timeout=30)) as con:
        integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        counts = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    return {"integrity": integrity, "counts": counts}


def _unlink_quietly(p: Path) -> bool:
    try:
        p.unlink(missing_ok=True)
        return True
    except OSError as exc:                      # e.g. file still open on Windows
        log.warning("Could not delete %s: %s", p, exc)
        return False


def _unlink_with_sidecars(p: Path) -> None:
    for q in (p, p.with_name(p.name + "-wal"), p.with_name(p.name + "-shm"), p.with_name(p.name + "-journal")):
        _unlink_quietly(q)


def _make_standalone(path: Path) -> None:
    """
    Switch a COPY to rollback-journal mode. Copies inherit WAL mode from the
    live database; in WAL mode even a read-only open creates -shm side files
    (and Windows file locks). Only the header's journal-mode setting changes;
    the bot switches a restored live database back to WAL when it connects.
    """
    with closing(sqlite3.connect(str(path))) as con:
        mode = con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    for side in ("-wal", "-shm"):
        _unlink_quietly(path.with_name(path.name + side))
    if str(mode).lower() != "delete":
        raise BackupError(f"could not switch copy to rollback-journal mode (got {mode})")


def _open_source(src: Path) -> Tuple[sqlite3.Connection, bool]:
    """
    Open the live database READ-ONLY (never checkpoints, never writes).
    Returns (connection, fallback_used). The writable fallback is only used if
    read-only access is impossible; closing it as the last connection may
    checkpoint the WAL into the main file, so a warning is always emitted.
    """
    try:
        con = sqlite3.connect(_ro_uri(src), uri=True, timeout=30)
        con.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return con, False
    except sqlite3.Error as exc:
        log.warning("Read-only open of %s failed (%s): using a WRITABLE fallback connection for the backup; "
                    "closing it may checkpoint the WAL into the database file (data-safe, but not read-only)",
                    src, exc)
        return sqlite3.connect(str(src), timeout=30), True


def _timestamp_name(stem: str, out_dir: Path) -> str:
    base = f"{stem}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    name, i = base, 1
    while (out_dir / f"{name}.db.gz").exists() or (out_dir / f"{name}.db").exists():
        name = f"{base}_{i}"
        i += 1
    return name


def _manifest_path(backup: Path) -> Path:
    """Each backup file has its own manifest: <file>.json."""
    return backup.with_name(backup.name + ".json")


def _legacy_manifest_path(backup: Path) -> Path:
    name = backup.name[:-3] if backup.name.endswith(".gz") else backup.name
    return backup.with_name(name[:-3] + ".json")            # <stem>_<ts>.json (first L1.0 draft)


def read_manifest(backup: Path) -> Optional[dict]:
    """The backup's own manifest, or a legacy manifest only if it names this exact file."""
    own = _manifest_path(backup)
    if own.exists():
        return json.loads(own.read_text(encoding="utf-8"))
    legacy = _legacy_manifest_path(backup)
    if legacy.exists():
        data = json.loads(legacy.read_text(encoding="utf-8"))
        if data.get("backup_file") == backup.name:
            return data
    return None


def _delete_backup_file(p: Path) -> bool:
    ok = _unlink_quietly(p)
    if ok:
        _unlink_quietly(_manifest_path(p))
        legacy = _legacy_manifest_path(p)
        if legacy.exists():
            try:
                if json.loads(legacy.read_text(encoding="utf-8")).get("backup_file") == p.name:
                    _unlink_quietly(legacy)
            except (OSError, ValueError):
                pass
    return ok


def _groups(backup_dir: Path, stem: Optional[str]) -> List[Tuple[str, Path, List[Path]]]:
    """[(timestamp, primary, duplicates)] oldest -> newest; a .db.gz is primary over a .db of the same name."""
    by_name: Dict[Tuple[str, str], List[Path]] = {}
    if not backup_dir.exists():
        return []
    for p in backup_dir.iterdir():
        m = BACKUP_RE.match(p.name)
        if m and p.is_file() and (stem is None or m.group("stem") == stem):
            by_name.setdefault((m.group("ts"), m.group("stem")), []).append(p)
    out = []
    for (ts, _), files in sorted(by_name.items()):
        files.sort(key=lambda q: (not q.name.endswith(".gz"), q.name))
        out.append((ts, files[0], files[1:]))
    return out


def list_backups(backup_dir: str, stem: Optional[str] = None) -> List[Path]:
    """Backups sorted oldest -> newest (one entry per backup; .db.gz preferred over its .db duplicate)."""
    return [primary for _, primary, _ in _groups(Path(backup_dir), stem)]


def cleanup_stale_partials(directory: Path, max_age_seconds: float = STALE_PARTIAL_SECONDS,
                           now: Optional[float] = None) -> List[Path]:
    """Remove hidden temporary files (.*.partial[-wal|-shm|-journal]) older than max_age. Never a backup."""
    removed: List[Path] = []
    if not directory.exists():
        return removed
    now = time.time() if now is None else now
    for p in directory.iterdir():
        if not p.is_file() or not PARTIAL_RE.match(p.name) or BACKUP_RE.match(p.name):
            continue
        try:
            if now - p.stat().st_mtime >= max_age_seconds and _unlink_quietly(p):
                removed.append(p)
        except OSError:
            continue
    if removed:
        log.info("Removed %d stale temporary file(s) in %s", len(removed), directory)
    return removed


def _cleanup_duplicates(backup_dir: Path, stem: str) -> List[str]:
    """Remove an uncompressed .db left next to its verified .db.gz (a deletion Windows refused earlier)."""
    warnings = []
    for _, primary, dups in _groups(backup_dir, stem):
        if not primary.name.endswith(".gz"):
            continue
        m = read_manifest(primary)
        if not (m and m.get("compressed")):
            continue
        for d in dups:
            if not _delete_backup_file(d):
                warnings.append(f"could not delete stale uncompressed duplicate {d.name} (will retry)")
    return warnings


# ---------------------------------------------------------------------------
# backup
# ---------------------------------------------------------------------------
def create_backup(db_path: str, backup_dir: str, keep: int = 14, compress: bool = True,
                  _source_con: Optional[sqlite3.Connection] = None) -> BackupResult:
    src = Path(db_path)
    if not src.exists():
        raise BackupError(f"database not found: {src}")
    out_dir = Path(backup_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []
    cleanup_stale_partials(out_dir)
    warnings += _cleanup_duplicates(out_dir, src.stem)

    name = _timestamp_name(src.stem, out_dir)
    tmp_db = out_dir / f".{name}.db.partial"
    final_db = out_dir / f"{name}.db"
    final_gz = out_dir / f"{name}.db.gz"
    tmp_gz = out_dir / f".{name}.db.gz.partial"

    # 1. consistent online copy
    try:
        if _source_con is not None:
            with closing(sqlite3.connect(str(tmp_db))) as bak:
                _source_con.backup(bak)
        else:
            con, fallback = _open_source(src)
            if fallback:
                warnings.append("writable fallback connection used for the source (read-only open failed)")
            with closing(con), closing(sqlite3.connect(str(tmp_db))) as bak:
                con.backup(bak)
        _make_standalone(tmp_db)
        # 2. verify the copy before it is trusted
        info = inspect_db(tmp_db)
        if info["integrity"] != "ok":
            raise BackupError(f"backup copy failed integrity_check: {info['integrity']}")
        sha = _sha256(tmp_db)
        db_bytes = tmp_db.stat().st_size
        os.replace(tmp_db, final_db)
    except BaseException:                     # incl. Ctrl+C: never leave a half-written copy behind
        _unlink_with_sidecars(tmp_db)
        raise

    kept, stored, compressed = final_db, db_bytes, False
    # 3. compression, verified against the uncompressed copy before that copy is removed
    if compress:
        try:
            with open(final_db, "rb") as fin, gzip.open(tmp_gz, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout, CHUNK)
            if _sha256_gz(tmp_gz) != sha:
                raise BackupError("compressed backup does not match the verified copy")
            os.replace(tmp_gz, final_gz)
            kept, stored, compressed = final_gz, final_gz.stat().st_size, True
        except KeyboardInterrupt:
            _unlink_quietly(tmp_gz)               # the verified uncompressed .db stays
            raise
        except Exception as exc:
            _unlink_quietly(tmp_gz)
            warnings.append(f"compression failed, kept uncompressed backup: {exc}")
            log.warning("Backup compression failed (%s); uncompressed backup kept at %s", exc, final_db)

    # 4. the kept file's own manifest (written before the uncompressed duplicate is removed)
    manifest = {
        "format": 2, "source_name": src.name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "backup_file": kept.name, "compressed": compressed, "sha256_db": sha, "db_bytes": db_bytes,
        "stored_bytes": stored, "integrity": info["integrity"], "counts": info["counts"],
        "sqlite_version": sqlite3.sqlite_version,
    }
    mpath = _manifest_path(kept)
    tmp_m = mpath.with_name("." + mpath.name + ".partial")
    tmp_m.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_m, mpath)
    if compressed and not _unlink_quietly(final_db):
        warnings.append(f"could not delete uncompressed copy {final_db.name} yet (will retry at next backup)")

    # 5. rotation over backups (a .db.gz and its stale .db count as one); never the backup just written
    groups = _groups(out_dir, src.stem)
    for _, primary, dups in groups[:-max(1, keep)]:
        if primary == kept or kept in dups:
            continue
        for f in [primary] + dups:
            _delete_backup_file(f)
    log.info("Database backup written to %s (%d -> %d bytes, %s)", kept, db_bytes, stored,
             "compressed" if compressed else "uncompressed")
    return BackupResult(kept, mpath, compressed, db_bytes, stored, sha, info["counts"], info["integrity"], warnings)


# ---------------------------------------------------------------------------
# verify / restore
# ---------------------------------------------------------------------------
def _materialize(backup: Path, dest_tmp: Path) -> str:
    """Write the (decompressed) backup to dest_tmp; return its SHA-256."""
    if backup.name.endswith(".gz"):
        try:
            with gzip.open(backup, "rb") as fin, open(dest_tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, CHUNK)
        except (OSError, EOFError, zlib.error) as exc:
            raise BackupError(f"backup file is damaged (gzip): {exc}") from exc
    else:
        shutil.copyfile(backup, dest_tmp)
    return _sha256(dest_tmp)


def _check(backup: Path, restored: Path, sha: str, manifest: Optional[dict]) -> Dict[str, object]:
    if manifest and manifest.get("sha256_db") != sha:
        raise BackupError("SHA-256 does not match the manifest: backup file was changed or damaged")
    try:
        _make_standalone(restored)          # on the temporary copy only (SHA already taken)
        info = inspect_db(restored)
    except sqlite3.DatabaseError as exc:
        raise BackupError(f"backup is not a valid SQLite database: {exc}") from exc
    if info["integrity"] != "ok":
        raise BackupError(f"restored database failed integrity_check: {info['integrity']}")
    if manifest and manifest.get("counts") != info["counts"]:
        raise BackupError(f"record counts differ from the manifest: {info['counts']} vs {manifest.get('counts')}")
    return {"sha256": sha, "integrity": info["integrity"], "counts": info["counts"], "manifest": bool(manifest)}


def verify_backup(backup_path: str, work_dir: Optional[str] = None) -> Dict[str, object]:
    """Decompress to a temporary file and check SHA-256, integrity and counts. Nothing else is touched."""
    backup = Path(backup_path)
    if not backup.exists():
        raise BackupError(f"backup not found: {backup}")
    wd = Path(work_dir) if work_dir else backup.parent
    tmp = wd / f".verify_{os.getpid()}_{backup.name}.partial"
    try:
        sha = _materialize(backup, tmp)
        return _check(backup, tmp, sha, read_manifest(backup))
    finally:
        _unlink_with_sidecars(tmp)


def is_bot_database(path: Path) -> bool:
    """
    True if the file is (or may be) a bot database: an SQLite file with a
    'signals' table, or an SQLite file that cannot be inspected (protected by
    default). Read-only; never modifies the file.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    try:
        with open(p, "rb") as fh:
            if fh.read(16) != SQLITE_HEADER:
                return False
    except OSError:
        return True
    try:
        with closing(sqlite3.connect(_ro_uri(p), uri=True, timeout=5)) as con:
            row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signals'").fetchone()
        return row is not None
    except sqlite3.Error:
        return True


def _same_file(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def _processes_holding(path: Path) -> List[int]:
    """Linux only (best effort): other PIDs with the file open. Windows: the OS refuses the rename instead."""
    if not sys.platform.startswith("linux"):
        return []
    real = os.path.realpath(path)
    names = {real, real + "-wal", real + "-shm"}
    pids = []
    me = os.getpid()
    proc = Path("/proc")
    for d in proc.iterdir() if proc.exists() else []:
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            for fd in (d / "fd").iterdir():
                try:
                    if os.readlink(fd) in names:
                        pids.append(int(d.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def _exclusive_connection(target: Path) -> sqlite3.Connection:
    """
    Exclusive lock on the live database without waiting. In WAL mode every
    connection that has used the database holds a shared lock, so this fails
    if any other connection (idle reader, writer) is open. Nothing is written.
    """
    con = sqlite3.connect(str(target), timeout=0, isolation_level=None)
    try:
        con.execute("PRAGMA locking_mode=EXCLUSIVE")
        con.execute("BEGIN EXCLUSIVE")
        con.execute("ROLLBACK")
        return con
    except sqlite3.OperationalError as exc:
        con.close()
        raise BackupError(f"another SQLite connection is using the live database ({exc}); "
                          "close all programs using it (bot, report/scan commands, DB browsers)") from exc


def _cleanup_restore_partials(target: Path) -> None:
    """Stale temporary restore files for this target (older than 24h) from an interrupted earlier run."""
    pat = re.compile(r"^\." + re.escape(target.name) + r"\.\d+\.restore\.partial(?:-wal|-shm|-journal)?$")
    now = time.time()
    for p in target.parent.iterdir() if target.parent.exists() else []:
        try:
            if pat.match(p.name) and now - p.stat().st_mtime >= STALE_PARTIAL_SECONDS:
                _unlink_quietly(p)
        except OSError:
            continue


def restore_backup(backup_path: str, target_path: str, live_db_path: Optional[str] = None,
                   overwrite: bool = False, replace_live: bool = False, lock_path: Optional[str] = None,
                   safety_backup_dir: Optional[str] = None, allow_no_manifest: bool = False,
                   _replace: Callable = None) -> RestoreResult:
    """
    Verify a backup and restore it. `_replace` is os.replace (injectable only
    so tests can simulate Windows file-lock and rename failures).
    """
    rename = _replace or os.replace
    backup = Path(backup_path)
    target = Path(target_path)
    if not backup.exists():
        raise BackupError(f"backup not found: {backup}")
    manifest = read_manifest(backup)
    warnings: List[str] = []
    if manifest is None:
        if not allow_no_manifest:
            raise BackupError("backup has no manifest (no checksum/record-count check possible); "
                              "re-run with --allow-no-manifest to restore it on integrity_check only")
        warnings.append("restored WITHOUT a manifest: only gzip CRC and integrity_check were verified")
    is_live = live_db_path is not None and _same_file(target, Path(live_db_path))
    if is_live and not replace_live:
        raise BackupError("target is the live database; use --replace-live (bot must be stopped)")
    if replace_live and not is_live:
        raise BackupError("--replace-live given but the target is not the configured live database")
    if not is_live and target.exists():
        if is_bot_database(target):
            raise BackupError(f"target {target} is a bot database: it is never overwritten by a plain restore "
                              "(use --replace-live with the bot's own configuration)")
        if not overwrite:
            raise BackupError(f"target exists: {target} (use --overwrite for a non-database file)")

    lock = None
    if is_live:
        from core.instance_lock import AlreadyRunning, InstanceLock
        if lock_path is None:
            raise BackupError("lock path required to replace the live database")
        try:
            lock = InstanceLock(lock_path).acquire()          # proves the bot is not running
        except AlreadyRunning as exc:
            raise BackupError("the bot is running - stop it before replacing the live database") from exc
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_restore_partials(target)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.restore.partial")
        try:
            sha = _materialize(backup, tmp)
            checked = _check(backup, tmp, sha, manifest)
        except BaseException:
            _unlink_with_sidecars(tmp)
            raise
        if not is_live:
            try:
                rename(tmp, target)
            except BaseException:
                _unlink_with_sidecars(tmp)
                raise
            log.info("Restored %s -> %s (%s)", backup, target, checked["counts"])
            return RestoreResult(target, backup, checked["sha256"], checked["counts"], checked["integrity"],
                                 warnings=warnings)
        try:
            safety, aside = _replace_live(tmp, target, checked, safety_backup_dir, rename)
        except BaseException:
            _unlink_with_sidecars(tmp)
            raise
        log.info("Restored %s -> LIVE %s (%s); previous database kept as %s and %s",
                 backup, target, checked["counts"], aside, safety)
        return RestoreResult(target, backup, checked["sha256"], checked["counts"], checked["integrity"],
                             safety, aside, warnings)
    finally:
        if lock is not None:
            lock.release()


def _replace_live(tmp: Path, target: Path, checked: dict, safety_backup_dir: Optional[str],
                  rename: Callable) -> Tuple[Optional[Path], Optional[Path]]:
    if not target.exists():
        rename(tmp, target)
        return None, None
    busy = _processes_holding(target)
    if busy:
        raise BackupError(f"other process(es) {busy} have the live database open - nothing was changed")
    # exclusive lock: refuses (changing nothing) if any other SQLite connection is open
    con = _exclusive_connection(target)
    try:
        ck = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if ck is not None and ck[0] != 0:
            raise BackupError(f"WAL checkpoint was blocked {tuple(ck)} - another connection is active")
        safety = create_backup(str(target), safety_backup_dir or str(target.parent / "backups"),
                               keep=10 ** 6, compress=True, _source_con=con).path
    finally:
        con.close()                                       # Windows: no own handle may stay open
    wal = target.with_name(target.name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise BackupError("the live WAL file is not empty after the checkpoint - a writer appeared; "
                          "nothing was replaced")
    if _processes_holding(target):
        raise BackupError("another process opened the live database meanwhile - nothing was replaced")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    aside = target.with_name(f"{target.name}.pre-restore-{stamp}")
    done: List[Tuple[Path, Path]] = []                    # (moved_to, original_location)
    try:
        rename(target, aside)                             # Windows: fails here if any process has it open
        done.append((aside, target))
        for suffix in ("-wal", "-shm"):
            side = target.with_name(target.name + suffix)
            if side.exists():
                side_aside = aside.with_name(aside.name + suffix)
                rename(side, side_aside)
                done.append((side_aside, side))
        rename(tmp, target)
        done.append((target, tmp))                        # undo = move the restored file back to tmp
        post = inspect_db(target)
        if post["integrity"] != "ok" or post["counts"] != checked["counts"]:
            raise BackupError(f"restored live database failed verification: {post}")
    except BaseException as exc:                          # incl. Ctrl+C: always roll back
        failures = []
        for moved_to, original in reversed(done):
            try:
                os.replace(moved_to, original)
            except OSError as rexc:
                failures.append(f"{moved_to} -> {original}: {rexc}")
        if failures:
            raise BackupError("CRITICAL: restore failed AND rollback was incomplete. Do not start the bot. "
                              f"Move these files back manually: {failures}. Safety backup: {safety}") from exc
        if isinstance(exc, KeyboardInterrupt):
            raise                                         # rolled back; let Ctrl+C end the program
        raise BackupError(f"restore failed and was rolled back; the live database is unchanged ({exc})") from exc
    return safety, aside
