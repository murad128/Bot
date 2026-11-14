"""L1.0: verified, compressed SQLite backups and safe restore (temporary databases only)."""
import gzip
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

import database.backup as bk
from core.signal_tracker import SignalTracker
from database.backup import BackupError, create_backup, inspect_db, list_backups, restore_backup, verify_backup
from database.database import Database
from tests.test_daily_report import U, _bar, _xau_sell


def _populate(cfg, n=60):
    db = Database(cfg.database_url)
    tr = SignalTracker(db, cfg.risk)
    for i in range(n):
        t = U("2026-01-05 08:00") + timedelta(hours=7 * i)
        rec = _xau_sell(db, cfg, t)
        tr.update_with_bars(rec, _bar(t + timedelta(hours=1), 3995, 3996, 3991, 3992), 0.25)
    db.mark_notification("k1", "system")
    db.kv_set("last_daily_report", "2026-01-05")
    return db, Path(cfg.database_url.replace("sqlite:///", ""))


def _content_hash(path):
    with closing(sqlite3.connect(bk._ro_uri(Path(path)), uri=True)) as con:
        tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
        data = {t: con.execute(f'select * from "{t}" order by 1').fetchall() for t in tabs}
    return hashlib.sha256(repr(data).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


@pytest.fixture
def live(cfg):
    db, path = _populate(cfg)
    return db, path


# ----------------------------------------------------------------------------- backup
def test_backup_is_verified_compressed_and_has_a_manifest(cfg, live):
    db, path = live
    before = _content_hash(path)
    r = create_backup(str(path), cfg.runtime.backup_dir, keep=14)
    assert r.compressed and r.path.name.endswith(".db.gz") and r.path.exists()
    assert not r.path.with_name(r.path.name[:-3]).exists()               # uncompressed copy removed after checks
    assert r.integrity == "ok" and r.counts == inspect_db(path)["counts"] and r.counts["signals"] == 60
    assert r.ratio > 2.0
    m = json.loads(r.manifest.read_text(encoding="utf-8"))
    assert m["sha256_db"] == r.sha256 and m["counts"] == r.counts and m["compressed"] is True
    with gzip.open(r.path, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == r.sha256
    assert _content_hash(path) == before                                 # source untouched
    assert not list(Path(cfg.runtime.backup_dir).glob("*.partial"))


def test_backup_includes_uncheckpointed_wal_data_and_does_not_write_the_source(cfg, live):
    db, path = live
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("INSERT INTO kv_state(key, value, updated_at) VALUES ('wal_only', '{\"v\": 1}', '2026-01-01')")
    con.commit()                                                          # committed, still only in the WAL
    wal = path.with_name(path.name + "-wal")
    assert wal.exists() and wal.stat().st_size > 0
    main_before, wal_before = _file_hash(path), _file_hash(wal)
    try:
        r = create_backup(str(path), cfg.runtime.backup_dir)
    finally:
        con.close()
    assert r.counts["kv_state"] == 2                                      # WAL row is in the backup
    out = Path(cfg.runtime.backup_dir) / "check.db"
    restore_backup(str(r.path), str(out))
    with closing(sqlite3.connect(str(out))) as c2:
        assert c2.execute("select count(*) from kv_state where key='wal_only'").fetchone()[0] == 1
    assert _file_hash(path) == main_before                                # source main file byte-identical
    assert _file_hash(wal) == wal_before or not wal.exists()              # (close may checkpoint, never the backup)


def test_compression_failure_keeps_the_verified_uncompressed_backup(cfg, live, monkeypatch):
    db, path = live
    real_open = bk.gzip.open

    def failing_open(p, mode="rb", **kw):
        if "w" in mode:
            raise OSError("disk full (simulated)")
        return real_open(p, mode, **kw)

    monkeypatch.setattr(bk.gzip, "open", failing_open)
    r = create_backup(str(path), cfg.runtime.backup_dir)
    assert not r.compressed and r.path.name.endswith(".db") and r.path.exists()
    assert "compression failed" in r.warnings[0]
    assert json.loads(r.manifest.read_text())["compressed"] is False
    assert inspect_db(r.path)["counts"] == inspect_db(path)["counts"]
    assert not list(Path(cfg.runtime.backup_dir).glob("*.partial")) and not list(Path(cfg.runtime.backup_dir).glob("*.gz"))


def test_compressed_copy_mismatch_falls_back_to_uncompressed(cfg, live, monkeypatch):
    db, path = live
    monkeypatch.setattr(bk, "_sha256_gz", lambda p: "0" * 64)
    r = create_backup(str(path), cfg.runtime.backup_dir)
    assert not r.compressed and r.path.exists() and "does not match" in r.warnings[0]


def test_rotation_keeps_newest_and_removes_manifests(cfg, live, monkeypatch):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    out.mkdir(parents=True, exist_ok=True)
    legacy = out / f"{path.stem}_20200101_000000.db"                     # old-format backup (before L1.0)
    with closing(sqlite3.connect(str(path))) as s, closing(sqlite3.connect(str(legacy))) as d:
        s.backup(d)
    unrelated = out / "notes.txt"
    unrelated.write_text("keep me")
    stamps = iter(["20260101_000001", "20260101_000002", "20260101_000003", "20260101_000004"])
    monkeypatch.setattr(bk, "_timestamp_name", lambda stem, d: f"{stem}_{next(stamps)}")
    results = [create_backup(str(path), str(out), keep=3) for _ in range(4)]
    kept = list_backups(str(out), path.stem)
    assert [p.name for p in kept] == [r.path.name for r in results[-3:]]
    assert not legacy.exists() and not results[0].path.exists() and not results[0].manifest.exists()
    assert all(r.manifest.exists() for r in results[-3:]) and unrelated.exists()


def test_database_backup_method_still_returns_the_backup_path(cfg, live):
    db, path = live
    p = db.backup(cfg.runtime.backup_dir, keep=2)
    assert p is not None and p.exists() and p.name.endswith(".db.gz")


# ----------------------------------------------------------------------------- restore
def test_restore_to_a_separate_file_matches_the_source(cfg, live, tmp_path):
    db, path = live
    live_hash = _file_hash(path)
    r = create_backup(str(path), cfg.runtime.backup_dir)
    target = tmp_path / "restore test" / "#restored%.db"                 # spaces and URI characters in the path
    res = restore_backup(str(r.path), str(target), live_db_path=str(path))
    assert res.integrity == "ok" and res.counts == r.counts
    assert _content_hash(target) == _content_hash(path)
    assert Database(f"sqlite:///{target}").open_signals() is not None     # usable by the bot's DB layer
    assert _file_hash(path) == live_hash                                  # live DB never touched
    assert not list(target.parent.glob("*.partial"))


def test_verify_only_writes_nothing(cfg, live):
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    files_before = sorted(Path(cfg.runtime.backup_dir).iterdir())
    v = verify_backup(str(r.path))
    assert v["integrity"] == "ok" and v["counts"] == r.counts and v["manifest"]
    assert sorted(Path(cfg.runtime.backup_dir).iterdir()) == files_before


@pytest.mark.parametrize("damage", ["flip_byte", "truncate", "manifest_counts", "manifest_sha", "not_sqlite"])
def test_damaged_backups_are_refused_and_nothing_is_written(cfg, live, tmp_path, damage):
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    if damage == "flip_byte":
        b = bytearray(r.path.read_bytes()); b[len(b) // 2] ^= 0xFF; r.path.write_bytes(bytes(b))
    elif damage == "truncate":
        r.path.write_bytes(r.path.read_bytes()[: r.stored_bytes // 2])
    elif damage in ("manifest_counts", "manifest_sha"):
        m = json.loads(r.manifest.read_text())
        if damage == "manifest_counts":
            m["counts"]["signals"] += 1
        else:
            m["sha256_db"] = "f" * 64
        r.manifest.write_text(json.dumps(m))
    else:
        r.manifest.unlink()
        with gzip.open(r.path, "wb") as fh:
            fh.write(b"this is not a database" * 100)
    target = tmp_path / "out.db"
    with pytest.raises(BackupError):
        restore_backup(str(r.path), str(target))
    assert not target.exists() and not list(tmp_path.glob("*.partial"))


def test_restore_refuses_unsafe_targets(cfg, live, tmp_path):
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    existing = tmp_path / "existing.db"
    existing.write_bytes(b"x")
    with pytest.raises(BackupError, match="target exists"):
        restore_backup(str(r.path), str(existing), live_db_path=str(path))
    assert existing.read_bytes() == b"x"
    restore_backup(str(r.path), str(existing), live_db_path=str(path), overwrite=True)    # explicit, non-live
    with pytest.raises(BackupError, match="live database"):
        restore_backup(str(r.path), str(path), live_db_path=str(path))                    # overwrite does not apply
    with pytest.raises(BackupError, match="live database"):
        restore_backup(str(r.path), str(path), live_db_path=str(path), overwrite=True)
    with pytest.raises(BackupError, match="not the configured live database"):
        restore_backup(str(r.path), str(tmp_path / "other.db"), live_db_path=str(path), replace_live=True)


def test_replace_live_refused_while_the_bot_runs(cfg, live, tmp_path):
    from core.instance_lock import InstanceLock
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    lock_path = tmp_path / "bot.lock"
    live_hash = _content_hash(path)
    with InstanceLock(str(lock_path)):                                    # "the bot is running"
        with pytest.raises(BackupError, match="bot is running"):
            restore_backup(str(r.path), str(path), live_db_path=str(path), replace_live=True,
                           lock_path=str(lock_path))
    assert _content_hash(path) == live_hash


def test_replace_live_takes_a_safety_backup_first(cfg, live, tmp_path):
    """Run against a COPY of the live DB in a temp folder - never the project database."""
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)                  # backup with 60 signals
    db.kv_set("after_backup", "newer data")                               # live DB now differs from the backup
    db.engine.dispose()
    live_before = _content_hash(path)
    res = restore_backup(str(r.path), str(path), live_db_path=str(path), replace_live=True,
                         lock_path=str(tmp_path / "bot.lock"), safety_backup_dir=str(tmp_path / "safety"))
    assert res.safety_backup is not None and res.safety_backup.exists()
    safety_copy = tmp_path / "safety_check.db"
    restore_backup(str(res.safety_backup), str(safety_copy))
    assert _content_hash(safety_copy) == live_before                     # previous live state fully preserved
    assert inspect_db(path)["counts"] == r.counts                         # live now equals the backup
    assert not path.with_name(path.name + "-wal").exists()
    assert Database(cfg.database_url).kv_get("after_backup") is None


def test_legacy_uncompressed_backup_without_manifest_can_be_restored(cfg, live, tmp_path):
    db, path = live
    legacy = tmp_path / f"{path.stem}_20250101_000000.db"
    with closing(sqlite3.connect(str(path))) as s, closing(sqlite3.connect(str(legacy))) as d:
        s.backup(d)
    with pytest.raises(BackupError, match="no manifest"):                  # explicit permission required
        restore_backup(str(legacy), str(tmp_path / "from_legacy.db"))
    assert not (tmp_path / "from_legacy.db").exists()
    res = restore_backup(str(legacy), str(tmp_path / "from_legacy.db"), allow_no_manifest=True)
    assert res.integrity == "ok" and res.counts == inspect_db(path)["counts"]
    assert any("WITHOUT a manifest" in w for w in res.warnings)


# ----------------------------------------------------------------------------- CLI
def test_cli_backup_list_verify_and_restore(cfg, live, tmp_path, monkeypatch, capsys):
    import main
    db, path = live
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    assert main.main(["backup"]) == 0
    out = capsys.readouterr().out
    assert "integrity: ok" in out and "compressed" in out
    b = list_backups(cfg.runtime.backup_dir, path.stem)[-1]
    assert main.main(["backup", "--list"]) == 0 and b.name in capsys.readouterr().out
    assert main.main(["restore", "--from", str(b), "--verify-only"]) == 0
    assert "OK:" in capsys.readouterr().out
    target = tmp_path / "cli_restore.db"
    assert main.main(["restore", "--from", str(b), "--to", str(target)]) == 0 and target.exists()
    assert main.main(["restore", "--from", str(b), "--to", str(path)]) == 1          # live: refused
    assert "RESTORE REFUSED" in capsys.readouterr().out


@pytest.mark.parametrize("with_shm", [True, False])
def test_backup_of_crashed_db_with_wal_leftover_never_writes_the_source(cfg, live, tmp_path, with_shm):
    """No other connection open (bot stopped/crashed): the backup must not checkpoint into the source."""
    import shutil
    db, path = live
    db.engine.dispose()
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("INSERT INTO kv_state(key, value, updated_at) VALUES ('crash_row', '{\"v\": 1}', '2026-01-01')")
    con.commit()
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    for suffix in ("", "-wal") + (("-shm",) if with_shm else ()):         # copy as if the process died here
        shutil.copyfile(str(path) + suffix, crashed / (path.name + suffix))
    con.close()
    src = crashed / path.name
    wal = crashed / (path.name + "-wal")
    assert wal.stat().st_size > 0
    main_before, wal_before = _file_hash(src), _file_hash(wal)
    r = create_backup(str(src), str(tmp_path / "bk"))
    assert r.counts["kv_state"] == 2 and r.integrity == "ok"              # WAL row captured
    assert _file_hash(src) == main_before                                 # main file not checkpointed into
    assert _file_hash(wal) == wal_before                                  # WAL left exactly as it was


# =============================================================================== L1.0 follow-up safety tests
import os
import subprocess
import sys
import time as _time


def _dir_state(path):
    """Bytes of the live db and every sibling file (names + content), to prove 'nothing changed'."""
    d = Path(path).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())
            if p.is_file() and p.name.startswith(Path(path).name)}


def _prepare_live(cfg, live, tmp_path, wal_rows=0):
    """A stopped bot: DB on disk (optionally with committed rows still in the WAL), one fresh backup."""
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    db.kv_set("after_backup", "newer")                       # live now differs from the backup
    db.engine.dispose()
    return path, r, tmp_path / "bot.lock"


def _live_restore(path, r, lock, tmp_path, **kw):
    return restore_backup(str(r.path), str(path), live_db_path=str(path), replace_live=True,
                          lock_path=str(lock), safety_backup_dir=str(tmp_path / "safety"), **kw)


# ---------------------------------------------------------------- other connections: refused, nothing changed
def test_idle_sqlite_connection_refuses_live_restore_without_changes(cfg, live, tmp_path, monkeypatch):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    monkeypatch.setattr(bk, "_processes_holding", lambda p: [])          # isolate the SQLite-level check
    other = sqlite3.connect(str(path))
    other.execute("select count(*) from signals").fetchone()          # idle reader (e.g. report command)
    before = _dir_state(path)
    with pytest.raises(BackupError, match="another SQLite connection"):
        _live_restore(path, r, lock, tmp_path)
    assert _dir_state(path) == before                                 # no live file changed
    assert not list(tmp_path.glob("safety/*")) and not list(path.parent.glob("*.pre-restore-*"))
    assert other.execute("select count(*) from signals").fetchone()[0] == 60
    other.close()


def test_concurrent_writer_process_refuses_live_restore(cfg, live, tmp_path):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    code = ("import sqlite3,sys,time\n"
            "c=sqlite3.connect(sys.argv[1]); c.execute('select 1').fetchone(); print('ready',flush=True)\n"
            "for i in range(400):\n"
            "  c.execute(\"insert into kv_state(key,value,updated_at) values(?, '{}', '2026')\", (f'w{i}',)); c.commit()\n"
            "  time.sleep(0.005)\n")
    writer = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert writer.stdout.readline().strip() == "ready"
        with pytest.raises(BackupError, match="other process|another SQLite connection"):
            _live_restore(path, r, lock, tmp_path)
    finally:
        writer.wait(timeout=60)
    assert not list(path.parent.glob("*.pre-restore-*"))
    with closing(sqlite3.connect(str(path))) as c:                     # every write of the writer survived
        assert c.execute("select count(*) from kv_state where key like 'w%'").fetchone()[0] == 400


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc check is Linux-only")
def test_process_holding_the_file_without_sqlite_lock_is_detected(cfg, live, tmp_path):
    """A connection opened but never used holds no SQLite lock - the /proc check still sees the open file."""
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    holder = subprocess.Popen([sys.executable, "-c", "import sys,time; f=open(sys.argv[1],'rb'); "
                               "print('ready',flush=True); time.sleep(30)", str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        before = _dir_state(path)
        with pytest.raises(BackupError, match="other process"):
            _live_restore(path, r, lock, tmp_path)
        assert _dir_state(path) == before
    finally:
        holder.kill()


# ---------------------------------------------------------------- every failure point rolls back
class FailingRename:
    """Fails the n-th forward rename (1: live->aside, 2/3: -wal/-shm->aside, last: restored->live)."""

    def __init__(self, fail_on, exc=None, create_sidecars_for=None):
        self.fail_on, self.calls = fail_on, 0
        self.exc = exc or PermissionError(32, "The process cannot access the file because it is being used "
                                              "by another process (simulated WinError 32)")
        self.sidecars = create_sidecars_for

    def __call__(self, a, b):
        self.calls += 1
        if self.calls == 1 and self.sidecars is not None:
            for s in ("-wal", "-shm"):                         # leftovers that must be rolled back too
                Path(str(self.sidecars) + s).write_bytes(b"")
        if self.calls == self.fail_on:
            raise self.exc
        os.replace(a, b)


@pytest.mark.parametrize("fail_on,label", [(1, "live->aside (Windows file lock)"), (2, "-wal->aside"),
                                           (3, "-shm->aside"), (4, "restored->live")])
def test_rollback_after_each_rename_failure(cfg, live, tmp_path, monkeypatch, fail_on, label):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    content_before = _content_hash(path)
    renamer = FailingRename(fail_on, create_sidecars_for=path)
    with pytest.raises(BackupError, match="rolled back"):
        _live_restore(path, r, lock, tmp_path, _replace=renamer)
    assert _content_hash(path) == content_before, label             # original database back in place
    assert Path(str(path) + "-wal").exists() and Path(str(path) + "-shm").exists(), label   # WAL/SHM restored
    assert not list(path.parent.glob("*.pre-restore-*")), label
    assert not list(path.parent.glob(".*.restore.partial*")), label
    assert Database(cfg.database_url).kv_get("after_backup") == "newer"


def test_rollback_when_restored_database_fails_post_verification(cfg, live, tmp_path, monkeypatch):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    before = _content_hash(path)
    real_inspect = bk.inspect_db
    monkeypatch.setattr(bk, "inspect_db", lambda p: (
        {"integrity": "corrupt (simulated)", "counts": {}} if Path(p) == path else real_inspect(p)))
    with pytest.raises(BackupError, match="rolled back"):
        _live_restore(path, r, lock, tmp_path)
    monkeypatch.setattr(bk, "inspect_db", real_inspect)
    assert _content_hash(path) == before and not list(path.parent.glob("*.pre-restore-*"))


def test_ctrl_c_during_live_replacement_rolls_back(cfg, live, tmp_path):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    before = _content_hash(path)
    with pytest.raises(KeyboardInterrupt):
        _live_restore(path, r, lock, tmp_path, _replace=FailingRename(2, exc=KeyboardInterrupt()))
    assert _content_hash(path) == before and not list(path.parent.glob("*.pre-restore-*"))
    assert not list(path.parent.glob(".*.restore.partial*"))


def test_failed_rollback_is_reported_as_critical_and_deletes_nothing(cfg, live, tmp_path, monkeypatch):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    real = os.replace

    def rollback_fails(a, b):                                         # rollback uses os.replace
        raise PermissionError(32, "locked (simulated)")
    renamer = FailingRename(4)

    def forward(a, b):
        renamer(a, b)
        if renamer.calls == 1:
            monkeypatch.setattr(bk.os, "replace", rollback_fails)   # the rollback will be refused too
    with pytest.raises(BackupError, match="CRITICAL"):
        _live_restore(path, r, lock, tmp_path, _replace=forward)
    monkeypatch.setattr(bk.os, "replace", real)
    aside = list(path.parent.glob("bot.db.pre-restore-*")) if path.name == "bot.db" else \
        list(path.parent.glob(f"{path.name}.pre-restore-*"))
    assert aside and not path.exists()                                # original kept aside, nothing deleted
    assert len(list((tmp_path / "safety").glob("*.db.gz"))) == 1      # plus a verified safety backup


def test_checkpoint_blocked_refuses_without_renaming(cfg, live, tmp_path, monkeypatch):
    path, r, lock = _prepare_live(cfg, live, tmp_path)

    class BusyCon:
        def __init__(self, real):
            self.real = real

        def execute(self, sql, *a):
            if "wal_checkpoint" in sql:
                class R:
                    def fetchone(self):
                        return (1, 5, 2)
                return R()
            return self.real.execute(sql, *a)

        def backup(self, *a, **k):
            return self.real.backup(*a, **k)

        def close(self):
            self.real.close()
    real_ex = bk._exclusive_connection
    monkeypatch.setattr(bk, "_exclusive_connection", lambda t: BusyCon(real_ex(t)))
    before = _dir_state(path)
    with pytest.raises(BackupError, match="checkpoint was blocked"):
        _live_restore(path, r, lock, tmp_path)
    assert _dir_state(path) == before


def test_successful_live_restore_keeps_previous_database_twice(cfg, live, tmp_path):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    before = _content_hash(path)
    res = _live_restore(path, r, lock, tmp_path)
    assert inspect_db(path)["counts"] == r.counts
    assert res.pre_restore_copy.exists() and _content_hash(res.pre_restore_copy) == before
    chk = tmp_path / "safety_check.db"
    restore_backup(str(res.safety_backup), str(chk))
    assert _content_hash(chk) == before
    db2 = Database(cfg.database_url)                                  # bot DB layer reopens it in WAL mode
    assert db2.kv_get("after_backup") is None
    with db2.engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"


# ---------------------------------------------------------------- Fix C: production DB never overwritten
def test_overwrite_never_replaces_a_bot_database_whatever_the_config(cfg, live, tmp_path, monkeypatch, capsys):
    import main
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    db.kv_set("after_backup", "newer")
    db.engine.dispose()
    before = _content_hash(path)
    # 1. API: the configured live DB is somewhere else (mismatched DATABASE_URL)
    with pytest.raises(BackupError, match="is a bot database"):
        restore_backup(str(r.path), str(path), live_db_path=str(tmp_path / "elsewhere.db"), overwrite=True)
    with pytest.raises(BackupError, match="not the configured live database"):
        restore_backup(str(r.path), str(path), live_db_path=str(tmp_path / "elsewhere.db"), replace_live=True,
                       lock_path=str(tmp_path / "bot.lock"))
    # 2. CLI with a mismatched configuration
    other_cfg = type(cfg)()
    other_cfg.database_url = f"sqlite:///{tmp_path / 'elsewhere.db'}"
    other_cfg.runtime.backup_dir = cfg.runtime.backup_dir
    monkeypatch.setattr(main, "load_config", lambda: other_cfg)
    assert main.main(["restore", "--from", str(r.path), "--to", str(path), "--overwrite"]) == 1
    assert "is a bot database" in capsys.readouterr().out
    assert _content_hash(path) == before
    # 3. non-database files may still be overwritten explicitly
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"not a database")
    restore_backup(str(r.path), str(junk), live_db_path=str(path), overwrite=True)
    assert inspect_db(junk)["counts"] == r.counts


def test_unreadable_sqlite_file_is_protected_by_default(tmp_path):
    p = tmp_path / "broken.db"
    p.write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)              # SQLite header, unreadable body
    assert bk.is_bot_database(p) is True
    q = tmp_path / "text.db"
    q.write_text("hello")
    assert bk.is_bot_database(q) is False


# ---------------------------------------------------------------- manifests and rotation
def test_every_backup_has_its_own_manifest_and_legacy_manifests_still_work(cfg, live, tmp_path):
    db, path = live
    r = create_backup(str(path), cfg.runtime.backup_dir)
    assert r.manifest.name == r.path.name + ".json"
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    gz = legacy_dir / f"{path.stem}_20250101_000000.db.gz"
    gz.write_bytes(r.path.read_bytes())
    m = json.loads(r.manifest.read_text())
    m["backup_file"] = gz.name
    (legacy_dir / f"{path.stem}_20250101_000000.json").write_text(json.dumps(m))   # first-draft naming
    assert bk.read_manifest(gz)["backup_file"] == gz.name
    assert verify_backup(str(gz))["manifest"]
    m["backup_file"] = "some_other_file.db"                           # legacy manifest for ANOTHER file: ignored
    (legacy_dir / f"{path.stem}_20250101_000000.json").write_text(json.dumps(m))
    assert bk.read_manifest(gz) is None


def test_rotation_when_windows_refuses_deletion(cfg, live, monkeypatch):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    stamps = iter([f"20260101_00000{i}" for i in range(1, 9)])
    monkeypatch.setattr(bk, "_timestamp_name", lambda stem, d: f"{stem}_{next(stamps)}")
    real = bk._unlink_quietly
    state = {"refuse": True}

    def flaky(p):                                                     # Windows/antivirus refuses deletions
        if state["refuse"] and p.name.endswith(".db"):
            return False
        return real(p)
    monkeypatch.setattr(bk, "_unlink_quietly", flaky)
    first = create_backup(str(path), str(out), keep=2)
    assert first.compressed and any("could not delete uncompressed copy" in w for w in first.warnings)
    assert (out / first.path.name[:-3]).exists() and first.manifest.exists()
    second = create_backup(str(path), str(out), keep=2)               # deletion still refused
    assert any("stale uncompressed duplicate" in w for w in second.warnings)
    assert first.manifest.exists() and second.manifest.exists()       # no manifest lost
    state["refuse"] = False
    third = create_backup(str(path), str(out), keep=2)                # retry succeeds, rotation by backup
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([second.path.name, second.manifest.name, third.path.name, third.manifest.name])
    assert verify_backup(str(second.path))["manifest"] and verify_backup(str(third.path))["manifest"]


def test_rotation_never_deletes_newest_even_if_older_deletions_fail(cfg, live, monkeypatch):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    stamps = iter([f"20260102_00000{i}" for i in range(1, 9)])
    monkeypatch.setattr(bk, "_timestamp_name", lambda stem, d: f"{stem}_{next(stamps)}")
    made = [create_backup(str(path), str(out), keep=10) for _ in range(3)]
    real = bk._unlink_quietly
    monkeypatch.setattr(bk, "_unlink_quietly", lambda p: False if p == made[0].path else real(p))
    newest = create_backup(str(path), str(out), keep=1)
    assert newest.path.exists() and newest.manifest.exists()
    assert made[0].path.exists() and made[0].manifest.exists()       # refused deletion: kept with its manifest


def test_cli_allow_no_manifest(cfg, live, tmp_path, monkeypatch, capsys):
    import main
    db, path = live
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    legacy = tmp_path / f"{path.stem}_20240101_000000.db"
    with closing(sqlite3.connect(str(path))) as s, closing(sqlite3.connect(str(legacy))) as d:
        s.backup(d)
    assert main.main(["restore", "--from", str(legacy), "--verify-only"]) == 0
    assert "NOT FOUND" in capsys.readouterr().out
    assert main.main(["restore", "--from", str(legacy), "--to", str(tmp_path / "a.db")]) == 1
    assert "no manifest" in capsys.readouterr().out and not (tmp_path / "a.db").exists()
    assert main.main(["restore", "--from", str(legacy), "--to", str(tmp_path / "a.db"), "--allow-no-manifest"]) == 0
    assert "WITHOUT a manifest" in capsys.readouterr().out


# ---------------------------------------------------------------- interrupted operations
def test_interrupted_backup_leaves_source_and_previous_backups_intact(cfg, live, monkeypatch):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    good = create_backup(str(path), str(out))
    good_bytes = good.path.read_bytes()
    src_before = _content_hash(path)

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(bk, "_make_standalone", interrupted)
    with pytest.raises(KeyboardInterrupt):
        create_backup(str(path), str(out))
    monkeypatch.undo()
    assert _content_hash(path) == src_before
    assert good.path.read_bytes() == good_bytes and good.manifest.exists()
    assert not list(out.glob(".*.partial*"))                          # half-written copy removed


def test_interrupted_compression_keeps_the_verified_uncompressed_backup(cfg, live, monkeypatch):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    real_open = bk.gzip.open

    def ctrl_c(p, mode="rb", **kw):
        if "w" in mode:
            raise KeyboardInterrupt
        return real_open(p, mode, **kw)
    monkeypatch.setattr(bk.gzip, "open", ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        create_backup(str(path), str(out))
    monkeypatch.undo()
    kept = [p for p in out.iterdir() if p.name.endswith(".db")]
    assert len(kept) == 1 and inspect_db(kept[0])["integrity"] == "ok"   # verified copy survived
    assert not list(out.glob("*.gz")) and not list(out.glob(".*.partial*"))


def test_stale_partials_are_cleaned_but_recent_ones_and_backups_are_kept(cfg, live):
    db, path = live
    out = Path(cfg.runtime.backup_dir)
    good = create_backup(str(path), str(out))
    old = [out / ".bot_20250101_000000.db.partial", out / ".bot_20250101_000000.db.gz.partial",
           out / ".bot_20250101_000000.db.partial-wal", out / ".verify_1_x.db.gz.partial"]
    fresh = out / ".bot_20991231_000000.db.partial"                  # could belong to a running backup
    lookalike = out / "bot_20250101_000000.db.partial.txt"           # not ours: never touched
    for p in old + [fresh, lookalike]:
        p.write_bytes(b"x")
    two_days = _time.time() - 2 * 86400
    for p in old + [lookalike]:
        os.utime(p, (two_days, two_days))
    removed = bk.cleanup_stale_partials(out)
    assert sorted(removed) == sorted(old)
    assert fresh.exists() and lookalike.exists() and good.path.exists() and good.manifest.exists()


def test_interrupted_restore_leaves_live_database_untouched(cfg, live, tmp_path, monkeypatch):
    path, r, lock = _prepare_live(cfg, live, tmp_path)
    before = _dir_state(path)

    def ctrl_c(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(bk, "_check", ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        _live_restore(path, r, lock, tmp_path)
    monkeypatch.undo()
    assert _dir_state(path) == before and not list(path.parent.glob(".*.restore.partial*"))
    # a killed earlier run left a temp file > 24h old: the next restore cleans it
    stale = path.parent / f".{path.name}.4242.restore.partial"
    stale.write_bytes(b"x")
    os.utime(stale, (_time.time() - 2 * 86400,) * 2)
    _live_restore(path, r, lock, tmp_path)
    assert not stale.exists()


# ---------------------------------------------------------------- writable fallback warning
def test_writable_fallback_connection_is_warned(cfg, live, monkeypatch, caplog):
    db, path = live
    real_connect = sqlite3.connect

    def no_readonly(target, *a, **kw):
        if kw.get("uri") and "mode=ro" in str(target) and Path(path).name in str(target):
            raise sqlite3.OperationalError("unable to open database file (simulated)")
        return real_connect(target, *a, **kw)
    monkeypatch.setattr(bk.sqlite3, "connect", no_readonly)
    with caplog.at_level("WARNING"):
        try:
            r = create_backup(str(path), cfg.runtime.backup_dir)
        except sqlite3.OperationalError:
            pytest.skip("inspect of the source path also uses the read-only URI")
    assert any("writable fallback" in w for w in r.warnings)
    assert "WRITABLE fallback" in caplog.text
