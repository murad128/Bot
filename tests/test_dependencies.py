"""P1: reproducible dependencies - requirements.txt pins, the hash lock, and time-zone data."""
import re
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"
LOCK = ROOT / "requirements.lock.txt"
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)\s*(?:;\s*(.+))?$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _direct():
    out = {}
    for raw in REQ.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = PIN.match(line)
        assert m, f"requirements.txt: not an exact '==' pin: {raw!r}"
        out[_norm(m.group(1))] = (m.group(2), (m.group(3) or "").replace('"', "'").strip())
    return out


def _locked():
    out, text = {}, LOCK.read_text(encoding="utf-8")
    for block in re.split(r"\n(?=[A-Za-z])", text):
        head = block.splitlines()[0]
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)\s*(?:;\s*([^\\]+?))?\s*\\?$", head)
        if m:
            out[_norm(m.group(1))] = (m.group(2), (m.group(3) or "").strip(), "--hash=sha256:" in block)
    return out


def test_every_direct_dependency_is_pinned_exactly():
    direct = _direct()
    for name in ("numpy", "pandas", "requests", "sqlalchemy", "python-dotenv", "tzdata", "metatrader5", "pytest"):
        assert name in direct, f"{name} missing from requirements.txt"
    assert direct["metatrader5"][1] == "sys_platform == 'win32'"       # Windows-only package


def test_lock_matches_requirements_and_is_fully_hashed():
    direct, locked = _direct(), _locked()
    assert locked, "requirements.lock.txt has no packages"
    for name, (version, marker) in direct.items():
        assert name in locked, f"{name} is in requirements.txt but not in the lock"
        assert locked[name][0] == version, f"{name}: requirements.txt {version} != lock {locked[name][0]}"
        if marker:
            assert locked[name][1] == marker, f"{name}: marker {marker!r} != lock {locked[name][1]!r}"
    unhashed = [n for n, (_, _, hashed) in locked.items() if not hashed]
    assert not unhashed, f"packages without hashes in the lock: {unhashed}"


def test_time_zones_used_by_the_bot_resolve():
    """zoneinfo needs the tzdata package on Windows; a missing database would break session/DST logic."""
    for zone in ("America/New_York", "Europe/London", "Asia/Tokyo", "Australia/Sydney"):
        assert ZoneInfo(zone).key == zone
