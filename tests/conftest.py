import atexit
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# --- test isolation: must run before `config` is imported -----------------
# Never read the real .env, never inherit real credentials / account settings
# from the shell, and never load the real MetaTrader5 package (tests use fakes).
os.environ["FXBOT_NO_DOTENV"] = "1"
_SCRUB_PREFIXES = ("TELEGRAM_", "MT5_")
_SCRUB_NAMES = {"FINNHUB_API_KEY", "FRED_API_KEY", "DATABASE_URL", "LOG_DIR", "BACKUP_DIR", "CSV_DATA_DIR",
                "CALENDAR_CSV_PATH"}
for _name in list(os.environ):
    if _name.startswith(_SCRUB_PREFIXES) or _name in _SCRUB_NAMES:
        del os.environ[_name]
sys.modules["MetaTrader5"] = None  # `import MetaTrader5` raises ImportError in tests

# Every configurable path points into a per-session sandbox, so even a bare Config() in a test can
# never reach the real logs/, data/bot.db, backups or historical CSVs of a bot running in the same folder.
SANDBOX = Path(tempfile.mkdtemp(prefix="fxbot-tests-"))
for _sub in ("logs", "data", "data/backups", "data/history"):
    (SANDBOX / _sub).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "LOG_DIR": str(SANDBOX / "logs"),
    "BACKUP_DIR": str(SANDBOX / "data" / "backups"),
    "DATABASE_URL": f"sqlite:///{SANDBOX / 'data' / 'bot.db'}",
    "CSV_DATA_DIR": str(SANDBOX / "data" / "history"),
    "CALENDAR_CSV_PATH": str(SANDBOX / "data" / "calendar.csv"),
})
atexit.register(shutil.rmtree, SANDBOX, ignore_errors=True)

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from core.domain import SymbolSpec  # noqa: E402

# Modules that build paths from the hard-coded BASE_DIR (calendar cache, bot.lock, MT5 offset state,
# diagnostics reports, the .env check). Their BASE_DIR is redirected to the sandbox for the whole run.
_BASE_DIR_MODULES = ("main", "core.orchestrator", "core.diagnostics")


@pytest.fixture(autouse=True, scope="session")
def _sandbox_base_dir():
    import importlib
    with pytest.MonkeyPatch.context() as mp:
        for name in _BASE_DIR_MODULES:
            mp.setattr(importlib.import_module(name), "BASE_DIR", SANDBOX)
        yield SANDBOX


@pytest.fixture(autouse=True)
def _restore_logging():
    """Close and remove every handler a test (e.g. main.setup_logging) attached, restore the levels."""
    loggers = [logging.getLogger(), logging.getLogger("signals")]
    before = [(lg, list(lg.handlers), lg.level) for lg in loggers]
    yield
    for lg, handlers, level in before:
        for h in list(lg.handlers):
            if h not in handlers:
                lg.removeHandler(h)
                h.close()
        lg.setLevel(level)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = load_config()
    c.database_url = f"sqlite:///{tmp_path / 'test.db'}"
    c.log_dir = str(tmp_path / "logs")
    c.telegram.bot_token = None
    c.telegram.chat_id = None
    c.runtime.backup_dir = str(tmp_path / "backups")
    return c


@pytest.fixture
def gold_spec():
    return SymbolSpec("XAUUSD", "XAUUSD", 0.01, 2, 0.10, 4.0, True, ["USD"], 0.0, 0.0)


@pytest.fixture
def eur_spec():
    return SymbolSpec("EURUSD", "EURUSD", 0.00001, 5, 0.0001, 2.0, False, ["EUR", "USD"], 0.0, 0.0)
