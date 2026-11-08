import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from core.domain import SymbolSpec  # noqa: E402


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
