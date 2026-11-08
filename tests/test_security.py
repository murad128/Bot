"""Secrets must never reach logs, stored errors, or the git index."""
import logging
import subprocess
from pathlib import Path

import requests

from config import TelegramConfig
from core.redact import RedactingFormatter, redact, register_secrets
from integrations.macro_data import FredClient
from integrations.news_provider import FinnhubNewsProvider
from integrations.telegram_bot import TelegramNotifier

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "999999:SECRETTOKENabcdefghijklmnop"


def _offline_session():
    s = requests.Session()
    s.trust_env = False      # no proxy: connections fail like a network outage
    return s


def test_network_errors_do_not_leak_secrets(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr("time.sleep", lambda s: None)
    log_file = tmp_path / "sys.log"
    h = logging.FileHandler(log_file)
    h.setFormatter(RedactingFormatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    try:
        from database.database import Database
        db = Database(cfg.database_url)
        tg = TelegramNotifier(TelegramConfig(bot_token=TOKEN, chat_id="1", max_retries=1, timeout_s=2), db,
                              session=_offline_session())
        monkeypatch.setattr("integrations.telegram_bot.API", "http://127.0.0.1:9/bot{token}/{method}")
        assert not tg.send("x", key="k")
        fred = FredClient("FAKEFREDKEY12345", session=_offline_session())
        fred.URL = "http://127.0.0.1:9/fred"
        fred.reading("DGS10")
        fh = FinnhubNewsProvider("SECRETKEYfinnhub", session=_offline_session())
        fh.URL = "http://127.0.0.1:9/news"
        try:
            fh.latest()
        except Exception:
            logging.getLogger("t").exception("headline poll failed")
        stored = db.outbox_pending()[0].last_error or ""
    finally:
        root.removeHandler(h)
        h.close()
    text = log_file.read_text() + stored + (fred.last_error or "")
    for secret in (TOKEN, "SECRETTOKEN", "FAKEFREDKEY12345", "SECRETKEYfinnhub"):
        assert secret not in text, secret


def test_generic_patterns_redacted_even_if_unregistered():
    s = "url: /bot123456789:FAKEfakeFAKEfakeFAKEfakeFAKEfake00/sendMessage?x=1&api_key=abc123&token=zz"
    out = redact(s)
    assert "FAKEfake" not in out and "abc123" not in out and "token=zz" not in out


def test_config_dump_masks_secrets(cfg):
    cfg.telegram.bot_token, cfg.mt5.password, cfg.news.fred_api_key = TOKEN, "pw123456", "k" * 32
    d = str(cfg.to_dict())
    assert TOKEN not in d and "pw123456" not in d and "k" * 32 not in d


def test_gitignore_excludes_sensitive_paths():
    paths = [".env", ".env.local", ".env.production", "data/bot.db", "data/bot.db-wal", "logs/system.log",
             "logs/telegram_dryrun.log", "backups/x.tar.gz", "data/backups/a.db", "data/mt5_server_offset.json",
             "data/bot.lock", "data/history/XAUUSD_M5.csv", "credentials.json", "secrets.toml", "id_rsa"]
    res = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT, capture_output=True, text=True)
    ignored = set(res.stdout.split())
    assert set(paths) == ignored, set(paths) - ignored
    keep = subprocess.run(["git", "check-ignore", "--no-index", ".env.example"], cwd=ROOT, capture_output=True)
    assert keep.returncode == 1          # template must stay tracked
