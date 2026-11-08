"""
Telegram delivery via the Bot HTTP API (no extra SDK needed).

* Deduplication: every notification may carry a `key`; a key is delivered at
  most once (persisted in the `notifications` table, survives restarts).
* Rate limits: HTTP 429 honours `retry_after`; a small client-side spacing
  keeps us under Telegram's ~1 msg/s per chat guideline.
* Failures: transient errors are retried with exponential backoff; if all
  retries fail the message goes to the DB outbox and is retried later.
* Dry-run: without TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID messages are written
  to logs/telegram_dryrun.log so the full pipeline can be tested offline.
* Commands (optional, read-only): /status /active /report /help - only from
  the configured chat id.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

from config import TelegramConfig

log = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000


class TelegramNotifier:
    def __init__(self, cfg: TelegramConfig, db=None, dry_run_path: Optional[str] = None,
                 session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.db = db
        self.session = session or requests.Session()
        self.dry_run_path = Path(dry_run_path) if dry_run_path else None
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._offset: Optional[int] = None
        self._commands: Dict[str, Callable[[], str]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    # ------------------------------------------------------------------ low level
    def _api(self, method: str, payload: dict, timeout: Optional[int] = None) -> dict:
        url = API.format(token=self.cfg.bot_token, method=method)
        resp = self.session.post(url, json=payload, timeout=timeout or self.cfg.timeout_s)
        try:
            data = resp.json()
        except ValueError:
            data = {"ok": False, "description": resp.text[:200]}
        if resp.status_code == 429:
            retry = int((data.get("parameters") or {}).get("retry_after", 5))
            raise _RetryAfter(retry)
        if resp.status_code >= 500:
            raise _Transient(f"HTTP {resp.status_code}")
        if not data.get("ok"):
            raise _Permanent(data.get("description", f"HTTP {resp.status_code}"))
        return data

    def _send_raw(self, text: str) -> Optional[int]:
        if not self.enabled:
            self._dry_run(text)
            return None
        chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)] or [""]
        msg_id = None
        for chunk in chunks:
            delay = 1.0
            for attempt in range(self.cfg.max_retries):
                with self._lock:
                    wait = 1.05 - (time.monotonic() - self._last_send)
                    if wait > 0:
                        time.sleep(wait)
                    try:
                        data = self._api("sendMessage", {"chat_id": self.cfg.chat_id, "text": chunk,
                                                         "parse_mode": "HTML", "disable_web_page_preview": True})
                        self._last_send = time.monotonic()
                        msg_id = data["result"]["message_id"]
                        break
                    except _RetryAfter as e:
                        self._last_send = time.monotonic()
                        log.warning("Telegram 429, retry after %ss", e.seconds)
                        time.sleep(min(e.seconds, 60))
                    except (_Transient, requests.RequestException) as e:
                        log.warning("Telegram transient error (%s), retry %d", e, attempt + 1)
                        time.sleep(delay)
                        delay = min(delay * 2, 30)
                    except _Permanent as e:
                        if "parse" in str(e).lower():
                            # fall back to plain text if HTML formatting was rejected
                            data = self._api("sendMessage", {"chat_id": self.cfg.chat_id, "text": chunk})
                            msg_id = data["result"]["message_id"]
                            break
                        raise
            else:
                raise _Transient("max retries exceeded")
        return msg_id

    def _dry_run(self, text: str) -> None:
        log.info("[telegram dry-run] %s", text.splitlines()[0] if text else "")
        if self.dry_run_path:
            self.dry_run_path.parent.mkdir(parents=True, exist_ok=True)
            with self.dry_run_path.open("a", encoding="utf-8") as fh:
                fh.write(f"----- {time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")

    # ------------------------------------------------------------------ public
    def send(self, text: str, kind: str = "info", key: Optional[str] = None) -> bool:
        """Send once per key. Returns True if delivered (or dry-run logged)."""
        if key and self.db is not None and self.db.notification_sent(key):
            log.debug("Notification %s already sent", key)
            return False
        try:
            self._send_raw(text)
        except Exception as exc:
            log.error("Telegram delivery failed (%s): %s", kind, exc)
            if self.db is not None:
                self.db.outbox_add(kind, text, key, str(exc))
            return False
        if key and self.db is not None:
            self.db.mark_notification(key, kind)
        return True

    def flush_outbox(self) -> int:
        if self.db is None or not self.enabled:
            return 0
        sent = 0
        for m in self.db.outbox_pending():
            if m.key and self.db.notification_sent(m.key):
                self.db.outbox_done(m.id)
                continue
            try:
                self._send_raw(m.text)
                self.db.outbox_done(m.id)
                if m.key:
                    self.db.mark_notification(m.key, m.kind)
                sent += 1
            except Exception as exc:
                self.db.outbox_failed(m.id, str(exc))
                break
        return sent

    def test_connection(self) -> str:
        if not self.enabled:
            return "Telegram not configured (dry-run mode)"
        data = self._api("getMe", {})
        return f"Connected as @{data['result'].get('username')}"

    # ------------------------------------------------------------------ commands
    def register_command(self, name: str, handler: Callable[[], str]) -> None:
        self._commands[name.lstrip("/")] = handler

    def start_polling(self) -> None:
        if not self.enabled or not self.cfg.enable_commands or self._thread:
            return
        self._thread = threading.Thread(target=self._poll_loop, name="telegram-commands", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = {"timeout": 25, "allowed_updates": ["message"]}
                if self._offset is not None:
                    payload["offset"] = self._offset
                data = self._api("getUpdates", payload, timeout=35)
                for upd in data.get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    if str((msg.get("chat") or {}).get("id")) != str(self.cfg.chat_id):
                        continue
                    text = (msg.get("text") or "").strip()
                    if not text.startswith("/"):
                        continue
                    cmd = text.split()[0][1:].split("@")[0].lower()
                    handler = self._commands.get(cmd) or self._commands.get("help")
                    if handler:
                        try:
                            self.send(handler(), kind="command")
                        except Exception as exc:
                            log.exception("Command %s failed: %s", cmd, exc)
            except Exception as exc:
                log.debug("Telegram polling error: %s", exc)
                self._stop.wait(10)


class _RetryAfter(Exception):
    def __init__(self, seconds: int):
        super().__init__(seconds)
        self.seconds = seconds


class _Transient(Exception):
    pass


class _Permanent(Exception):
    pass
