"""
Secret redaction for logs and stored error messages.

HTTP libraries put the request URL into exception messages. The Telegram
token is part of the API URL and the FRED key is a query parameter, so a
network error would otherwise write them into logs/system.log. Every
configured secret is registered here and replaced by "***" in all log output
(including tracebacks) and in error text stored in the database.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List

_SECRETS: List[str] = []
# generic patterns as a second line of defence (values never configured here)
_PATTERNS = [
    re.compile(r"(bot)\d{6,12}:[A-Za-z0-9_-]{20,}"),                       # Telegram token in URL
    re.compile(r"((?:api_key|token|apikey|password)=)[^&\s'\")]+", re.I),  # secrets in query strings
]


def register_secrets(values: Iterable[str]) -> None:
    for v in values:
        if v and len(v) >= 6 and v not in _SECRETS:
            _SECRETS.append(v)
    _SECRETS.sort(key=len, reverse=True)


def redact(text: str) -> str:
    if not text:
        return text
    for s in _SECRETS:
        text = text.replace(s, "***")
    text = _PATTERNS[0].sub(r"\1***", text)
    text = _PATTERNS[1].sub(r"\1***", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def secrets_from_config(cfg) -> List[str]:
    return [cfg.telegram.bot_token, cfg.mt5.password, cfg.news.finnhub_api_key, cfg.news.fred_api_key,
            str(cfg.mt5.login) if cfg.mt5.login else None]
