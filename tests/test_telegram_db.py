from datetime import datetime, timedelta, timezone

import pytest

from config import TelegramConfig
from core.domain import ComponentScore, Direction, RiskPlan, SignalCandidate
from core.signal_tracker import SignalTracker
from database.database import Database
from integrations import telegram_messages as tm
from integrations.telegram_bot import TelegramNotifier

NOW = datetime(2024, 3, 4, 10, 0, tzinfo=timezone.utc)


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload
        self.text = str(payload)

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        r = self.responses.pop(0) if self.responses else Resp(200, {"ok": True, "result": {"message_id": 1}})
        if isinstance(r, Exception):
            raise r
        return r


def candidate(symbol="XAUUSD", t=NOW):
    plan = RiskPlan(entry=2000.0, entry_low=1999.7, entry_high=2000.0, stop_loss=1997.0,
                    take_profits=[2003.0, 2006.0, 2009.0], risk=3.0, rr_targets=[1, 2, 3], primary_rr=2.0,
                    room_to_obstacle_r=None)
    return SignalCandidate(symbol, Direction.BUY, "OB_RETEST", t, "M15", "H1", "Bullish", "London", 84.0,
                           "Strong", plan, [ComponentScore("smc", 0.8, 20)],
                           ["H1 bullish market structure", "M15 liquidity sweep"], ["High volatility"], 2.0, 0.2,
                           {"session_bucket": "London", "tags": ["zone:OB"], "components": {"smc": 0.8}})


@pytest.fixture
def db(cfg):
    return Database(cfg.database_url)


def tg_cfg():
    return TelegramConfig(bot_token="123:ABC", chat_id="42", max_retries=3)


def test_signal_message_contains_required_fields(gold_spec):
    text = tm.signal_message(candidate(), gold_spec, 7)
    for s in ("XAUUSD – BUY SIGNAL", "Entry:", "Stop Loss:", "TP1:", "TP2:", "TP3:", "Risk/Reward: 1:2.0",
              "Timeframe: M15 / H1", "Signal Score:</b> 84/100", "Technical Reasons", "Session:", "ACTIVE",
              "not financial advice", "#7"):
        assert s.replace("</b>", "") in text.replace("<b>", "").replace("</b>", "") or s in text


def test_dedup_and_dry_run(db, tmp_path):
    n = TelegramNotifier(TelegramConfig(), db, dry_run_path=str(tmp_path / "dry.log"))
    assert n.send("hello", key="k1")
    assert not n.send("hello again", key="k1")
    assert (tmp_path / "dry.log").read_text().count("hello") == 1


def test_retry_after_429_then_success(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    sess = FakeSession([Resp(429, {"ok": False, "parameters": {"retry_after": 1}}),
                        Resp(200, {"ok": True, "result": {"message_id": 5}})])
    n = TelegramNotifier(tg_cfg(), db, session=sess)
    assert n.send("x", key="a")
    assert len(sess.calls) == 2


def test_failure_goes_to_outbox_and_is_flushed(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    sess = FakeSession([Resp(502, {})] * 3)
    n = TelegramNotifier(tg_cfg(), db, session=sess)
    assert not n.send("important", key="evt:1")
    assert len(db.outbox_pending()) == 1 and not db.notification_sent("evt:1")
    assert n.flush_outbox() == 1
    assert db.outbox_pending() == [] and db.notification_sent("evt:1")


def test_html_parse_error_falls_back_to_plain(db, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    sess = FakeSession([Resp(400, {"ok": False, "description": "Bad Request: can't parse entities"}),
                        Resp(200, {"ok": True, "result": {"message_id": 9}})])
    assert TelegramNotifier(tg_cfg(), db, session=sess).send("<b>broken")
    assert "parse_mode" not in sess.calls[1][1]


def test_signal_persistence_tracking_and_dedup(db, cfg):
    tr = SignalTracker(db, cfg.risk)
    rec = tr.register(candidate())
    assert rec is not None and tr.register(candidate()) is None        # duplicate prevented
    import pandas as pd
    idx = pd.date_range(NOW, periods=4, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": [2000, 2001, 2003.5, 2004], "high": [2001, 2003.2, 2004, 2004.5],
                         "low": [1999.5, 2000.6, 2002.5, 2001.0], "close": [2001, 2003, 2003.8, 2001.5]}, index=idx)
    ev = tr.update_with_bars(rec, bars, spread=0.2)
    kinds = [e.kind for e in ev]
    assert kinds[0] == "TP1_HIT"
    again = tr.update_with_bars(db.get_signal(rec.id), bars, spread=0.2)
    assert again == []                                               # bars already processed
    stored = db.get_signal(rec.id)
    assert stored.tp_hits >= 1 and stored.current_sl == pytest.approx(2000.0)


def test_consecutive_losses_and_backup(db, cfg):
    from database.models import SignalRecord
    for i, outcome in enumerate(["WIN", "LOSS", "LOSS"]):
        db.add_signal(SignalRecord(dedup_key=f"k{i}", symbol="EURUSD", direction="BUY", setup_type="X",
                                   entry_timeframe="M15", confirm_timeframe="H1", session="London", score=80,
                                   grade="Strong", entry=1, entry_low=1, entry_high=1, stop_loss=0.9, current_sl=0.9,
                                   tp1=1.1, tp2=1.2, tp3=1.3, risk=0.1, rr=2, status="STOPPED", outcome=outcome,
                                   created_at=NOW + timedelta(hours=i), closed_at=NOW + timedelta(hours=i, minutes=30)))
    assert db.consecutive_losses() == 2
    assert db.last_signal_time("EURUSD") == NOW + timedelta(hours=2)
    path = db.backup(cfg.runtime.backup_dir, keep=2)
    assert path is not None and path.exists()


def test_daily_report_explains_calculation():
    from core.performance_analyzer import compute_stats
    s = compute_stats([])
    text = tm.daily_report_message("2024-03-04", s, s, {}, {}, None, None, 0)
    assert "DAILY PERFORMANCE REPORT" in text and "not separate wins" in text and "resolved" in text
