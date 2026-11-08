"""Diagnostic report + production/demo/read-only guards, against the simulated Vantage terminal."""
import pytest

from config import MT5Config
from core.diagnostics import classify_gaps, run_diagnostics
from core.domain import Timeframe
from core.market_data import DataProviderError
from integrations.mt5_connector import MT5DataProvider, ReadOnlyMT5, mask_login
from tests.fake_vantage import FakeVantageMT5
from tests.test_telegram_db import FakeSession


def _cfg(cfg, tmp_path):
    cfg.mt5.server_utc_offset = "2"
    cfg.mt5.server_time_mode = "ny_close"      # Vantage convention: per-date DST offset
    cfg.instruments = {k: cfg.instrument(k) for k in ("XAUUSD", "EURUSD", "USDJPY")}
    cfg.news.calendar_provider = "csv"
    cal = tmp_path / "cal.csv"
    cal.write_text("time_utc,currency,impact,title\n2030-01-01T13:30:00Z,USD,High,CPI m/m\n")
    cfg.news.calendar_csv_path = str(cal)
    cfg.telegram.bot_token, cfg.telegram.chat_id = "999999:SECRETTOKENabcdefghijklmnop", "42"
    cfg.csv_data_dir = str(tmp_path / "history")
    return cfg


@pytest.fixture(scope="module")
def fake():
    return FakeVantageMT5(days=75)


def test_read_only_proxy_blocks_trading(fake):
    ro = ReadOnlyMT5(fake)
    for fn in ("order_send", "order_check", "positions_get", "orders_get", "history_deals_get", "order_calc_margin"):
        with pytest.raises(PermissionError):
            getattr(ro, fn)
    assert ro.copy_rates_from_pos and ro.TIMEFRAME_M5 == 5
    with pytest.raises(PermissionError):
        ro.order_send = lambda *a: None
    assert fake.orders_sent == 0


def test_real_account_refused():
    p = MT5DataProvider(MT5Config(server_utc_offset="2", max_reconnect_attempts=1), mt5_module=FakeVantageMT5(trade_mode=2, days=3))
    with pytest.raises(DataProviderError, match=r"is a REAL account\. MT5_REQUIRE_DEMO=true: refusing"):
        p.connect()
    assert not p.is_connected()


def test_mask_login():
    assert mask_login(24681357) == "*****357" and mask_login(None) == "?"


def test_diagnostic_report_end_to_end(cfg, tmp_path, fake):
    from integrations.telegram_bot import TelegramNotifier
    cfg = _cfg(cfg, tmp_path)
    sess = FakeSession([])
    notifier = TelegramNotifier(cfg.telegram, session=sess)
    prov = MT5DataProvider(cfg.mt5, mt5_module=fake)
    rep = run_diagnostics(cfg, provider=prov, notifier=notifier, days=60)
    st = {c.component: c.status for c in rep.checks}
    detail = {c.component: c.detail for c in rep.checks}
    # the simulated terminal is NOT the official package -> production check must flag it
    assert st["Production data source"] == "FAIL" and "not the official package" in detail["Production data source"]
    assert st["Trading lock"] == "PASS" and st["Demo-only guard"] == "PASS"
    assert st["Account"] == "PASS" and "DEMO account *****357" in detail["Account"]
    assert st["Broker"] == "PASS"
    assert "balance 10,000.00 USD" in detail["Account balance"] and "1:500" in detail["Account balance"]
    assert st["Gold symbol"] == "PASS" and detail["Gold symbol"].startswith("'XAUUSD+'")
    assert "XAUEUR" not in detail["Gold symbol"]                     # only gold vs USD candidates
    assert st["Other symbols"] == "WARN" and "USDJPY" in detail["Other symbols"]
    assert "bid" in detail["Live price"] and "spread 25 points = 2.5 pips" in detail["Live price"]
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        assert st[f"History {tf}"] in ("PASS", "WARN"), detail[f"History {tf}"]
        assert (tmp_path / "history" / f"XAUUSD_{tf}.csv").exists()
    # broker builds H4/D1 on server time; the NY-close alignment must reproduce them
    for tf in ("M15", "H1", "H4", "D1"):
        assert st[f"Consistency M5->{tf}"] == "PASS", detail[f"Consistency M5->{tf}"]
    assert "UTC alignment would match" in detail["Consistency M5->H4"]
    assert st["Telegram"] == "PASS" and any("sendMessage" in u for u, _ in sess.calls)
    assert st["Economic calendar"] == "PASS"
    assert st["Signal engine"] == "PASS" and "2/3" in detail["Signal engine"]
    md = rep.to_markdown()
    for secret in ("SECRETTOKEN", "24681357", "SHOULD NEVER BE PRINTED"):
        assert secret not in md and secret not in rep.to_console()
    assert fake.orders_sent == 0


def test_daily_break_and_weekend_gap_classification(fake):
    import pandas as pd
    m5 = fake._tf_bars(5).copy()
    m5.index = (m5.index - pd.Timedelta(hours=2)).tz_localize("UTC") if m5.index.tz is None else m5.index - pd.Timedelta(hours=2)
    g = classify_gaps(m5, Timeframe.M5)
    assert g["weekend"] and g["daily_break"] and not g["other"]


def test_live_run_refuses_offline_provider(cfg, monkeypatch):
    import main
    cfg.data_provider = "csv"
    assert "Refusing to run the live bot" in main.production_guard(cfg, offline=False)
    cfg.telegram.bot_token, cfg.telegram.chat_id = "x:y", "1"
    assert main.production_guard(cfg, offline=True) == "" and not cfg.telegram.enabled


def test_ny_close_conversion_is_dst_correct():
    from datetime import datetime, timezone
    import pandas as pd
    p = MT5DataProvider(MT5Config(server_utc_offset="2", server_time_mode="ny_close"), mt5_module=FakeVantageMT5(days=2))
    summer = int(pd.Timestamp("2024-07-10 10:00").value // 10**9)     # server wall clock (UTC+3)
    winter = int(pd.Timestamp("2024-01-10 10:00").value // 10**9)     # server wall clock (UTC+2)
    utc = p.server_to_utc([summer, winter])
    assert utc[0] == pd.Timestamp("2024-07-10 07:00", tz="UTC")
    assert utc[1] == pd.Timestamp("2024-01-10 08:00", tz="UTC")
    back = p.utc_to_server(datetime(2024, 7, 10, 7, 0, tzinfo=timezone.utc))
    assert back.replace(tzinfo=None) == datetime(2024, 7, 10, 10, 0)


def test_constant_mode_preserved_with_explicit_offset():
    import pandas as pd
    p = MT5DataProvider(MT5Config(server_utc_offset="3"), mt5_module=FakeVantageMT5(days=2))
    assert p.time_mode == "constant"
    t = int(pd.Timestamp("2024-01-10 10:00").value // 10**9)
    assert p.server_to_utc([t])[0] == pd.Timestamp("2024-01-10 07:00", tz="UTC")


def test_auto_detection_picks_ny_close_for_vantage_style_server(tmp_path):
    from datetime import datetime, timezone
    from integrations.mt5_connector import ny_close_offset
    from tests.test_mt5_connector import FakeMT5, OFFSET_H
    p = MT5DataProvider(MT5Config(server_utc_offset="auto"), mt5_module=FakeMT5(), state_path=str(tmp_path / "o.json"))
    p.connect()
    p.server_offset("XAUUSD+")
    expected = "ny_close" if OFFSET_H == ny_close_offset(datetime.now(timezone.utc)) else "constant"
    assert p.time_mode == expected
    assert '"mode": "%s"' % expected in (tmp_path / "o.json").read_text()


def test_history_across_dst_change_is_continuous(tmp_path):
    """Bars downloaded across the US DST change must not jump by an hour."""
    from datetime import datetime, timezone
    import pandas as pd
    fake = FakeVantageMT5(days=20, end=datetime(2024, 11, 12, 12, 0, tzinfo=timezone.utc))   # spans 3 Nov 2024
    p = MT5DataProvider(MT5Config(server_utc_offset="2", server_time_mode="ny_close"), mt5_module=fake)
    p.connect()
    df = p.get_rates_range("XAUUSD+", Timeframe.M1, datetime(2024, 10, 28, tzinfo=timezone.utc),
                           datetime(2024, 11, 8, tzinfo=timezone.utc))
    truth = fake.m1[(fake.m1.index >= df.index[0]) & (fake.m1.index <= df.index[-1])]
    assert len(df) == len(truth)
    assert (df.index == truth.index).all()                         # identical UTC timestamps on both sides of DST
    assert (df["close"].to_numpy() == truth["close"].to_numpy()).all()
