"""
History freshness and gap classification across DST transitions, weekends,
the daily gold break and genuinely missing candles.

Two kinds of evidence:
* hand-counted cases (every expected number worked out manually below), and
* an oracle test: the simulated Vantage terminal builds real bar grids on a
  New-York+7h server clock; for random (last bar, now) pairs the number of
  bars that actually closed in between must equal expected_missing_bars().
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.diagnostics import classify_gaps, consistency, expected_missing_bars, is_history_stale
from core.domain import Timeframe
from tests.fake_vantage import FakeVantageMT5

NY = ZoneInfo("America/New_York")
U = lambda s: pd.Timestamp(s, tz="UTC").to_pydatetime()


# ----------------------------------------------------------------------------- hand-counted cases
# US DST 2026: starts Sun 8 Mar (NY close 22:00 -> 21:00 UTC), ends Sun 1 Nov (21:00 -> 22:00 UTC)
@pytest.mark.parametrize("last_close,now,tf,missing", [
    # DST START weekend: last bar before it closed Fri 6 Mar 22:00 UTC (17:00 EST)
    ("2026-03-06 22:00", "2026-03-09 20:30", Timeframe.D1, 0),   # Monday D1 closes 21:00 UTC (17:00 EDT)
    ("2026-03-06 22:00", "2026-03-09 21:30", Timeframe.D1, 1),
    ("2026-03-06 22:00", "2026-03-09 05:10", Timeframe.H4, 2),   # Monday H4 closes 01:00 and 05:00 UTC
    ("2026-03-06 22:00", "2026-03-09 04:30", Timeframe.H4, 1),
    ("2026-03-06 22:00", "2026-03-08 22:04", Timeframe.M5, 0),   # Sunday open 21:00 UTC, break until 22:00
    ("2026-03-06 22:00", "2026-03-08 22:30", Timeframe.M5, 5),   # 22:05 .. 22:25
    # DST END weekend: last bar before it closed Fri 30 Oct 21:00 UTC (17:00 EDT)
    ("2026-10-30 21:00", "2026-11-02 21:30", Timeframe.D1, 0),   # Monday D1 closes 22:00 UTC (17:00 EST)
    ("2026-10-30 21:00", "2026-11-02 22:30", Timeframe.D1, 1),
    ("2026-10-30 21:00", "2026-11-02 06:10", Timeframe.H4, 2),   # Monday H4 closes 02:00 and 06:00 UTC
    ("2026-10-30 21:00", "2026-11-01 23:04", Timeframe.M5, 0),   # Sunday open 22:00 UTC, break until 23:00
    ("2026-10-30 21:00", "2026-11-01 23:30", Timeframe.H1, 0),   # first H1 bar 23:00-00:00 not closed
    ("2026-10-30 21:00", "2026-11-02 01:10", Timeframe.H1, 2),   # 00:00 and 01:00
    # ordinary weekend (winter)
    ("2026-11-06 22:00", "2026-11-08 23:30", Timeframe.M5, 5),
    ("2026-11-06 22:00", "2026-11-09 03:00", Timeframe.H4, 1),
    # daily gold break in SUMMER (21:00-22:00 UTC) and WINTER (22:00-23:00 UTC)
    ("2026-07-15 21:00", "2026-07-15 22:04", Timeframe.M5, 0),
    ("2026-07-15 21:00", "2026-07-15 22:30", Timeframe.M5, 5),
    ("2026-07-15 21:00", "2026-07-15 22:30", Timeframe.H1, 0),   # 21-22 bar does not exist, 22-23 not closed
    ("2026-11-04 22:00", "2026-11-04 23:04", Timeframe.M5, 0),
    ("2026-11-04 22:00", "2026-11-05 00:10", Timeframe.H1, 1),   # only 23:00-00:00
])
def test_expected_missing_bars_hand_counted(last_close, now, tf, missing):
    assert expected_missing_bars(U(last_close), U(now), tf) == missing


@pytest.mark.parametrize("last_close,now,tf,stale", [
    # Monday after the DST-start weekend, history complete or one bar late -> not stale
    ("2026-03-06 22:00", "2026-03-09 09:00", Timeframe.D1, False),
    ("2026-03-06 22:00", "2026-03-09 04:30", Timeframe.H4, False),
    # ... genuinely missing -> stale
    ("2026-03-06 22:00", "2026-03-09 05:10", Timeframe.H4, True),
    ("2026-03-06 22:00", "2026-03-10 21:30", Timeframe.D1, True),
    ("2026-10-30 21:00", "2026-11-02 10:00", Timeframe.M5, True),
])
def test_staleness_across_dst_weekends(last_close, now, tf, stale):
    assert is_history_stale(U(last_close), U(now), tf) is stale


# ----------------------------------------------------------------------------- oracle (simulated broker)
CODE = {"M5": 5, "M15": 15, "H1": 16385, "H4": 16388, "D1": 16408}


def _broker_closes_utc(fake, tf):
    b = fake._tf_bars(CODE[tf.value])                                # server clock = NY + 7h
    c = (b.index + pd.Timedelta(minutes=tf.minutes) - pd.Timedelta(hours=7))
    c = c.tz_localize(NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    return c[~c.isna()]


@pytest.mark.parametrize("end", [datetime(2026, 3, 20, 12, tzinfo=timezone.utc),     # spans DST start
                                 datetime(2026, 11, 13, 12, tzinfo=timezone.utc),    # spans DST end
                                 datetime(2026, 7, 24, 12, tzinfo=timezone.utc)])    # summer, no change
@pytest.mark.parametrize("tf", [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1])
def test_expected_missing_bars_matches_simulated_broker(end, tf):
    fake = FakeVantageMT5(days=20, end=end, seed=11)
    closes = _broker_closes_utc(fake, tf)
    data_end = fake.m1.index[-1].to_pydatetime()
    rng = np.random.default_rng(7)
    checked = 0
    while checked < 120:
        i = int(rng.integers(0, len(closes) - 3))
        lc = closes[i].to_pydatetime()
        now = lc + timedelta(minutes=int(rng.integers(1, 5 * 24 * 60)))
        if now > data_end:
            continue
        truth = int(((closes > lc) & (closes <= now - timedelta(minutes=5))).sum())
        assert expected_missing_bars(lc, now, tf) == truth, (tf, lc, now)
        checked += 1


# ----------------------------------------------------------------------------- gap classification
def _bars(start, end, tf):
    idx = pd.date_range(start, end, freq=tf.pandas_rule, tz="UTC", inclusive="left")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)


def test_gaps_weekend_daily_break_and_holes_across_dst():
    fake = FakeVantageMT5(days=20, end=datetime(2026, 11, 13, 12, tzinfo=timezone.utc), seed=3)
    m5 = fake._tf_bars(5)
    m5.index = (m5.index - pd.Timedelta(hours=7)).tz_localize(NY, ambiguous="NaT").tz_convert("UTC")
    m5 = m5[~m5.index.isna()]
    g = classify_gaps(m5, Timeframe.M5)
    assert len(g["weekend"]) >= 2 and not g["other"]                       # incl. the DST-end weekend
    hours = sorted({a.hour for a, _ in g["daily_break"]})
    assert hours == [21, 22]                                               # summer 21:00, winter 22:00 UTC
    # punch a 6h hole on a Wednesday and a 2h hole at the break time -> both are "other"
    wed = m5.index[(m5.index.dayofweek == 2) & (m5.index.hour == 10)][0]
    holed = m5[~((m5.index >= wed) & (m5.index < wed + timedelta(hours=6)))]
    brk = holed.index[(holed.index.dayofweek == 3) & (holed.index.hour == 20)][0]
    holed = holed[~((holed.index >= brk) & (holed.index < brk + timedelta(hours=2)))]
    g2 = classify_gaps(holed, Timeframe.M5)
    assert len(g2["other"]) == 2
    starts = sorted(a for a, _ in g2["other"])
    assert starts[0] == wed


def test_single_missing_candle_is_reported_as_gap():
    df = _bars("2026-11-04 08:00", "2026-11-04 12:00", Timeframe.M5)
    df = df.drop(df.index[10])                                              # one M5 candle missing
    g = classify_gaps(df, Timeframe.M5)
    assert len(g["other"]) == 1 and g["other"][0][0] == pd.Timestamp("2026-11-04 08:50", tz="UTC")


# ----------------------------------------------------------------------------- consistency across DST
def test_rebuilt_h4_d1_match_broker_across_dst():
    from integrations.mt5_connector import MT5DataProvider
    from config import MT5Config
    fake = FakeVantageMT5(days=25, end=datetime(2026, 11, 13, 12, tzinfo=timezone.utc), seed=5)
    p = MT5DataProvider(MT5Config(server_utc_offset="2", server_time_mode="ny_close"), mt5_module=fake)
    p.connect()
    a, b = datetime(2026, 10, 21, tzinfo=timezone.utc), datetime(2026, 11, 13, tzinfo=timezone.utc)
    m5 = p.get_rates_range("XAUUSD+", Timeframe.M5, a, b)
    for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
        broker = p.get_rates_range("XAUUSD+", tf, a, b)
        assert consistency(m5, broker, tf, "ny_close", 0.01) == 1.0, tf
    # UTC-aligned H4 bars never coincide with the broker's NY-close bars
    assert consistency(m5, p.get_rates_range("XAUUSD+", Timeframe.H4, a, b), Timeframe.H4, "utc", 0.01) == 0.0
