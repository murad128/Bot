from datetime import datetime, timedelta, timezone

from core.domain import Direction
from core.fundamental_analysis import FundamentalAnalyzer
from integrations.economic_calendar import CSVCalendar, EconomicCalendar, ForexFactoryCalendar, NullCalendar
from integrations.macro_data import DxyReading, YieldReading

NOW = datetime(2024, 3, 8, 13, 0, tzinfo=timezone.utc)


def write_cal(tmp_path, rows):
    p = tmp_path / "cal.csv"
    p.write_text("time_utc,currency,impact,title,forecast,previous\n" + "\n".join(rows) + "\n")
    cal = EconomicCalendar(CSVCalendar(str(p)), max_staleness_hours=1e9)
    assert cal.refresh_if_needed(force=True)
    return cal


def test_blackout_high_impact(tmp_path, cfg, eur_spec):
    cal = write_cal(tmp_path, ["2024-03-08T13:20:00Z,EUR,High,ECB Press Conference,,"])
    fa = FundamentalAnalyzer(cfg.news, cal)
    a = fa.assess(eur_spec, Direction.BUY, NOW)
    assert a.blocked and "ECB" in a.block_reason
    # 31 minutes before -> outside default 30 min window
    assert not fa.assess(eur_spec, Direction.BUY, NOW - timedelta(minutes=11)).blocked


def test_gold_tier1_wider_window(tmp_path, cfg, gold_spec, eur_spec):
    cal = write_cal(tmp_path, ["2024-03-08T13:40:00Z,USD,High,Non-Farm Employment Change,200K,180K"])
    fa = FundamentalAnalyzer(cfg.news, cal)
    assert fa.assess(gold_spec, Direction.BUY, NOW).blocked           # 40 min < 45 gold window
    assert not fa.assess(eur_spec, Direction.BUY, NOW).blocked        # 40 min > 30 default


def test_other_currency_and_low_impact_ignored(tmp_path, cfg, eur_spec):
    cal = write_cal(tmp_path, ["2024-03-08T13:10:00Z,JPY,High,BoJ,,", "2024-03-08T13:05:00Z,EUR,Low,Speech,,"])
    a = FundamentalAnalyzer(cfg.news, cal).assess(eur_spec, Direction.SELL, NOW)
    assert not a.blocked


def test_post_event_window(tmp_path, cfg, eur_spec):
    cal = write_cal(tmp_path, ["2024-03-08T12:40:00Z,USD,High,CPI m/m,,"])
    assert FundamentalAnalyzer(cfg.news, cal).assess(eur_spec, Direction.BUY, NOW).blocked


def test_unavailable_calendar_blocks_by_default(cfg, eur_spec):
    cal = EconomicCalendar(NullCalendar())
    assert FundamentalAnalyzer(cfg.news, cal).assess(eur_spec, Direction.BUY, NOW).blocked
    a = FundamentalAnalyzer(cfg.news, cal, block_if_unavailable=False).assess(eur_spec, Direction.BUY, NOW)
    assert not a.blocked and any("unavailable" in w for w in a.warnings)


class FakeMacro:
    def __init__(self, dxy_trend, yld_chg):
        self._d, self._y = dxy_trend, yld_chg

    def dxy(self):
        return None if self._d is None else DxyReading("DXY", self._d, 0.5, NOW)

    def yields(self):
        return {"DFII10": None if self._y is None else YieldReading("DFII10", 2.0, self._y, "2024-03-07")}


def test_gold_macro_conflict_and_support(tmp_path, cfg, gold_spec):
    cal = write_cal(tmp_path, ["2024-03-11T13:00:00Z,USD,High,CPI m/m,,"])
    conflict = FundamentalAnalyzer(cfg.news, cal, macro=FakeMacro(+1, +15)).assess(gold_spec, Direction.BUY, NOW)
    support = FundamentalAnalyzer(cfg.news, cal, macro=FakeMacro(-1, -15)).assess(gold_spec, Direction.BUY, NOW)
    assert conflict.conflict and any("conflict" in w.lower() for w in conflict.warnings)
    assert support.score > conflict.score and not support.conflict
    missing = FundamentalAnalyzer(cfg.news, cal, macro=FakeMacro(None, None)).assess(gold_spec, Direction.BUY, NOW)
    assert any("DXY data unavailable" in w for w in missing.warnings)


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Sess:
    def __init__(self, resp):
        self.resp = resp

    def get(self, *a, **k):
        return self.resp


def test_forexfactory_parsing_to_utc():
    payload = [{"title": "CPI m/m", "country": "USD", "date": "2024-03-12T08:30:00-04:00", "impact": "High",
                "forecast": "0.4%", "previous": "0.3%"}, {"bad": "row"}]
    ev = ForexFactoryCalendar(session=_Sess(_Resp(200, payload))).fetch()
    assert len(ev) == 1 and ev[0].time == datetime(2024, 3, 12, 12, 30, tzinfo=timezone.utc) and ev[0].is_tier1


def test_calendar_failure_keeps_cache_and_reports(tmp_path):
    prov = ForexFactoryCalendar(session=_Sess(_Resp(429, {})))
    cal = EconomicCalendar(prov, cache_path=str(tmp_path / "c.json"))
    assert not cal.refresh_if_needed(force=True)
    assert not cal.available and "429" in cal.last_error
