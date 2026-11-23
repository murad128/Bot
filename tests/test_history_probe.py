"""tools/history_probe.py against a mocked MetaTrader5 module (no terminal, no network)."""
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from config import MT5Config
from integrations.mt5_connector import MT5DataProvider, ReadOnlyMT5
from tools import history_probe as hp

TF_CODE = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408}
DT = np.dtype([("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
               ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])


def epoch(s):
    return int((datetime.fromisoformat(s) - datetime(1970, 1, 1)).total_seconds())


def server_bars(start, end, step, daily_break=True, holes=(), zero_spread_every=0):
    """Broker-like server-clock bars: Mon 00:00 - Fri 24:00, optional 00:00-01:00 daily break."""
    t = np.arange(epoch(start), epoch(end), step, dtype=np.int64)
    wd = ((t // 86400) + 3) % 7
    keep = wd < 5
    if daily_break and step < 3600 * 4:
        keep &= ((t % 86400) // 3600) != 0
    for a, b in holes:
        keep &= ~((t >= epoch(a)) & (t < epoch(b)))
    t = t[keep]
    arr = np.zeros(len(t), dtype=DT)
    arr["time"] = t
    arr["open"] = arr["high"] = arr["low"] = arr["close"] = 2650.0
    arr["spread"] = 25
    if zero_spread_every:
        arr["spread"][::zero_spread_every] = 0
    return arr


class MockMT5:
    __name__ = "MockMT5"
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30 = 1, 5, 15, 30
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408

    def __init__(self, data, trade_mode=0, errors=None, none_success=()):
        self.data = data                    # {(symbol, tf_code): structured array}
        self.trade_mode = trade_mode
        self.errors = errors or {}          # {(symbol, tf_code): (code, msg)} -> return None
        self.none_success = set(none_success)
        self._err = (1, "Success")
        self.calls = {"rates": 0, "order_send": 0}
        self.symbols = {s: types.SimpleNamespace(name=s, point=0.01, digits=2) for s, _ in data} | \
                       {s: types.SimpleNamespace(name=s, point=0.01, digits=2) for s, _ in self.errors}

    def initialize(self, **kw): return True
    def shutdown(self): pass
    def last_error(self): return self._err
    def terminal_info(self): return types.SimpleNamespace(connected=True, build=4755, trade_allowed=False)

    def account_info(self):
        return types.SimpleNamespace(login=24681357, server="VantageInternational-Demo", trade_mode=self.trade_mode,
                                     company="Vantage", currency="USD", balance=1e4, equity=1e4, leverage=500,
                                     name="HOLDER NAME MUST NOT APPEAR")

    def symbol_info(self, name): return self.symbols.get(name)
    def symbol_select(self, name, enable): return True
    def symbols_get(self, pattern="*"): return []

    def symbol_info_tick(self, name):
        return types.SimpleNamespace(time=0, bid=1.0, ask=1.1)          # stale -> offset from config

    def copy_rates_from_pos(self, symbol, tf, start, count):
        self.calls["rates"] += 1
        key = (symbol, tf)
        if key in self.errors:
            self._err = self.errors[key]
            return None
        if key in self.none_success:
            self._err = (1, "Success")
            return None
        arr = self.data.get(key)
        if arr is None:
            self._err = (-1, "Terminal: Call failed")
            return None
        n = len(arr)
        if n == 0:                          # symbol/timeframe exists but has no bars
            self._err = (1, "Success")
            return arr
        hi = n - start
        if hi <= 0:
            self._err = (-1, "Terminal: Invalid params")
            return None
        self._err = (1, "Success")
        return arr[max(0, hi - count):hi]

    def order_send(self, *a, **k):          # must stay unreachable through ReadOnlyMT5
        self.calls["order_send"] += 1
        raise AssertionError("order_send called")


@pytest.fixture
def cfg_probe(cfg):
    cfg.mt5.server_utc_offset = "2"
    cfg.mt5.max_reconnect_attempts = 1
    cfg.instruments = {k: cfg.instrument(k) for k in ("XAUUSD", "EURUSD")}
    return cfg


def _run(cfg, mock, **kw):
    prov = MT5DataProvider(cfg.mt5, mt5_module=mock)
    return hp.run_probe(cfg, provider=prov, **kw)


# ----------------------------------------------------------------------------- counts and dates
def test_exact_counts_dates_and_spread_coverage(cfg_probe):
    m5 = server_bars("2026-10-05 00:00", "2026-11-07 00:00", 300, zero_spread_every=4)   # 25% without spread
    mock = MockMT5({("XAUUSD", 5): m5})
    code, header, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], chunk=1000)
    r = res[0]
    assert code == 0 and r.status == "OK"
    assert r.bars == len(m5)
    assert r.first == int(m5["time"][0]) and r.last == int(m5["time"][-1])
    assert r.spread_coverage == pytest.approx(0.75, abs=0.001)
    assert r.gaps["weekend"] == 4 and r.gaps["daily_break"] == 25 - 5 and r.gaps["unexpected"] == 0
    # every batch is read, then one more call confirms the end (a short batch does not end paging)
    assert mock.calls["rates"] == -(-len(m5) // 1000) + 1


def test_weekend_and_daily_break_are_expected_for_every_timeframe(cfg_probe):
    data = {("XAUUSD", TF_CODE[tf]): server_bars("2026-10-05 00:00", "2026-10-31 00:00", hp.TF_SECONDS[tf])
            for tf in hp.TIMEFRAMES}
    code, _, res = _run(cfg_probe, MockMT5(data), symbols=["XAUUSD"], chunk=500)
    assert code == 0 and len(res) == 7
    for r in res:
        assert r.status == "OK" and r.gaps["unexpected"] == 0, (r.timeframe, r.gaps)
        assert r.gaps["weekend"] == 3, (r.timeframe, r.gaps)


def test_midweek_hole_is_unexpected_also_across_chunk_boundary(cfg_probe):
    m5 = server_bars("2026-10-05 00:00", "2026-10-24 00:00", 300, holes=[("2026-10-14 10:00", "2026-10-14 16:00")])
    for chunk in (100_000, 37):                   # one chunk / hole split across many chunk boundaries
        code, _, res = _run(cfg_probe, MockMT5({("XAUUSD", 5): m5}), symbols=["XAUUSD"], timeframes=["M5"], chunk=chunk)
        r = res[0]
        assert r.gaps["unexpected"] == 1, (chunk, r.gaps)
        g = r.largest_unexpected[0]
        assert g.start == epoch("2026-10-14 10:00") and g.end == epoch("2026-10-14 16:00")
        assert r.bars == len(m5)


def test_cap_is_reported(cfg_probe):
    m5 = server_bars("2026-10-05 00:00", "2026-10-24 00:00", 300)
    _, _, res = _run(cfg_probe, MockMT5({("XAUUSD", 5): m5}), symbols=["XAUUSD"], timeframes=["M5"],
                     chunk=100, max_bars=300)
    assert res[0].status == "LIMITED" and res[0].bars == 300 and "bar limit 300" in res[0].error
    text = hp.format_report(3, {}, res)
    assert "| LIMITED |" in text and "**SCAN INCOMPLETE**" in text and "*" in text


# ----------------------------------------------------------------------------- unavailable vs zero bars
def test_unavailable_no_bars_and_not_found_are_distinguished(cfg_probe):
    cfg_probe.instruments = {k: cfg_probe.instrument(k) for k in ("XAUUSD", "EURUSD", "GBPJPY")}
    empty = np.zeros(0, dtype=DT)
    mock = MockMT5({("XAUUSD", 5): empty, ("XAUUSD", 15): server_bars("2026-10-05", "2026-10-10", 900)},
                   errors={("EURUSD", 5): (-10004, "No IPC connection")}, none_success={("XAUUSD", 16385)})
    code, header, res = _run(cfg_probe, mock, timeframes=["M5", "M15", "H1"])
    st = {(r.symbol, r.timeframe): r for r in res}
    assert st[("XAUUSD", "M5")].status == "NO BARS"                    # empty array
    assert st[("XAUUSD", "H1")].status == "NO BARS"                    # None + success code
    assert st[("XAUUSD", "M15")].status == "OK"
    assert st[("EURUSD", "M5")].status == "UNAVAILABLE"
    assert "-10004" in st[("EURUSD", "M5")].error
    assert st[("EURUSD", "M15")].status == "UNAVAILABLE"              # no data registered -> call failed
    assert st[("GBPJPY", "-")].status == "NOT FOUND"
    text = hp.format_report(code, header, res)
    assert "| XAUUSD | M5 | NO BARS |" in text and "request succeeded, 0 bars" in text
    assert "| EURUSD | M5 | UNAVAILABLE |" in text and "| GBPJPY | - | NOT FOUND |" in text
    assert "1 not found" in text and "**SCAN INCOMPLETE**" in text        # unavailable rows: not complete


# ----------------------------------------------------------------------------- safety
def test_read_only_wrapper_and_no_orders(cfg_probe):
    mock = MockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-10", 300)})
    prov = MT5DataProvider(cfg_probe.mt5, mt5_module=mock)
    assert isinstance(prov.mt5, ReadOnlyMT5)
    hp.run_probe(cfg_probe, provider=prov, symbols=["XAUUSD"], timeframes=["M5"])
    with pytest.raises(PermissionError):
        prov.mt5.order_send
    assert mock.calls["order_send"] == 0
    src = Path(hp.__file__).read_text(encoding="utf-8")
    for fn in ("order_send(", "order_check(", "positions_get(", "orders_get(", "order_calc_margin(", "history_deals_get("):
        assert fn not in src


def test_demo_only_guard_refuses_real_account_before_any_data(cfg_probe):
    mock = MockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-10", 300)}, trade_mode=2)
    code, header, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"])
    assert code == 2 and res == []
    assert "REAL account" in header["error"] and mock.calls["rates"] == 0
    assert "FAILED" in hp.format_report(code, header, res)


def test_report_masks_login_and_hides_secrets(cfg_probe):
    from core.redact import register_secrets
    cfg_probe.mt5.password = "MyMt5Pass99"
    cfg_probe.telegram.bot_token = "123456789:FAKEfakeFAKEfakeFAKEfakeFAKEfake00"
    register_secrets([cfg_probe.mt5.password, cfg_probe.telegram.bot_token])
    mock = MockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-10", 300)})
    code, header, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"])
    header["error"] = f"debug {cfg_probe.mt5.password} {cfg_probe.telegram.bot_token}"   # worst case: leaked into text
    text = hp.format_report(code, header, res) + hp.format_report(2, header, [])
    assert "24681357" not in text and "*****357" in text
    assert "HOLDER NAME" not in text and "MyMt5Pass99" not in text and "FAKEfake" not in text


# ----------------------------------------------------------------------------- CLI
def test_cli_filters_and_out_file(cfg_probe, tmp_path, monkeypatch):
    mock = MockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-10", 300),
                    ("XAUUSD", 16385): server_bars("2026-10-05", "2026-10-10", 3600)})
    real = hp.run_probe
    monkeypatch.setattr(hp, "run_probe", lambda cfg, **kw: real(
        cfg_probe, provider=MT5DataProvider(cfg_probe.mt5, mt5_module=mock), **kw))
    out = tmp_path / "probe.md"
    code = hp.main(["--symbols", "xauusd", "--timeframes", "m5,h1", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert code == 0 and "| XAUUSD | M5 | OK |" in text and "| XAUUSD | H1 | OK |" in text and "| M15 |" not in text
    with pytest.raises(SystemExit):
        hp.main(["--timeframes", "M2"])


@pytest.mark.parametrize("a,b,step,kind", [
    ("2026-10-09 23:55", "2026-10-12 01:00", 300, "weekend"),        # Fri last bar -> Mon after break
    ("2026-10-09 00:00", "2026-10-12 00:00", 86400, "weekend"),      # D1: Friday bar -> Monday bar
    ("2026-10-12 23:55", "2026-10-13 01:00", 300, "daily_break"),    # gold break 00:00-01:00 server
    ("2026-10-12 23:00", "2026-10-13 01:00", 3600, "daily_break"),   # H1: the 00:00 bar missing
    ("2026-10-13 10:00", "2026-10-13 12:00", 300, "unexpected"),     # mid-week hole
    ("2026-12-24 23:55", "2026-12-28 01:00", 300, "unexpected"),     # Thu -> Mon (holiday): flagged
])
def test_classify_gaps_rules(a, b, step, kind):
    gaps = hp.classify_gaps(np.array([epoch(a), epoch(b)]), step)
    assert [g.kind for g in gaps] == [kind]
    assert gaps[0].start == epoch(a) + step and gaps[0].end == epoch(b)


def test_contiguous_bars_have_no_gaps():
    t = np.arange(epoch("2026-10-13 10:00"), epoch("2026-10-13 12:00"), 300)
    assert hp.classify_gaps(t, 300) == []


# ----------------------------------------------------------------------------- regression: TypeError / paging
class StrictMockMT5(MockMT5):
    """Mimics the compiled MetaTrader5 module (5.0.6231): positional only, exact built-in types.
    The error texts are the ones embedded in MetaTrader5/_core.pyd."""

    def __init__(self, *a, max_per_call=None, raise_type_error_at=None, new_bar_after_first_call=False, **kw):
        super().__init__(*a, **kw)
        self.max_per_call = max_per_call
        self.raise_type_error_at = raise_type_error_at      # start_pos at which the package raises
        self.new_bar_after_first_call = new_bar_after_first_call
        self.requests = []

    def copy_rates_from_pos(self, *args, **kwargs):
        if kwargs:
            raise TypeError("Unnamed arguments not allowed")
        if len(args) != 4:
            raise TypeError("Invalid number of arguments")
        symbol, tf, start, count = args
        if type(symbol) is not str:
            raise TypeError("Invalid 1st unnamed argument")
        if type(tf) is not int:
            raise TypeError("Invalid 2nd unnamed argument")
        if type(start) is not int:
            raise TypeError("Invalid 3rd unnamed argument")
        if type(count) is not int:
            raise TypeError("Invalid arguments")
        if self.raise_type_error_at is not None and start >= self.raise_type_error_at:
            raise TypeError("Invalid 3rd unnamed argument")
        self.requests.append((start, count))
        out = super().copy_rates_from_pos(symbol, tf, start, count if not self.max_per_call
                                          else min(count, self.max_per_call))
        if self.new_bar_after_first_call and len(self.requests) == 1:
            key = (symbol, tf)
            arr = self.data[key]
            new = arr[-1:].copy()
            new["time"] = arr["time"][-1] + 300          # a new bar forms: every position shifts by one
            self.data[key] = np.concatenate([arr, new])
        return out


def test_numpy_integer_arguments_are_converted_before_the_call(cfg_probe):
    """numpy integers are not accepted by the compiled module's strict checks -> converted to int first."""
    m5 = server_bars("2026-10-05", "2026-10-17", 300)
    mock = StrictMockMT5({("XAUUSD", 5): m5})
    code, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"],
                        chunk=np.int64(1000), max_bars=np.int64(10_000_000))
    assert code == 0 and res[0].status == "OK" and res[0].bars == len(m5)
    assert all(type(s) is int and type(c) is int for s, c in mock.requests)


def test_strict_mock_rejects_wrong_types_like_the_real_package():
    mock = StrictMockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-06", 300)})
    with pytest.raises(TypeError, match="Invalid 3rd unnamed argument"):
        mock.copy_rates_from_pos("XAUUSD", 5, np.int64(0), 10)
    with pytest.raises(TypeError, match="Unnamed arguments not allowed"):
        mock.copy_rates_from_pos("XAUUSD", 5, 0, count=10)


def test_type_error_is_reported_not_raised_and_other_timeframes_continue(cfg_probe):
    data = {("XAUUSD", 5): server_bars("2026-10-05", "2026-10-10", 300),
            ("XAUUSD", 16385): server_bars("2026-10-05", "2026-10-10", 3600)}
    mock = StrictMockMT5(data, raise_type_error_at=0)                  # the package rejects every call
    code, header, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5", "H1"])
    assert code == 3 and [r.status for r in res] == ["UNAVAILABLE", "UNAVAILABLE"]   # 3 = incomplete scan
    assert "TypeError: Invalid 3rd unnamed argument" in res[0].error
    assert "argument types: str, int, int, int" in res[0].error
    text = hp.format_report(code, header, res)
    assert "| XAUUSD | M5 | UNAVAILABLE |" in text and "Invalid 3rd unnamed argument" in text


def test_type_error_mid_paging_reports_incomplete_with_bars_so_far(cfg_probe):
    m5 = server_bars("2026-10-05", "2026-10-17", 300)
    mock = StrictMockMT5({("XAUUSD", 5): m5}, raise_type_error_at=2000)
    _, header, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], chunk=1000)
    r = res[0]
    assert r.status == "INCOMPLETE" and r.bars == 2000 and "TypeError" in r.error
    assert r.last == int(m5["time"][-1]) and r.first == int(m5["time"][-2000])
    assert "| XAUUSD | M5 | INCOMPLETE |" in hp.format_report(0, header, res)


@pytest.mark.parametrize("max_per_call", [999, 300, 1])
def test_short_batches_do_not_end_paging(cfg_probe, max_per_call):
    """A terminal returning fewer bars than requested must not hide older history."""
    m5 = server_bars("2026-10-05", "2026-10-09", 300)
    mock = StrictMockMT5({("XAUUSD", 5): m5}, max_per_call=max_per_call)
    _, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], chunk=1000)
    r = res[0]
    assert r.status == "OK" and r.bars == len(m5)
    assert r.first == int(m5["time"][0]) and r.last == int(m5["time"][-1])
    starts = [s for s, _ in mock.requests]
    assert starts == sorted(set(starts)) and starts[0] == 0                # every position requested once, in order


def test_new_bar_during_paging_is_not_double_counted(cfg_probe):
    m5 = server_bars("2026-10-05", "2026-10-09", 300)
    original = len(m5)
    mock = StrictMockMT5({("XAUUSD", 5): m5}, new_bar_after_first_call=True)
    _, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], chunk=100)
    r = res[0]
    assert r.bars == original                      # the boundary bar is not counted twice
    assert r.first == int(m5["time"][0])
    assert r.gaps["unexpected"] == 0


def test_paging_counts_every_bar_exactly_once_across_many_chunks(cfg_probe):
    m5 = server_bars("2026-09-07", "2026-11-07", 300, holes=[("2026-10-14 10:00", "2026-10-14 16:00")])
    for chunk in (7, 1000, 100_000):
        mock = StrictMockMT5({("XAUUSD", 5): m5})
        _, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], chunk=chunk)
        r = res[0]
        assert r.bars == len(m5) and r.gaps["unexpected"] == 1, chunk
        assert r.first == int(m5["time"][0]) and r.last == int(m5["time"][-1]), chunk


# ----------------------------------------------------------------------------- batching, progress, limits
class FakeClock:
    """Deterministic clock: TimedMockMT5 advances it by `per_call` seconds per MT5 request."""

    def __init__(self, per_call=1.0):
        self.t, self.per_call = 0.0, per_call

    def __call__(self):
        return self.t


class TimedMockMT5(StrictMockMT5):
    def __init__(self, *a, clock=None, interrupt_on_call=None, **kw):
        super().__init__(*a, **kw)
        self.clock, self.interrupt_on_call = clock, interrupt_on_call

    def copy_rates_from_pos(self, *args, **kwargs):
        if self.interrupt_on_call is not None and len(self.requests) + 1 == self.interrupt_on_call:
            raise KeyboardInterrupt
        out = super().copy_rates_from_pos(*args, **kwargs)
        if self.clock is not None:
            self.clock.t += self.clock.per_call
        return out


def _limits(**kw):
    kw.setdefault("max_seconds", 0)
    return hp.Limits(**kw)


def test_batches_use_the_configured_chunk_and_stop_exactly_at_the_bar_limit(cfg_probe):
    m5 = server_bars("2026-10-05", "2026-10-17", 300)
    mock = StrictMockMT5({("XAUUSD", 5): m5})
    _, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"],
                     limits=_limits(chunk=700, max_bars=2_000))
    r = res[0]
    assert [c for _, c in mock.requests] == [700, 700, 600]               # last request trimmed to the limit
    assert [s for s, _ in mock.requests] == [0, 700, 1400]
    assert r.status == "LIMITED" and r.bars == 2_000 and r.batches == 3
    assert r.last == int(m5["time"][-1]) and r.first == int(m5["time"][-2000])


def test_default_chunk_and_time_limit_are_practical():
    assert hp.Limits().chunk == 20_000 and hp.Limits().max_seconds == 300


def test_progress_after_every_successful_batch(cfg_probe):
    m5 = server_bars("2026-10-05", "2026-10-09", 300)
    clock = FakeClock(per_call=2.0)
    mock = TimedMockMT5({("XAUUSD", 5): m5}, clock=clock)
    events = []
    _, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"],
                     limits=_limits(chunk=400), progress=events.append, clock=clock)
    n = -(-len(m5) // 400)
    assert len(events) == res[0].batches == n
    assert [e["added"] for e in events] == [400] * (n - 1) + [len(m5) - 400 * (n - 1)]
    assert [e["total"] for e in events] == [min(400 * (i + 1), len(m5)) for i in range(n)]
    assert all(e["symbol"] == "XAUUSD" and e["timeframe"] == "M5" for e in events)
    assert all(e["batch_seconds"] == pytest.approx(2.0) for e in events)
    assert [e["elapsed"] for e in events] == pytest.approx([2.0 * (i + 1) for i in range(n)])
    assert events[-1]["oldest"] == int(m5["time"][0])


def test_progress_line_format(capsys):
    hp.stderr_progress({"symbol": "XAUUSD", "timeframe": "M1", "added": 20000, "total": 140000,
                        "oldest": epoch("2026-01-05 10:00"), "batch_seconds": 1.25, "elapsed": 63.4})
    err = capsys.readouterr().err.strip()
    assert err == "[   63.4s] XAUUSD M1: +20,000 bars (total 140,000, oldest 2026-01-05 10:00) batch 1.2s"


def test_time_limit_per_timeframe_marks_limited_with_exact_partial_counts(cfg_probe):
    m5 = server_bars("2026-10-05", "2026-10-17", 300)
    clock = FakeClock(per_call=10.0)
    data = {("XAUUSD", 5): m5, ("XAUUSD", 16385): server_bars("2026-10-05", "2026-10-17", 3600)}
    code, header, res = _run(cfg_probe, TimedMockMT5(data, clock=clock), symbols=["XAUUSD"],
                             timeframes=["M5", "H1"], limits=_limits(chunk=500, max_seconds=25), clock=clock)
    m5r, h1r = res
    assert m5r.status == "LIMITED" and m5r.bars == 1500 and "time limit 25s" in m5r.error
    assert m5r.first == int(m5["time"][-1500])          # exactly the newest 1500 bars: none skipped, none doubled
    assert h1r.status == "OK"                            # the limit applies per timeframe
    assert code == 3
    text = hp.format_report(code, header, res)
    assert "**SCAN INCOMPLETE**" in text and "1 limited" in text
    assert f"| XAUUSD | M5 | LIMITED | {hp._d(m5r.first)}* |" in text


def test_total_time_limit_leaves_remaining_rows_not_scanned(cfg_probe):
    data = {("XAUUSD", TF_CODE[tf]): server_bars("2026-10-05", "2026-10-09", hp.TF_SECONDS[tf])
            for tf in ("M5", "M15", "H1")}
    clock = FakeClock(per_call=6.0)     # M5: data + end check = 12s; M15 starts at 12s, stopped after 18s
    code, header, res = _run(cfg_probe, TimedMockMT5(data, clock=clock), symbols=["XAUUSD"],
                             timeframes=["M5", "M15", "H1"],
                             limits=_limits(chunk=100_000, max_total_seconds=15), clock=clock)
    assert [r.status for r in res] == ["OK", "LIMITED", "NOT SCANNED"]
    assert code == 3 and "1 not scanned" in hp.format_report(code, header, res)


def test_per_timeframe_bar_limits(cfg_probe):
    data = {("XAUUSD", 1): server_bars("2026-10-05", "2026-10-07", 60),
            ("XAUUSD", 16385): server_bars("2026-10-05", "2026-10-07", 3600)}
    _, _, res = _run(cfg_probe, StrictMockMT5(data), symbols=["XAUUSD"], timeframes=["M1", "H1"],
                     limits=_limits(chunk=1000, max_bars=10, max_bars_tf={"M1": 1500}))
    m1, h1 = res
    assert m1.status == "LIMITED" and m1.bars == 1500          # per-timeframe override
    assert h1.status == "LIMITED" and h1.bars == 10            # global per symbol/timeframe limit
    assert hp._parse_tf_limits("m1=200_000, H4=0") == {"M1": 200_000, "H4": 0}
    for bad in ("M2=5", "M1", "M1=-1"):
        with pytest.raises(ValueError):
            hp._parse_tf_limits(bad)


def test_ctrl_c_mid_paging_reports_interrupted_and_not_scanned(cfg_probe):
    data = {("XAUUSD", 5): server_bars("2026-10-05", "2026-10-17", 300),
            ("XAUUSD", 16385): server_bars("2026-10-05", "2026-10-17", 3600)}
    code, header, res = _run(cfg_probe, TimedMockMT5(data, interrupt_on_call=4), symbols=["XAUUSD", "EURUSD"],
                             timeframes=["M5", "H1"], limits=_limits(chunk=500))
    assert code == 130 and header["interrupted"]
    assert [r.status for r in res] == ["INTERRUPTED", "NOT SCANNED", "NOT SCANNED", "NOT SCANNED"]
    r = res[0]
    assert r.bars == 1500 and r.first == int(data[("XAUUSD", 5)]["time"][-1500])
    text = hp.format_report(code, header, res)
    assert "**SCAN INCOMPLETE**" in text and "**SCAN COMPLETE**" not in text
    assert "| XAUUSD | M5 | INTERRUPTED |" in text and "stopped by Ctrl+C" in text and "3 not scanned" in text


def test_ctrl_c_report_is_still_written_by_the_cli(cfg_probe, tmp_path, monkeypatch):
    mock = TimedMockMT5({("XAUUSD", 5): server_bars("2026-10-05", "2026-10-17", 300)}, interrupt_on_call=2)
    real = hp.run_probe
    monkeypatch.setattr(hp, "run_probe", lambda cfg, **kw: real(
        cfg_probe, provider=MT5DataProvider(cfg_probe.mt5, mt5_module=mock), **kw))
    out = tmp_path / "partial.md"
    code = hp.main(["--symbols", "XAUUSD", "--timeframes", "M5", "--chunk", "500", "--quiet", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert code == 130 and "INTERRUPTED" in text and "**SCAN INCOMPLETE**" in text


def test_terminal_that_re_serves_oldest_bars_does_not_loop_forever(cfg_probe):
    """Reproduces the never-ending scan: batches that add no new bars must end paging."""

    class ClippingMT5(StrictMockMT5):
        def copy_rates_from_pos(self, *args, **kwargs):
            if len(self.requests) > 200:
                raise AssertionError("probe is looping")
            symbol, tf, start, count = args
            self.requests.append((start, count))
            arr = self.data[(symbol, tf)]
            hi = max(len(arr) - start, min(count, len(arr)))     # beyond the oldest bar: oldest bars again
            return arr[max(0, hi - count):hi]

    m5 = server_bars("2026-10-05", "2026-10-09", 300)
    mock = ClippingMT5({("XAUUSD", 5): m5})
    code, _, res = _run(cfg_probe, mock, symbols=["XAUUSD"], timeframes=["M5"], limits=_limits(chunk=200))
    r = res[0]
    assert r.status == "OK" and r.bars == len(m5) and "repeated" in r.note
    assert len(mock.requests) == -(-len(m5) // 200) + 3                # the data, then 3 repeats -> stop


def test_complete_scan_says_complete_and_exits_zero(cfg_probe):
    data = {("XAUUSD", 5): server_bars("2026-10-05", "2026-10-09", 300)}
    code, header, res = _run(cfg_probe, StrictMockMT5(data), symbols=["XAUUSD"], timeframes=["M5"],
                             limits=_limits(chunk=300))
    text = hp.format_report(code, header, res)
    table = text.split("| Note |")[1].split("Summary")[0]
    assert code == 0 and "**SCAN COMPLETE**" in text and "*" not in table


def test_cli_rejects_invalid_limits():
    for argv in (["--chunk", "0"], ["--max-seconds", "-1"], ["--max-bars-tf", "M9=10"]):
        with pytest.raises(SystemExit):
            hp.main(argv)
