import numpy as np
import pandas as pd

from core.price_action import candle_patterns, detect_sweeps, list_swings, support_resistance
from core.smc_analysis import SMCAnalyzer


def path(points, bars_per_leg=8, wick=0.2):
    closes = []
    for a, b in zip(points[:-1], points[1:]):
        closes.extend(np.linspace(a, b, bars_per_leg, endpoint=False))
    closes.append(points[-1])
    closes = np.array(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + wick,
                         "low": np.minimum(opens, closes) - wick, "close": closes,
                         "tick_volume": 100.0, "atr": 1.0}, index=idx)


def test_bos_then_choch():
    # higher highs / higher lows, then a break of the last higher low
    df = path([100, 110, 105, 115, 108, 120, 112, 104])
    snap = SMCAnalyzer(swing_strength=3, window=1000).analyze(df)
    kinds = [(e.kind, e.direction) for e in snap.events]
    assert ("BOS", 1) in kinds
    assert kinds[-1] == ("CHOCH", -1)
    assert snap.trend == -1


def test_order_block_created_on_break():
    df = path([100, 110, 105, 116])
    snap = SMCAnalyzer(swing_strength=3, window=1000).analyze(df)
    bull_obs = [o for o in snap.order_blocks if o.direction > 0]
    assert bull_obs, "bullish OB expected after break of the swing high"
    ob = bull_obs[0]
    assert ob.bottom < 106 and ob.displacement_atr > 1.0


def test_fvg_detection_and_fill():
    closes = [100, 100.2, 100.1, 103, 103.2, 103.1, 103.0, 101.5, 100.5]
    opens = [100, 100, 100.2, 100.3, 103, 103.2, 103.1, 103, 101.5]
    highs = [100.3, 100.4, 100.4, 103.1, 103.4, 103.3, 103.2, 103.0, 101.6]
    lows = [99.8, 99.9, 99.9, 100.2, 102.9, 102.9, 102.8, 101.4, 100.2]
    idx = pd.date_range("2024-01-02", periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": 1.0,
                       "atr": 1.0}, index=idx)
    snap = SMCAnalyzer(swing_strength=2, window=1000, fvg_min_atr=0.1).analyze(df)
    bull = [g for g in snap.fvgs if g.direction > 0]
    assert bull, "gap between bar 2 high (100.4) and bar 4 low (102.9)"
    g = bull[0]
    assert g.bottom == 100.4 and g.top == 102.9
    assert g.filled_at is not None  # bar 8 low 100.2 < 100.4


def test_equal_highs_and_sweep():
    df = path([100, 110, 104, 110.05, 103, 111, 106], bars_per_leg=8)
    # make the last bar of the push above 110 close back below -> sweep
    snap = SMCAnalyzer(swing_strength=3, window=1000, equal_tol_atr=0.2).analyze(df)
    eq = [p for p in snap.pools if p.side == "buy_side" and p.is_equal]
    assert eq and abs(eq[0].price - 110.25) < 0.2
    assert eq[0].swept_at is not None


def test_sweep_detection_close_back_inside():
    df = path([100, 105, 102, 104], bars_per_leg=8)
    swings = list_swings(df, 3)
    low = [s for s in swings if s.kind == "low"][-1]
    extra = pd.DataFrame({"open": [104.0], "high": [104.2], "low": [low.price - 0.5], "close": [103.9],
                          "tick_volume": [100.0], "atr": [1.0]},
                         index=[df.index[-1] + pd.Timedelta(minutes=15)])
    df2 = pd.concat([df, extra])
    ev = detect_sweeps(df2, list_swings(df2, 3), lookback=2)
    assert any(e.direction == "bull" and abs(e.level - low.price) < 1e-9 for e in ev)


def test_premium_discount_position():
    df = path([100, 110, 105, 115, 108, 120, 111])
    snap = SMCAnalyzer(swing_strength=3, window=1000).analyze(df)
    assert snap.range_position is not None and 0 <= snap.range_position <= 1


def test_candle_patterns():
    df = pd.DataFrame({"open": [10.0, 9.5], "high": [10.1, 11.2], "low": [9.4, 9.3], "close": [9.5, 11.0]},
                      index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))
    p = candle_patterns(df)
    assert bool(p["bull_engulf"].iloc[-1]) and not bool(p["bear_engulf"].iloc[-1])
    pin = pd.DataFrame({"open": [10.0], "high": [10.15], "low": [8.0], "close": [10.1]},
                       index=pd.date_range("2024-01-01", periods=1, freq="5min", tz="UTC"))
    assert bool(candle_patterns(pin)["bull_pin"].iloc[0])


def test_support_resistance_clusters():
    df = path([100, 110, 104, 110, 104, 110, 107])
    lv = support_resistance(df, list_swings(df, 3), tol=0.5)
    res = [l for l in lv if l.kind == "resistance"]
    assert res and res[0].touches >= 2


def test_assess_rewards_valid_setup():
    # uptrend, pullback into discount, bullish structure
    df = path([100, 110, 104, 116, 108, 109.5])
    snap = SMCAnalyzer(swing_strength=3, window=1000).analyze(df)
    bull = snap.assess(1, lookback=12, min_displacement=1.0)
    bear = snap.assess(-1, lookback=12, min_displacement=1.0)
    assert bull.score > bear.score
