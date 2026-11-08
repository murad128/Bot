from datetime import datetime, timedelta, timezone

import pytest

from config import RiskConfig
from core.domain import Direction
from core.performance_analyzer import breakdown, compute_stats
from core.risk_management import PlanRequest, RiskManager
from core.trade_review import TradeReviewer, review_signal, wilson

NOW = datetime(2024, 3, 4, 10, 0, tzinfo=timezone.utc)


def test_plan_structural_stop_and_targets(gold_spec):
    rm = RiskManager(RiskConfig())
    plan, rej = rm.build_plan(PlanRequest(Direction.BUY, 2000.0, atr=2.0, invalidation=1998.0, spread=0.2), gold_spec)
    assert not rej
    # SL = 1998 - (0.25*2 + 0.2) = 1997.3 -> distance 2.7 (>= 1 ATR)
    assert plan.stop_loss == pytest.approx(1997.3)
    assert plan.take_profits == pytest.approx([2002.7, 2005.4, 2008.1])
    assert plan.primary_rr == pytest.approx(2.0)


def test_plan_min_distance_and_too_wide(gold_spec):
    rm = RiskManager(RiskConfig())
    plan, _ = rm.build_plan(PlanRequest(Direction.SELL, 2000.0, 2.0, 2000.1, 0.0), gold_spec)
    assert plan.stop_loss == pytest.approx(2002.0)       # widened to 1 ATR
    plan, rej = rm.build_plan(PlanRequest(Direction.BUY, 2000.0, 2.0, 1990.0, 0.0), gold_spec)
    assert plan is None and "too wide" in rej[0]


def test_plan_obstacle_and_spread_rejections(gold_spec):
    rm = RiskManager(RiskConfig())
    plan, rej = rm.build_plan(PlanRequest(Direction.BUY, 2000.0, 2.0, 1998.5, 0.0, liquidity_targets=[2003.0]), gold_spec)
    assert plan is None and "Opposing liquidity" in rej[0]
    plan, rej = rm.build_plan(PlanRequest(Direction.BUY, 2000.0, 2.0, 1998.5, 0.5), gold_spec)
    assert plan is None and any("Spread" in r for r in rej)


def test_tp3_extends_to_liquidity(gold_spec):
    rm = RiskManager(RiskConfig())
    plan, _ = rm.build_plan(PlanRequest(Direction.BUY, 2000.0, 2.0, 1998.0, 0.0, liquidity_targets=[2010.0]), gold_spec)
    assert plan.take_profits[-1] == pytest.approx(2009.8)


def test_portfolio_limits():
    rm = RiskManager(RiskConfig(max_signals_per_day=2, max_consecutive_losses=2, max_active_per_currency=1))
    todays = [{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}]
    active = [{"symbol": "EURUSD", "currencies": ["EUR", "USD"]}]
    out = rm.check_limits("XAUUSD", ["USD"], NOW, todays, active, 2, NOW - timedelta(minutes=5))
    text = " | ".join(out)
    assert "Daily signal limit" in text and "consecutive losses" in text and "exposure" in text and "cooldown" in text
    assert rm.check_limits("XAUUSD", ["USD"], NOW, [], [], 0, None) == []


def rec(outcome, r, hits=0, status="STOPPED", sym="XAUUSD", sess="London", day=0, rr=2.0):
    t = NOW + timedelta(hours=day)
    return {"outcome": outcome, "realized_r": r, "tp_hits": hits, "status": status, "symbol": sym,
            "session_bucket": sess, "created_at": t, "closed_at": t + timedelta(hours=1), "rr": rr,
            "entry_timeframe": "M15", "confirm_timeframe": "H1"}


def test_stats_definitions():
    rows = [
        rec("WIN", 1.4, hits=3, status="TP3_HIT", day=0),
        rec("WIN", 0.5, hits=1, status="STOPPED", day=1),       # TP1 then BE
        rec("LOSS", -1.0, day=2),
        rec("LOSS", -1.0, day=3),
        rec("CANCELLED", -0.2, status="CANCELLED", day=4),
        rec("OPEN", 0.0, status="ACTIVE", day=5),
    ]
    s = compute_stats(rows)
    assert s.total == 6 and s.open == 1 and s.cancelled == 1
    assert s.resolved == 4 and s.wins == 2 and s.losses == 2
    assert s.win_rate == pytest.approx(0.5)          # cancelled excluded
    assert s.tp1 == 2 and s.tp2 == 1 and s.tp3 == 1  # milestones, not wins
    assert s.sl == 2
    assert s.net_r == pytest.approx(-0.3)
    assert s.profit_factor == pytest.approx(1.9 / 2.2)
    assert s.max_drawdown_r == pytest.approx(2.2)     # peak 1.9 -> trough -0.3
    assert s.expectancy_r == pytest.approx(-0.3 / 5)


def test_breakdown_by_symbol():
    rows = [rec("WIN", 1.0, 1, sym="EURUSD"), rec("LOSS", -1.0, sym="XAUUSD")]
    b = breakdown(rows, "symbol")
    assert b["EURUSD"].wins == 1 and b["XAUUSD"].losses == 1


def test_review_signal_and_reviewer_gate():
    note = review_signal({"outcome": "LOSS", "mfe_r": 0.1, "mae_r": -1, "tp_hits": 0, "status": "STOPPED"})
    assert "Immediate adverse move" in note["notes"][0]
    findings = TradeReviewer(min_samples=30).analyze([rec("WIN", 1.0, 1)] * 5)
    assert findings[0].category == "info" and "at least 30" in findings[0].text


def test_reviewer_finds_consistent_weak_tag():
    rows = []
    for i in range(200):
        bad = i % 2 == 0
        outcome, r = (("LOSS", -1.0) if i % 4 != 2 else ("WIN", 1.0)) if bad else (("WIN", 1.0) if i % 3 else ("LOSS", -1.0))
        d = rec(outcome, r, hits=1 if outcome == "WIN" else 0, day=i)
        d["context"] = {"tags": ["weak_tag"] if bad else ["good_tag"], "components": {}}
        d["setup_type"] = "X"
        d["score"] = 75
        rows.append(d)
    findings = TradeReviewer(min_samples=30).analyze(rows)
    texts = " ".join(f.text for f in findings)
    assert "'weak_tag' underperforms" in texts


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21
