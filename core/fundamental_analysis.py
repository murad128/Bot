"""
Fundamental filter and context.

Responsibilities
1. News blackout (hard gate): no new signals inside the configured window
   before/after high-impact (optionally medium-impact) events for the
   instrument's currencies. Gold uses wider windows for tier-1 USD events
   (CPI, NFP, FOMC, PCE, ...).
2. Calendar availability: if the calendar cannot be loaded, the default is to
   BLOCK new signals (config BLOCK_IF_CALENDAR_UNAVAILABLE) because "no data"
   must never be treated as "no news".
3. Gold macro context: DXY trend and US real/nominal yield changes. These
   only adjust the fundamentals component and add explicit "conflict"
   warnings when they contradict the technical direction.

Everything returned is traceable to an actual data point; when an input is
missing the reason says so ("DXY data unavailable") instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

from config import NewsConfig
from core.domain import Direction, SymbolSpec
from integrations.economic_calendar import CalendarEvent, EconomicCalendar


@dataclass
class FundamentalAssessment:
    blocked: bool
    block_reason: Optional[str]
    score: float                       # 0..1 for the fundamentals component
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    conflict: bool = False
    upcoming: List[CalendarEvent] = field(default_factory=list)
    data_available: bool = False


class FundamentalAnalyzer:
    def __init__(self, cfg: NewsConfig, calendar: Optional[EconomicCalendar],
                 macro=None, news_provider=None, block_if_unavailable: Optional[bool] = None):
        self.cfg = cfg
        self.calendar = calendar
        self.macro = macro
        self.news = news_provider
        self.block_if_unavailable = cfg.block_if_calendar_unavailable if block_if_unavailable is None else block_if_unavailable
        self.recent_headlines = []    # filled by the orchestrator (NewsItem list)

    # ------------------------------------------------------------------ blackout
    def blackout(self, spec: SymbolSpec, now: datetime) -> Optional[str]:
        if self.calendar is None or not self.calendar.available:
            if self.block_if_unavailable:
                return "Economic calendar unavailable - new signals blocked (safety default)"
            return None
        cfg = self.cfg
        horizon_before = max(cfg.high_impact_before_min, cfg.gold_tier1_before_min, cfg.medium_impact_before_min)
        horizon_after = max(cfg.high_impact_after_min, cfg.gold_tier1_after_min, cfg.medium_impact_after_min)
        events = self.calendar.events_between(now - timedelta(minutes=horizon_after),
                                              now + timedelta(minutes=horizon_before), spec.currencies)
        for e in events:
            before, after = self._window(e, spec)
            if before is None:
                continue
            if e.time - timedelta(minutes=before) <= now <= e.time + timedelta(minutes=after):
                rel = int((e.time - now).total_seconds() // 60)
                when = f"in {rel} min" if rel >= 0 else f"{-rel} min ago"
                return f"News blackout: {e.currency} {e.title} ({e.impact}) {when}"
        return None

    def _window(self, e: CalendarEvent, spec: SymbolSpec):
        cfg = self.cfg
        if e.is_high:
            if spec.is_gold and e.currency == "USD" and e.is_tier1:
                return cfg.gold_tier1_before_min, cfg.gold_tier1_after_min
            return cfg.high_impact_before_min, cfg.high_impact_after_min
        if e.is_medium and cfg.block_on_medium:
            return cfg.medium_impact_before_min, cfg.medium_impact_after_min
        return None, None

    # ------------------------------------------------------------------ assessment
    def assess(self, spec: SymbolSpec, direction: Direction, now: datetime) -> FundamentalAssessment:
        reason = self.blackout(spec, now)
        a = FundamentalAssessment(blocked=reason is not None, block_reason=reason, score=0.5)
        if a.blocked:
            return a

        if self.calendar is not None and self.calendar.available:
            a.data_available = True
            upcoming = [e for e in self.calendar.events_between(now, now + timedelta(hours=4), spec.currencies)
                        if e.is_high]
            a.upcoming = upcoming
            if upcoming:
                e = upcoming[0]
                mins = int((e.time - now).total_seconds() // 60)
                a.score -= 0.1
                a.warnings.append(f"High-impact {e.currency} event in {mins} min: {e.title} - consider managing the trade before release")
            else:
                a.score += 0.2
                a.reasons.append("No high-impact news for the next 4h")
        else:
            a.warnings.append("Economic calendar unavailable - news risk not checked")

        if spec.is_gold:
            self._gold_context(direction, a)

        risky = [n for n in self.recent_headlines if n.risk_keywords and now - n.time <= timedelta(hours=2)]
        if risky:
            a.warnings.append(f"Headline risk: \"{risky[-1].headline[:90]}\" ({risky[-1].source})")
            if spec.is_gold and direction is Direction.BUY:
                a.reasons.append("Risk-off headlines may support safe-haven demand")
        a.score = max(0.0, min(1.0, a.score))
        return a

    def _gold_context(self, direction: Direction, a: FundamentalAssessment) -> None:
        if self.macro is None:
            a.warnings.append("Gold macro inputs (DXY / yields) not configured")
            return
        d = direction.sign
        dxy = None
        try:
            dxy = self.macro.dxy()
        except Exception:
            dxy = None
        if dxy is None:
            a.warnings.append("DXY data unavailable")
        else:
            a.data_available = True
            # gold is typically inversely related to USD strength
            if dxy.trend == 0:
                a.reasons.append("DXY flat")
            elif dxy.trend == -d:
                a.score += 0.2
                a.reasons.append(f"DXY {'falling' if dxy.trend < 0 else 'rising'} - supports gold {direction.value}")
            else:
                a.score -= 0.25
                a.conflict = True
                a.warnings.append(f"Fundamental conflict: DXY {'rising' if dxy.trend > 0 else 'falling'} against gold {direction.value}")
        ylds = {}
        try:
            ylds = self.macro.yields() or {}
        except Exception:
            ylds = {}
        real = ylds.get("DFII10") or ylds.get("DGS10")
        if real is None:
            a.warnings.append("US yield data unavailable (FRED_API_KEY not set or fetch failed)")
            return
        a.data_available = True
        chg = real.change_5d_bp
        label = "real" if real.series == "DFII10" else "nominal"
        if abs(chg) < 8:
            a.reasons.append(f"US 10Y {label} yield stable ({chg:+.0f}bp/5d)")
            return
        rising = chg > 0
        supportive = (rising and d < 0) or (not rising and d > 0)
        if supportive:
            a.score += 0.15
            a.reasons.append(f"US 10Y {label} yield {'rising' if rising else 'falling'} {chg:+.0f}bp/5d - supports gold {direction.value}")
        else:
            a.score -= 0.2
            a.conflict = True
            a.warnings.append(f"Fundamental conflict: US 10Y {label} yield {chg:+.0f}bp/5d against gold {direction.value}")
