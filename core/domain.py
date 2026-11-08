"""Domain objects shared by every module (no I/O here)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.BUY else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SELL if self is Direction.BUY else Direction.BUY

    @staticmethod
    def from_sign(sign: float) -> Optional["Direction"]:
        if sign > 0:
            return Direction.BUY
        if sign < 0:
            return Direction.SELL
        return None


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    @property
    def minutes(self) -> int:
        return {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[self.value]

    @property
    def delta(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    @property
    def pandas_rule(self) -> str:
        return {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min",
                "H1": "1h", "H4": "4h", "D1": "1D"}[self.value]


class SignalStatus(str, Enum):
    ACTIVE = "ACTIVE"          # position assumed open, no TP yet
    TP1_HIT = "TP1_HIT"        # still running after TP1
    TP2_HIT = "TP2_HIT"        # still running after TP2
    # terminal states
    TP3_HIT = "TP3_HIT"
    STOPPED = "STOPPED"        # SL hit (could be the original SL or a trailed SL after TP1/TP2)
    CANCELLED = "CANCELLED"    # invalidated before TP1/SL, closed at market (excluded from win rate)
    EXPIRED = "EXPIRED"        # max holding time reached, closed at market

    @property
    def is_open(self) -> bool:
        return self in (SignalStatus.ACTIVE, SignalStatus.TP1_HIT, SignalStatus.TP2_HIT)


class Outcome(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    CANCELLED = "CANCELLED"
    OPEN = "OPEN"


@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    time: datetime  # UTC

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2.0


@dataclass
class SymbolSpec:
    symbol: str
    broker_symbol: str
    point: float
    digits: int
    pip_size: float
    max_spread_pips: float
    is_gold: bool = False
    currencies: List[str] = field(default_factory=list)
    commission_pips: float = 0.0
    slippage_pips: float = 0.0

    def round_price(self, price: float) -> float:
        return round(price, self.digits)

    def to_pips(self, price_distance: float) -> float:
        return price_distance / self.pip_size


@dataclass
class ComponentScore:
    name: str
    score: float                     # 0..1
    weight: float                    # share of 100
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def points(self) -> float:
        return max(0.0, min(1.0, self.score)) * self.weight


@dataclass
class RiskPlan:
    entry: float
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profits: List[float]
    risk: float                      # |entry - SL| in price
    rr_targets: List[float]          # R multiple of each TP
    primary_rr: float                # R of the TP that represents "the" RR shown
    room_to_obstacle_r: Optional[float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class SignalCandidate:
    symbol: str
    direction: Direction
    setup_type: str
    created_at: datetime
    entry_timeframe: str
    confirm_timeframe: str
    trend_label: str
    session: str
    score: float
    grade: str
    plan: RiskPlan
    components: List[ComponentScore]
    reasons: List[str]
    warnings: List[str]
    atr: float
    spread: float
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        bucket = self.created_at.replace(second=0, microsecond=0)
        return f"{self.symbol}:{self.direction.value}:{self.setup_type}:{bucket.isoformat()}"


@dataclass
class EvaluationResult:
    symbol: str
    time: datetime
    candidate: Optional[SignalCandidate]
    rejections: List[str] = field(default_factory=list)
    direction: Optional[Direction] = None
    score: Optional[float] = None
    components: List[ComponentScore] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.candidate is not None
