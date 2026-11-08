"""
Central configuration.

All values can be overridden through environment variables (or a `.env` file in
the project root). Nothing in here is "tuned for profit": defaults are
conservative starting points that MUST be validated with the backtester /
walk-forward optimizer on real historical data before being trusted.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on", "y")


def _env_list(name: str, default: List[str]) -> List[str]:
    v = _env(name)
    if v is None:
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Instrument specification (fallback values; live values come from MT5)
# ---------------------------------------------------------------------------
@dataclass
class InstrumentConfig:
    symbol: str                 # canonical name, e.g. XAUUSD
    point: float                # minimal price increment
    digits: int
    pip_size: float             # what a "pip" means for this instrument
    max_spread_pips: float      # hard spread filter
    commission_pips: float = 0.0  # round-turn commission expressed in pips (for backtests)
    slippage_pips: float = 0.0    # assumed slippage per fill (for backtests)
    is_gold: bool = False
    enabled: bool = True
    priority: int = 1           # lower = scanned first
    broker_symbol: Optional[str] = None  # override e.g. "XAUUSD+" on Vantage

    @property
    def base(self) -> str:
        return self.symbol[:3]

    @property
    def quote(self) -> str:
        return self.symbol[3:6]

    @property
    def currencies(self) -> List[str]:
        """Currencies whose news can move this instrument."""
        if self.is_gold or self.base in ("XAU", "XAG"):
            return ["USD"]
        return [self.base, self.quote]


DEFAULT_INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "XAUUSD": InstrumentConfig("XAUUSD", 0.01, 2, 0.10, max_spread_pips=4.0,
                               commission_pips=0.7, slippage_pips=0.5, is_gold=True, priority=0),
    "EURUSD": InstrumentConfig("EURUSD", 0.00001, 5, 0.0001, 2.0, 0.7, 0.2),
    "GBPUSD": InstrumentConfig("GBPUSD", 0.00001, 5, 0.0001, 2.5, 0.7, 0.2),
    "USDJPY": InstrumentConfig("USDJPY", 0.001, 3, 0.01, 2.0, 0.7, 0.2),
    "GBPJPY": InstrumentConfig("GBPJPY", 0.001, 3, 0.01, 4.0, 0.7, 0.3),
    "AUDUSD": InstrumentConfig("AUDUSD", 0.00001, 5, 0.0001, 2.0, 0.7, 0.2),
    "NZDUSD": InstrumentConfig("NZDUSD", 0.00001, 5, 0.0001, 2.5, 0.7, 0.2),
    "USDCAD": InstrumentConfig("USDCAD", 0.00001, 5, 0.0001, 2.5, 0.7, 0.2),
    "USDCHF": InstrumentConfig("USDCHF", 0.00001, 5, 0.0001, 2.5, 0.7, 0.2),
}


def generic_instrument(symbol: str) -> InstrumentConfig:
    """Fallback spec for instruments added later without an explicit entry."""
    jpy = symbol.endswith("JPY")
    return InstrumentConfig(
        symbol=symbol,
        point=0.001 if jpy else 0.00001,
        digits=3 if jpy else 5,
        pip_size=0.01 if jpy else 0.0001,
        max_spread_pips=3.0,
        commission_pips=0.7,
        slippage_pips=0.3,
    )


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------
@dataclass
class MT5Config:
    enabled: bool = _env_bool("MT5_ENABLED", True)
    terminal_path: Optional[str] = _env("MT5_TERMINAL_PATH")
    login: Optional[int] = int(_env("MT5_LOGIN")) if _env("MT5_LOGIN") else None
    password: Optional[str] = _env("MT5_PASSWORD")
    server: Optional[str] = _env("MT5_SERVER")  # e.g. "VantageInternational-Live"
    symbol_suffix: str = _env("MT5_SYMBOL_SUFFIX", "") or ""
    # Broker server time offset from UTC in hours. "auto" = detect from ticks.
    server_utc_offset: str = _env("MT5_SERVER_UTC_OFFSET", "auto") or "auto"
    timeout_ms: int = _env_int("MT5_TIMEOUT_MS", 60000)
    max_reconnect_attempts: int = _env_int("MT5_MAX_RECONNECT_ATTEMPTS", 10)
    # Optional DXY / index symbols available at the broker (empty = disabled)
    dxy_symbol: Optional[str] = _env("DXY_SYMBOL")


@dataclass
class TelegramConfig:
    bot_token: Optional[str] = _env("TELEGRAM_BOT_TOKEN")
    chat_id: Optional[str] = _env("TELEGRAM_CHAT_ID")
    enable_commands: bool = _env_bool("TELEGRAM_ENABLE_COMMANDS", True)
    timeout_s: int = _env_int("TELEGRAM_TIMEOUT_S", 15)
    max_retries: int = _env_int("TELEGRAM_MAX_RETRIES", 5)

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class NewsConfig:
    calendar_provider: str = _env("CALENDAR_PROVIDER", "forexfactory") or "forexfactory"  # forexfactory|csv|none
    calendar_csv_path: str = _env("CALENDAR_CSV_PATH", str(BASE_DIR / "data" / "calendar.csv"))
    calendar_refresh_minutes: int = _env_int("CALENDAR_REFRESH_MINUTES", 60)
    finnhub_api_key: Optional[str] = _env("FINNHUB_API_KEY")
    fred_api_key: Optional[str] = _env("FRED_API_KEY")
    # Blackout windows (minutes) around events
    high_impact_before_min: int = _env_int("NEWS_HIGH_BEFORE_MIN", 30)
    high_impact_after_min: int = _env_int("NEWS_HIGH_AFTER_MIN", 30)
    medium_impact_before_min: int = _env_int("NEWS_MEDIUM_BEFORE_MIN", 15)
    medium_impact_after_min: int = _env_int("NEWS_MEDIUM_AFTER_MIN", 10)
    block_on_medium: bool = _env_bool("NEWS_BLOCK_ON_MEDIUM", False)
    # Gold-specific: tier-1 USD events get wider windows
    gold_tier1_before_min: int = _env_int("GOLD_TIER1_BEFORE_MIN", 45)
    gold_tier1_after_min: int = _env_int("GOLD_TIER1_AFTER_MIN", 60)
    alert_lead_minutes: List[int] = field(default_factory=lambda: [int(x) for x in _env_list("NEWS_ALERT_LEAD_MIN", ["60", "15"])])
    # If the calendar cannot be fetched: block new signals (safe) or continue with a warning
    block_if_calendar_unavailable: bool = _env_bool("BLOCK_IF_CALENDAR_UNAVAILABLE", True)
    headline_poll_minutes: int = _env_int("HEADLINE_POLL_MINUTES", 10)


@dataclass
class ScoringConfig:
    # Component weights (normalised to 100 at runtime)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "htf_trend": 20,
        "market_structure": 15,
        "smc": 20,
        "momentum": 10,
        "volatility": 5,
        "price_action": 10,
        "fundamentals": 10,
        "risk_reward": 5,
        "execution": 5,
    })
    min_score: float = _env_float("MIN_SIGNAL_SCORE", 70.0)
    strong_score: float = 80.0
    very_strong_score: float = 90.0
    min_htf_bias: float = _env_float("MIN_HTF_BIAS", 0.35)


@dataclass
class RiskConfig:
    min_rr: float = _env_float("MIN_RR", 2.0)
    tp_r_multiples: List[float] = field(default_factory=lambda: [float(x) for x in _env_list("TP_R_MULTIPLES", ["1.0", "2.0", "3.0"])])
    # partial-close assumption used to compute realised R (must sum to 1)
    tp_close_fractions: List[float] = field(default_factory=lambda: [float(x) for x in _env_list("TP_CLOSE_FRACTIONS", ["0.5", "0.3", "0.2"])])
    move_sl_to_be_after_tp1: bool = _env_bool("MOVE_SL_TO_BE_AFTER_TP1", True)
    sl_atr_buffer: float = _env_float("SL_ATR_BUFFER", 0.25)    # extra buffer beyond structure
    sl_min_atr: float = _env_float("SL_MIN_ATR", 1.0)           # min SL distance (x ATR of entry TF)
    sl_max_atr: float = _env_float("SL_MAX_ATR", 3.0)           # reject if structure SL wider than this
    max_spread_atr_ratio: float = _env_float("MAX_SPREAD_ATR_RATIO", 0.15)
    max_signals_per_day: int = _env_int("MAX_SIGNALS_PER_DAY", 6)
    max_signals_per_symbol_per_day: int = _env_int("MAX_SIGNALS_PER_SYMBOL_PER_DAY", 2)
    max_consecutive_losses: int = _env_int("MAX_CONSECUTIVE_LOSSES", 3)
    max_active_signals: int = _env_int("MAX_ACTIVE_SIGNALS", 4)
    max_active_per_currency: int = _env_int("MAX_ACTIVE_PER_CURRENCY", 2)
    symbol_cooldown_minutes: int = _env_int("SYMBOL_COOLDOWN_MINUTES", 90)
    max_holding_hours: float = _env_float("MAX_HOLDING_HOURS", 24.0)
    high_volatility_percentile: float = 0.90
    low_volatility_percentile: float = 0.10


@dataclass
class AnalysisConfig:
    # Number of closed bars requested per timeframe for live analysis
    bars: Dict[str, int] = field(default_factory=lambda: {
        "M1": 300, "M5": 600, "M15": 600, "M30": 400, "H1": 600, "H4": 400, "D1": 300,
    })
    # SMC window (bars) per timeframe -- identical in live and backtest
    smc_window: int = _env_int("SMC_WINDOW", 300)
    swing_strength: Dict[str, int] = field(default_factory=lambda: {
        "M1": 3, "M5": 3, "M15": 3, "M30": 3, "H1": 3, "H4": 3, "D1": 2,
    })
    internal_swing_strength: int = 1
    fvg_min_atr: float = 0.15
    ob_min_displacement_atr: float = 1.0
    equal_level_atr_tol: float = 0.10
    sweep_lookback_bars: int = 6
    trigger_lookback_m15: int = 4
    trigger_lookback_m5: int = 8
    use_m1: bool = _env_bool("USE_M1_TIMING", False)
    # execution-quality gates
    abnormal_spread_mult: float = _env_float("ABNORMAL_SPREAD_MULT", 3.0)     # x median M5 spread of last day
    week_open_buffer_min: int = _env_int("WEEK_OPEN_BUFFER_MIN", 60)
    week_close_buffer_min: int = _env_int("WEEK_CLOSE_BUFFER_MIN", 120)
    gold_max_macro_conflicts: int = _env_int("GOLD_MAX_MACRO_CONFLICTS", 1)  # reject above this
    allowed_sessions: List[str] = field(default_factory=lambda: _env_list(
        "ALLOWED_SESSIONS", ["London", "New York"]))
    gold_allowed_sessions: List[str] = field(default_factory=lambda: _env_list(
        "GOLD_ALLOWED_SESSIONS", ["London", "New York"]))


@dataclass
class RuntimeConfig:
    scan_interval_s: int = _env_int("SCAN_INTERVAL_S", 20)
    tracker_interval_s: int = _env_int("TRACKER_INTERVAL_S", 15)
    daily_report_utc: str = _env("DAILY_REPORT_UTC", "21:15") or "21:15"
    backup_dir: str = _env("BACKUP_DIR", str(BASE_DIR / "data" / "backups")) or ""
    backup_keep: int = _env_int("BACKUP_KEEP", 14)
    stale_tick_seconds: int = _env_int("STALE_TICK_SECONDS", 180)
    error_alert_cooldown_min: int = _env_int("ERROR_ALERT_COOLDOWN_MIN", 30)


@dataclass
class Config:
    instruments: Dict[str, InstrumentConfig] = field(default_factory=dict)
    mt5: MT5Config = field(default_factory=MT5Config)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    database_url: str = _env("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'bot.db'}") or ""
    log_dir: str = _env("LOG_DIR", str(BASE_DIR / "logs")) or ""
    log_level: str = _env("LOG_LEVEL", "INFO") or "INFO"
    data_provider: str = _env("DATA_PROVIDER", "mt5") or "mt5"   # mt5 | csv
    csv_data_dir: str = _env("CSV_DATA_DIR", str(BASE_DIR / "data" / "history")) or ""

    def instrument(self, symbol: str) -> InstrumentConfig:
        if symbol not in self.instruments:
            self.instruments[symbol] = generic_instrument(symbol)
        return self.instruments[symbol]

    @property
    def active_symbols(self) -> List[str]:
        items = [i for i in self.instruments.values() if i.enabled]
        return [i.symbol for i in sorted(items, key=lambda x: (x.priority, x.symbol))]

    def validate(self) -> List[str]:
        """Return a list of configuration problems (empty = OK)."""
        problems: List[str] = []
        r = self.risk
        if abs(sum(r.tp_close_fractions) - 1.0) > 1e-6:
            problems.append("TP_CLOSE_FRACTIONS must sum to 1.0")
        if len(r.tp_close_fractions) != len(r.tp_r_multiples):
            problems.append("TP_CLOSE_FRACTIONS and TP_R_MULTIPLES must have the same length")
        if sorted(r.tp_r_multiples) != r.tp_r_multiples:
            problems.append("TP_R_MULTIPLES must be ascending")
        if r.min_rr <= 0:
            problems.append("MIN_RR must be positive")
        if max(r.tp_r_multiples) < r.min_rr:
            problems.append("Highest TP multiple is below MIN_RR")
        if r.sl_min_atr >= r.sl_max_atr:
            problems.append("SL_MIN_ATR must be < SL_MAX_ATR")
        s = self.scoring
        if not (0 < s.min_score <= 100):
            problems.append("MIN_SIGNAL_SCORE must be in (0, 100]")
        if not self.telegram.enabled:
            problems.append("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) - messages will be written to the dry-run log only")
        return problems

    def to_dict(self) -> dict:
        d = asdict(self)
        # never leak secrets into logs
        d["mt5"]["password"] = "***" if self.mt5.password else None
        d["telegram"]["bot_token"] = "***" if self.telegram.bot_token else None
        d["news"]["finnhub_api_key"] = "***" if self.news.finnhub_api_key else None
        d["news"]["fred_api_key"] = "***" if self.news.fred_api_key else None
        return d


def load_config() -> Config:
    cfg = Config()
    enabled = _env_list("SYMBOLS", list(DEFAULT_INSTRUMENTS.keys()))
    for sym in enabled:
        sym = sym.upper()
        inst = DEFAULT_INSTRUMENTS.get(sym) or generic_instrument(sym)
        cfg.instruments[sym] = InstrumentConfig(**asdict(inst))
    # Optional per-instrument overrides as JSON, e.g.
    # INSTRUMENT_OVERRIDES={"XAUUSD": {"broker_symbol": "XAUUSD+", "max_spread_pips": 5}}
    overrides = _env("INSTRUMENT_OVERRIDES")
    if overrides:
        for sym, values in json.loads(overrides).items():
            inst = cfg.instrument(sym.upper())
            for k, v in values.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)
    return cfg
