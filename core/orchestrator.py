"""
Live orchestration loop.

Every SCAN_INTERVAL_S seconds:
  1. ensure the data provider is connected (auto-reconnect, alert on failure)
  2. refresh the economic calendar / headlines; send upcoming-event alerts
  3. advance every open signal with new closed M1 bars (TP/SL/BE/expiry),
     cancel open signals invalidated before TP1, send event notifications
  4. for each symbol whose M5 bar just closed: run the full analysis and,
     if a signal passes every gate, store it and send it to Telegram
  5. daily report, trade review, database backup, outbox retry

The bot never sends orders. Every notification has a dedup key, so restarts
or retries never produce duplicate messages.
"""
from __future__ import annotations

import logging
import signal as _signal
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import BASE_DIR, Config
from core.domain import Direction, Timeframe
from core.fundamental_analysis import FundamentalAnalyzer
from core.market_data import DataProvider, MarketDataService
from core.performance_analyzer import best_of, breakdown, compute_stats, fmt_pct
from core.risk_management import RiskManager
from core.sessions import is_fx_market_open, session_label
from core.signal_engine import MarketContext, SignalEngine, prepare_frame
from core.signal_tracker import SignalTracker
from core.trade_review import TradeReviewer, review_signal
from database.database import Database, _aware
from integrations import telegram_messages as tm
from integrations.economic_calendar import EconomicCalendar, build_calendar
from integrations.macro_data import FredClient, MacroDataService
from integrations.news_provider import NewsProvider, build_news_provider
from integrations.telegram_bot import TelegramNotifier

log = logging.getLogger(__name__)

SCAN_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1)


def build_provider(cfg: Config) -> DataProvider:
    if cfg.data_provider.lower() == "csv":
        from core.market_data import CSVDataProvider
        return CSVDataProvider(cfg.csv_data_dir)
    from integrations.mt5_connector import MT5DataProvider
    return MT5DataProvider(cfg.mt5, state_path=str(BASE_DIR / "data" / "mt5_server_offset.json"))


class SignalBot:
    def __init__(self, cfg: Config, provider: Optional[DataProvider] = None, db: Optional[Database] = None,
                 notifier: Optional[TelegramNotifier] = None, calendar: Optional[EconomicCalendar] = None,
                 news: Optional[NewsProvider] = None):
        self.cfg = cfg
        self.db = db or Database(cfg.database_url)
        self.provider = provider or build_provider(cfg)
        self.market = MarketDataService(self.provider, cfg)
        self.calendar = calendar or build_calendar(cfg.news, str(BASE_DIR / "data"))
        self.news = news or build_news_provider(cfg.news)
        self.macro = MacroDataService(FredClient(cfg.news.fred_api_key), self.market, cfg.mt5.dxy_symbol)
        self.fund = FundamentalAnalyzer(cfg.news, self.calendar, self.macro, self.news)
        self.risk = RiskManager(cfg.risk)
        self.engine = SignalEngine(cfg, self.fund, self.risk)
        self.tracker = SignalTracker(self.db, cfg.risk)
        self.notifier = notifier or TelegramNotifier(cfg.telegram, self.db,
                                                     dry_run_path=str(Path(cfg.log_dir) / "telegram_dryrun.log"))
        self.reviewer = TradeReviewer()
        self._stop = threading.Event()
        self._alert_times: Dict[str, float] = {}
        self._last_headline_poll = 0.0
        self._last_headline_time: Optional[datetime] = None
        self._last_eval: Dict[str, str] = {}
        self.started_at = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ lifecycle
    def stop(self, *_):
        log.info("Shutdown requested")
        self._stop.set()

    def install_signal_handlers(self):
        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                _signal.signal(sig, self.stop)
            except (ValueError, OSError):  # not in main thread / unsupported
                pass

    def run_forever(self) -> None:
        self.install_signal_handlers()
        for p in self.cfg.validate():
            log.warning("Config: %s", p)
        while not self._stop.is_set() and not self.provider.connect():
            self.alert("connect", "error", f"Cannot connect to data provider '{self.provider.name}'. Retrying in 60s.")
            self._stop.wait(60)
        if self._stop.is_set():
            return
        self._register_commands()
        self.notifier.start_polling()
        self.notifier.send(tm.system_message("ok", f"Signal bot started. Monitoring: {', '.join(self.cfg.active_symbols)}"),
                           kind="system")
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    self.tick()
                except Exception as exc:
                    log.exception("Main loop error: %s", exc)
                    self.alert("loop", "error", f"Main loop error: {exc}")
                self._stop.wait(max(1.0, self.cfg.runtime.scan_interval_s - (time.monotonic() - t0)))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            self.notifier.send(tm.system_message("info", "Signal bot stopped."), kind="system")
        except Exception:
            pass
        self.notifier.stop()
        self.provider.shutdown()
        log.info("Shutdown complete")

    # ------------------------------------------------------------------ alerts
    def alert(self, kind: str, level: str, text: str) -> None:
        """System alert, rate-limited per kind."""
        now = time.monotonic()
        cooldown = self.cfg.runtime.error_alert_cooldown_min * 60
        if now - self._alert_times.get(kind, -1e9) < cooldown:
            return
        self._alert_times[kind] = now
        from core.redact import redact
        text = redact(text)
        log.log(logging.ERROR if level == "error" else logging.WARNING, text)
        self.notifier.send(tm.system_message(level, text), kind="system")

    # ------------------------------------------------------------------ main tick
    def tick(self, now: Optional[datetime] = None) -> None:
        now = now or self.provider.now()
        if not self.market.ensure_connected():
            self.alert("disconnected", "error", f"Data provider '{self.provider.name}' disconnected; reconnect failed.")
            return
        self.calendar.refresh_if_needed(now)
        if not self.calendar.available and self.cfg.news.calendar_provider != "none":
            self.alert("calendar", "warning", f"Economic calendar unavailable ({self.calendar.last_error}). "
                       + ("New signals are blocked until it recovers." if self.fund.block_if_unavailable
                          else "Signals continue WITHOUT news filtering."))
        self.poll_headlines(now)
        self.send_news_alerts(now)
        if is_fx_market_open(now):
            self.update_open_signals(now)
            for sym in self.cfg.active_symbols:
                if self._stop.is_set():
                    break
                try:
                    self.scan_symbol(sym, now)
                except Exception as exc:
                    log.exception("Scan %s failed: %s", sym, exc)
                    self.alert(f"scan:{sym}", "warning", f"Analysis of {sym} failed: {exc}")
        self.maybe_daily_tasks(now)
        self.notifier.flush_outbox()

    # ------------------------------------------------------------------ analysis
    def _frames(self, symbol: str, m5) -> Optional[Dict[Timeframe, object]]:
        frames = {Timeframe.M5: prepare_frame(m5)}
        tfs = list(SCAN_TFS[1:]) + ([Timeframe.M1] if self.cfg.analysis.use_m1 else [])
        for tf in tfs:
            df = self.market.get_candles(symbol, tf, self.cfg.analysis.bars.get(tf.value, 600))
            if df is None or df.empty:
                return None
            frames[tf] = prepare_frame(df)
        return frames

    def portfolio_rejections(self, symbol: str, now: datetime) -> List[str]:
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        todays = [{"symbol": s.symbol} for s in self.db.signals_between(day_start, now + timedelta(seconds=1))]
        active = [{"symbol": s.symbol, "currencies": self.cfg.instrument(s.symbol).currencies}
                  for s in self.db.open_signals()]
        return self.risk.check_limits(symbol, self.cfg.instrument(symbol).currencies, now, todays, active,
                                      self.db.consecutive_losses(since=day_start), self.db.last_signal_time(symbol))

    def scan_symbol(self, symbol: str, now: datetime):
        m5 = self.market.get_candles(symbol, Timeframe.M5, self.cfg.analysis.bars["M5"])
        if m5 is None or m5.empty or not self.market.is_new_bar(symbol, Timeframe.M5, m5):
            return None
        last_close = m5.index[-1] + Timeframe.M5.delta
        if now - last_close.to_pydatetime() > timedelta(minutes=15):
            log.info("%s: latest M5 bar is stale (%s) - skipping", symbol, last_close)
            return None
        tick = self.market.get_tick(symbol)
        if tick is None or self.market.tick_is_stale(tick, self.cfg.runtime.stale_tick_seconds):
            log.info("%s: no fresh tick - skipping", symbol)
            return None
        frames = self._frames(symbol, m5)
        if frames is None:
            log.warning("%s: incomplete multi-timeframe data", symbol)
            return None
        spec = self.market.spec(symbol)
        ctx = MarketContext(symbol, spec, now, frames, tick.bid, tick.ask, self.portfolio_rejections(symbol, now))
        res = self.engine.evaluate(ctx)
        summary = (f"score={res.score:.0f} " if res.score is not None else "") + \
                  ("SIGNAL" if res.accepted else f"no signal: {res.rejections[0] if res.rejections else '-'}")
        if self._last_eval.get(symbol) != summary:
            log.info("%s %s %s", symbol, res.direction.value if res.direction else "-", summary)
            self._last_eval[symbol] = summary
        if res.candidate is None:
            return res
        rec = self.tracker.register(res.candidate)
        if rec is None:
            return res
        msg = tm.signal_message(res.candidate, spec, rec.id)
        self.notifier.send(msg, kind="signal", key=f"signal:{rec.id}")
        return res

    # ------------------------------------------------------------------ tracking
    def update_open_signals(self, now: datetime) -> None:
        for rec in self.db.open_signals():
            try:
                self._update_signal(rec, now)
            except Exception as exc:
                log.exception("Tracking signal %s failed: %s", rec.id, exc)
                self.alert(f"track:{rec.id}", "warning", f"Tracking of signal #{rec.id} failed: {exc}")

    def _update_signal(self, rec, now: datetime) -> None:
        spec = self.market.spec(rec.symbol)
        last = _aware(rec.last_checked_at) or _aware(rec.created_at)
        minutes = int((now - last).total_seconds() // 60) + 5
        bars = self.market.get_candles(rec.symbol, Timeframe.M1, max(10, min(minutes, 5000)))
        tick = self.market.get_tick(rec.symbol)
        spread = tick.spread if tick else rec.spread
        events = self.tracker.update_with_bars(rec, bars, spread) if bars is not None and not bars.empty else []
        for e in events:
            self.notifier.send(tm.event_message(rec, e.kind, e.price, e.r, spec.digits), kind="event",
                               key=f"event:{rec.id}:{e.kind}")
        if rec.outcome == "OPEN" and rec.tp_hits == 0 and tick is not None:
            m15 = self.market.get_candles(rec.symbol, Timeframe.M15, self.cfg.analysis.bars["M15"])
            if m15 is not None and len(m15) > 50:
                reason = self.engine.invalidation_reason(rec.symbol, Direction(rec.direction),
                                                         _aware(rec.created_at), prepare_frame(m15))
                if reason:
                    px = tick.bid if rec.direction == "BUY" else tick.ask
                    ev = self.tracker.cancel(rec, px, now, reason)
                    if ev:
                        self.notifier.send(tm.event_message(rec, "CANCELLED", px, ev.r, spec.digits, reason),
                                           kind="event", key=f"event:{rec.id}:CANCELLED")
        if rec.outcome != "OPEN" and not (rec.review or {}).get("notes"):
            rec.review = dict(rec.review or {}, **review_signal(rec))
            self.db.save_signal(rec)

    # ------------------------------------------------------------------ news
    def send_news_alerts(self, now: datetime) -> None:
        if not self.calendar.available:
            return
        horizon = max(self.cfg.news.alert_lead_minutes or [0])
        events = [e for e in self.calendar.events_between(now, now + timedelta(minutes=horizon + 1)) if e.is_high]
        for e in events:
            mins = int((e.time - now).total_seconds() // 60)
            for lead in sorted(self.cfg.news.alert_lead_minutes):
                if mins <= lead:
                    syms = [s for s in self.cfg.active_symbols if e.currency in self.cfg.instrument(s).currencies]
                    if syms:
                        self.notifier.send(tm.news_alert_message(e, max(mins, 0), syms), kind="news",
                                           key=f"news:{e.key}:{lead}")
                    break

    def poll_headlines(self, now: datetime) -> None:
        if not self.news.available:
            return
        if time.monotonic() - self._last_headline_poll < self.cfg.news.headline_poll_minutes * 60:
            return
        self._last_headline_poll = time.monotonic()
        try:
            items = self.news.latest(since=self._last_headline_time or now - timedelta(hours=2))
        except Exception as exc:
            log.warning("Headline poll failed: %s", exc)
            return
        if items:
            self._last_headline_time = items[-1].time
        risky = [i for i in items if i.risk_keywords]
        self.fund.recent_headlines = (self.fund.recent_headlines + risky)[-50:]
        for item in risky[-3:]:
            self.notifier.send(tm.headline_message(item), kind="headline", key=f"headline:{item.id}")

    # ------------------------------------------------------------------ daily
    def maybe_daily_tasks(self, now: datetime) -> None:
        hh, mm = (int(x) for x in self.cfg.runtime.daily_report_utc.split(":"))
        report_time = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if now < report_time or now.weekday() >= 5:
            return
        day = now.date().isoformat()
        if self.db.kv_get("last_daily_report") == day:
            return
        self.send_daily_report(now)
        self.run_review()
        try:
            self.db.backup(self.cfg.runtime.backup_dir, self.cfg.runtime.backup_keep)
        except Exception as exc:
            self.alert("backup", "warning", f"Database backup failed: {exc}")
        self.db.kv_set("last_daily_report", day)

    def build_daily_report(self, now: datetime) -> str:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        created = self.db.signals_between(start, end)
        closed = self.db.closed_between(start, end)
        by_symbol = breakdown(closed, "symbol")
        by_session = breakdown(closed, "session_bucket")
        all_time = compute_stats(self.db.all_closed())
        return tm.daily_report_message(start.date().isoformat(), compute_stats(created), compute_stats(closed),
                                       by_symbol, by_session, best_of(by_symbol), best_of(by_session),
                                       len(self.db.open_signals()), all_time)

    def send_daily_report(self, now: datetime) -> None:
        text = self.build_daily_report(now)
        self.notifier.send(text, kind="report", key=f"daily:{now.date().isoformat()}")

    def run_review(self) -> List[str]:
        findings = self.reviewer.analyze(self.db.all_closed())
        new = []
        for f in findings:
            if f.category == "info":
                continue
            if self.db.add_recommendation(f.category, f.text, f.evidence):
                new.append(f.text)
        if new:
            body = "\n".join(f"• {tm.esc(t)}" for t in new[:5])
            self.notifier.send("🧠 <b>Trade review – new recommendations (pending approval)</b>\n" + body +
                               "\n<i>Nothing is changed automatically. Validate with walk-forward before applying.</i>",
                               kind="review")
        return new

    # ------------------------------------------------------------------ commands
    def status_text(self) -> str:
        open_sigs = self.db.open_signals()
        stats = compute_stats(self.db.all_closed())
        now = datetime.now(timezone.utc)
        return (f"🤖 <b>Status</b>\nProvider: {self.provider.name} "
                f"({'connected' if self.provider.is_connected() else 'DISCONNECTED'})\n"
                f"Calendar: {'OK' if self.calendar.available else 'UNAVAILABLE'}\n"
                f"Session: {session_label(now)}\n"
                f"Open signals: {len(open_sigs)}\n"
                f"All-time resolved: {stats.resolved}, WR {fmt_pct(stats.win_rate)}, {stats.net_r:+.2f}R\n"
                f"Uptime since {self.started_at:%Y-%m-%d %H:%M} UTC")

    def active_text(self) -> str:
        rows = self.db.open_signals()
        if not rows:
            return "No active signals."
        return "\n".join(f"#{r.id} {r.symbol} {r.direction} entry {r.entry} SL {r.current_sl} "
                         f"TP hits {r.tp_hits} ({r.status})" for r in rows)

    def _register_commands(self) -> None:
        n = self.notifier
        n.register_command("status", self.status_text)
        n.register_command("active", self.active_text)
        n.register_command("report", lambda: self.build_daily_report(datetime.now(timezone.utc)))
        n.register_command("help", lambda: "/status – bot health\n/active – open signals\n/report – today's report")
