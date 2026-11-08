#!/usr/bin/env python3
"""
Forex & Gold signal bot - command line entry point.

  python main.py run                      live loop (MT5 -> analysis -> Telegram)
  python main.py check                    diagnostics: config, MT5, Telegram, calendar
  python main.py scan [--symbol XAUUSD]   one analysis pass, prints decisions (no Telegram)
  python main.py download --symbol XAUUSD --days 365           MT5 history -> data/history
  python main.py backtest --symbol XAUUSD --data data/history/XAUUSD_M5.csv
  python main.py walkforward --symbol XAUUSD --data ...         walk-forward validation
  python main.py report                   print today's report (and send with --send)
  python main.py review                   trade review -> recommendations (pending approval)
  python main.py recommendations [--approve ID | --reject ID]
  python main.py test-telegram            send a test message

SIGNAL-ONLY: no command places orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import BASE_DIR, Config, load_config


def setup_logging(cfg: Config) -> None:
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(cfg.log_level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.handlers.RotatingFileHandler(log_dir / "system.log", maxBytes=10_000_000, backupCount=10,
                                              encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sig = logging.getLogger("signals")
    sh2 = logging.handlers.RotatingFileHandler(log_dir / "signals.log", maxBytes=10_000_000, backupCount=20,
                                               encoding="utf-8")
    sh2.setFormatter(fmt)
    sig.addHandler(sh2)
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
def cmd_run(cfg, args):
    from core.instance_lock import AlreadyRunning, InstanceLock
    from core.orchestrator import SignalBot
    try:
        lock = InstanceLock(str(BASE_DIR / "data" / "bot.lock")).acquire()
    except AlreadyRunning as exc:
        logging.getLogger("main").error("%s - refusing to start a second instance.", exc)
        return 2
    try:
        SignalBot(cfg).run_forever()
    finally:
        lock.release()


def cmd_check(cfg, args):
    from core.orchestrator import build_provider
    from core.market_data import MarketDataService
    from core.domain import Timeframe
    from integrations.economic_calendar import build_calendar
    from integrations.telegram_bot import TelegramNotifier
    ok = True
    print("== Configuration")
    for p in cfg.validate():
        print("  !", p)
    print("  symbols:", ", ".join(cfg.active_symbols))
    print("== Data provider:", cfg.data_provider)
    try:
        prov = build_provider(cfg)
        if prov.connect():
            md = MarketDataService(prov, cfg)
            for s in cfg.active_symbols:
                bs = md.broker_symbol(s)
                if bs is None:
                    print(f"  x {s}: not found at provider")
                    ok = False
                    continue
                t = md.get_tick(s)
                df = md.get_candles(s, Timeframe.M5, 50)
                spec = md.spec(s)
                sp = f"{spec.to_pips(t.spread):.1f} pips" if t else "n/a"
                last = df.index[-1] if df is not None and len(df) else "n/a"
                print(f"  ok {s} -> {bs}: last M5 bar {last}, spread {sp}, digits {spec.digits}")
            prov.shutdown()
        else:
            print("  x could not connect")
            ok = False
    except Exception as exc:
        print("  x", exc)
        ok = False
    print("== Telegram")
    try:
        print("  ", TelegramNotifier(cfg.telegram).test_connection())
    except Exception as exc:
        print("  x", exc)
        ok = False
    print("== Economic calendar:", cfg.news.calendar_provider)
    cal = build_calendar(cfg.news, str(BASE_DIR / "data"))
    cal.refresh_if_needed(force=True)
    if cal.available:
        now = datetime.now(timezone.utc)
        upcoming = [e for e in cal.events_between(now, now + timedelta(days=7)) if e.is_high]
        print(f"  ok {len(cal.events)} events, {len(upcoming)} high-impact in next 7 days")
        for e in upcoming[:5]:
            print(f"     {e.time:%a %H:%M} UTC {e.currency} {e.title}")
    else:
        print("  x unavailable:", cal.last_error)
        ok = False
    print("== News headlines:", "Finnhub" if cfg.news.finnhub_api_key else "not configured (optional)")
    print("== US yields (FRED):", "configured" if cfg.news.fred_api_key else "not configured (optional, gold macro)")
    print("== DXY symbol:", cfg.mt5.dxy_symbol or "not configured (optional, gold macro)")
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def cmd_scan(cfg, args):
    from core.orchestrator import SignalBot
    bot = SignalBot(cfg)
    if not bot.provider.connect():
        print("Cannot connect to provider")
        return 1
    now = bot.provider.now()
    bot.calendar.refresh_if_needed(now)
    symbols = [args.symbol.upper()] if args.symbol else cfg.active_symbols
    from core.domain import Timeframe
    from core.signal_engine import MarketContext
    for s in symbols:
        m5 = bot.market.get_candles(s, Timeframe.M5, cfg.analysis.bars["M5"])
        tick = bot.market.get_tick(s)
        frames = bot._frames(s, m5) if m5 is not None and not m5.empty else None
        if frames is None or tick is None:
            print(f"{s}: data unavailable")
            continue
        ctx = MarketContext(s, bot.market.spec(s), now, frames, tick.bid, tick.ask, bot.portfolio_rejections(s, now))
        res = bot.engine.evaluate(ctx)
        if res.candidate:
            from integrations.telegram_messages import signal_message
            print(signal_message(res.candidate, bot.market.spec(s)))
        else:
            sc = f" (score {res.score:.0f})" if res.score is not None else ""
            print(f"{s}: no signal{sc} - " + "; ".join(res.rejections))
    bot.provider.shutdown()
    return 0


def cmd_download(cfg, args):
    from core.orchestrator import build_provider
    from core.market_data import MarketDataService
    from core.domain import Timeframe
    from backtesting.historical_data import download_mt5
    prov = build_provider(cfg)
    if not prov.connect():
        print("Cannot connect to MT5")
        return 1
    md = MarketDataService(prov, cfg)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    for tf in args.timeframes.split(","):
        tf = Timeframe(tf.strip().upper())
        path = Path(cfg.csv_data_dir) / f"{args.symbol.upper()}_{tf.value}.csv"
        df = download_mt5(md, args.symbol.upper(), tf, start, end, str(path))
        print(f"saved {len(df)} bars -> {path}")
    prov.shutdown()
    return 0


def _load_backtest_inputs(cfg, args):
    from backtesting.historical_data import load_csv
    from core.domain import SymbolSpec
    from integrations.economic_calendar import CSVCalendar, EconomicCalendar
    sym = args.symbol.upper()
    inst = cfg.instrument(sym)
    df = load_csv(args.data, server_utc_offset=args.server_offset, point=inst.point)
    spec = SymbolSpec(sym, sym, inst.point, inst.digits, inst.pip_size, inst.max_spread_pips, inst.is_gold,
                      inst.currencies, inst.commission_pips, inst.slippage_pips)
    cal = None
    if args.calendar:
        cal = EconomicCalendar(CSVCalendar(args.calendar), max_staleness_hours=1e9)
        cal.refresh_if_needed(force=True)
        if not cal.available:
            raise SystemExit(f"Could not load calendar: {cal.last_error}")
    return sym, df, spec, cal


def cmd_backtest(cfg, args):
    from backtesting.backtest_engine import Backtester
    sym, df, spec, cal = _load_backtest_inputs(cfg, args)
    if args.min_score is not None:
        cfg.scoring.min_score = args.min_score
    bt = Backtester(cfg, calendar=cal, default_spread_pips=args.spread_pips)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else None
    if start is None:
        start = df.index[0].to_pydatetime() + timedelta(days=45)  # indicator warm-up
    res = bt.run(sym, df, spec, start, end)
    print(res.to_text())
    out = Path(args.out or BASE_DIR / "data" / "backtests")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    res.trades_frame().to_csv(out / f"{sym}_{stamp}_trades.csv", index=False)
    (out / f"{sym}_{stamp}_summary.json").write_text(json.dumps(
        {"stats": res.stats.to_dict(), "assumptions": res.assumptions, "rejections": res.rejections},
        default=str, indent=2))
    print(f"\nTrades written to {out}/{sym}_{stamp}_trades.csv")
    return 0


def cmd_walkforward(cfg, args):
    from backtesting.backtest_engine import Backtester, ReplayParams
    from backtesting.strategy_optimizer import walk_forward
    sym, df, spec, cal = _load_backtest_inputs(cfg, args)
    bt = Backtester(cfg, calendar=cal, default_spread_pips=args.spread_pips)
    data = bt.prepare(sym, df, spec)
    res = walk_forward(bt, data, ReplayParams.from_config(cfg), train_days=args.train_days,
                       test_days=args.test_days, min_trades=args.min_trades)
    print(res.to_text())
    return 0


def cmd_report(cfg, args):
    from core.orchestrator import SignalBot
    from core.market_data import CSVDataProvider
    bot = SignalBot(cfg, provider=CSVDataProvider(cfg.csv_data_dir))
    text = bot.build_daily_report(datetime.now(timezone.utc))
    print(text)
    if args.send:
        bot.notifier.send(text, kind="report")
    return 0


def cmd_review(cfg, args):
    from core.trade_review import TradeReviewer
    from database.database import Database
    db = Database(cfg.database_url)
    findings = TradeReviewer(min_samples=args.min_samples).analyze(db.all_closed())
    for f in findings:
        print(f"[{f.category}] {f.text}")
        if f.category != "info":
            db.add_recommendation(f.category, f.text, f.evidence)
    return 0


def cmd_recommendations(cfg, args):
    from database.database import Database
    db = Database(cfg.database_url)
    if args.approve:
        print("approved" if db.set_recommendation_status(args.approve, "APPROVED") else "not found")
        print("NOTE: approval only records your decision. Apply the change in .env yourself after validation.")
        return 0
    if args.reject:
        print("rejected" if db.set_recommendation_status(args.reject, "REJECTED") else "not found")
        return 0
    for r in db.recommendations():
        print(f"#{r.id} [{r.status}] {r.category}: {r.text}")
    return 0


def cmd_test_telegram(cfg, args):
    from integrations.telegram_bot import TelegramNotifier
    from integrations.telegram_messages import system_message
    n = TelegramNotifier(cfg.telegram, dry_run_path=str(Path(cfg.log_dir) / "telegram_dryrun.log"))
    print(n.test_connection())
    print("sent" if n.send(system_message("ok", "Test message from the signal bot.")) else "failed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Forex & Gold signal bot (signal-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("check")
    p = sub.add_parser("scan")
    p.add_argument("--symbol")
    p = sub.add_parser("download")
    p.add_argument("--symbol", required=True)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--timeframes", default="M5,M1")
    for name in ("backtest", "walkforward"):
        p = sub.add_parser(name)
        p.add_argument("--symbol", required=True)
        p.add_argument("--data", required=True, help="M5 CSV (MT5 export or generic)")
        p.add_argument("--calendar", help="historical calendar CSV (time_utc,currency,impact,title)")
        p.add_argument("--server-offset", type=float, default=0.0,
                       help="hours to subtract from CSV timestamps (MT5 export in server time: 2 or 3)")
        p.add_argument("--spread-pips", type=float, default=None, help="spread if the data has none")
        if name == "backtest":
            p.add_argument("--start")
            p.add_argument("--end")
            p.add_argument("--min-score", type=float)
            p.add_argument("--out")
        else:
            p.add_argument("--train-days", type=int, default=90)
            p.add_argument("--test-days", type=int, default=30)
            p.add_argument("--min-trades", type=int, default=15)
    p = sub.add_parser("report")
    p.add_argument("--send", action="store_true")
    p = sub.add_parser("review")
    p.add_argument("--min-samples", type=int, default=30)
    p = sub.add_parser("recommendations")
    p.add_argument("--approve", type=int)
    p.add_argument("--reject", type=int)
    sub.add_parser("test-telegram")
    args = ap.parse_args(argv)
    cfg = load_config()
    setup_logging(cfg)
    handler = {
        "run": cmd_run, "check": cmd_check, "scan": cmd_scan, "download": cmd_download,
        "backtest": cmd_backtest, "walkforward": cmd_walkforward, "report": cmd_report,
        "review": cmd_review, "recommendations": cmd_recommendations, "test-telegram": cmd_test_telegram,
    }[args.cmd]
    return handler(cfg, args) or 0


if __name__ == "__main__":
    sys.exit(main())
