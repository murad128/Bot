# Forex & Gold Signal Bot (MT5 → analysis → Telegram)

A **signal-only** analysis system for XAUUSD and the major FX pairs. It reads
market data from MetaTrader 5 (Vantage), combines multi-timeframe technical,
price-action, Smart-Money-Concepts and fundamental filters, and sends
explained signals to Telegram. It then tracks each signal's outcome, reports
daily, and reviews its own history.

> **It never places orders.** The MT5 connector only calls read functions
> (`copy_rates_*`, `symbol_info*`). A test checks that no trading call exists.
> Nothing here guarantees profit. The 0–100 score measures confluence; it is
> **not** a probability of success. Whether the strategy has an edge on your
> instruments must be established with the backtester on **real** history.

---

## 1. Audit of the starting point

The repository was **empty** (no commits, no files). There was no existing
code to audit, so the project was built from scratch to the requested
architecture. Everything below is new.

### Audit log (second pass)

A backup was taken first (`backups/code_backup_*.tar.gz`, git tag `backup-before-audit-*`).
Each bug below was reproduced before it was fixed, and each has a regression test (`tests/test_audit_fixes.py`).

| Issue found | Impact | Fix |
|---|---|---|
| Tracker skipped the M1 bar that contained the entry moment | a stop hit in the first minute was missed → optimistic live stats | entry-minute bar is included (pessimistic) |
| MT5 offset assumed 0 when started with the market closed | every bar shifted 2–3 h; `download` would write wrong history | persisted offset; error instead of guess; tracking/scanning paused while the market is closed |
| No protection against two running instances | duplicate signals | OS file lock |
| Telegram command thread shared the sender's HTTP session | possible thread-safety errors | separate session |
| Live-loop test passed even with zero signals | weak test | now asserts signals, single delivery, resolution, post-mortems |
| Missing execution-quality gates | signals at rollover spreads / Monday gaps / pre-weekend | abnormal-spread, week-open and week-close gates; gold double-macro-conflict rejection |

## 2. Architecture

```
main.py                      CLI: run | check | scan | download | backtest | walkforward | report | review
config.py                    every parameter, env/.env overridable, validated at start-up
core/
  domain.py                  shared types (Direction, Timeframe, SignalCandidate, RiskPlan, ...)
  market_data.py             DataProvider interface, validation, closed-bar filter, CSV provider
  sessions.py                Sydney/Tokyo/London/New York sessions (DST-correct via zoneinfo)
  technical_analysis.py      EMA/SMA/MACD/ADX/Supertrend/RSI/StochRSI/CCI/ATR/BB/Keltner/volume
  price_action.py            swings, S/R clustering, engulfing/pin bars, breakout-retest, sweeps, false breakouts
  smc_analysis.py            BOS/CHOCH, order blocks, FVGs, liquidity pools, EQH/EQL, premium/discount, inducement
  fundamental_analysis.py    news blackout gate, gold macro context (DXY / US yields), conflict warnings
  signal_engine.py           MTF pipeline + weighted 0..100 scoring (shared by live and backtest)
  risk_management.py         structural ATR-buffered SL, R-multiple TPs, obstacle check, portfolio limits
  trade_state.py             outcome state machine (partials, BE, trailing, pessimistic intrabar)
  signal_tracker.py          persists signals and advances them with M1 bars
  performance_analyzer.py    win rate / PF / expectancy / drawdown / breakdowns
  trade_review.py            post-mortems + recommendations (never auto-applied)
  orchestrator.py            live loop, reconnects, alerts, daily tasks, graceful shutdown
integrations/
  mt5_connector.py           MetaTrader5 package, server-time→UTC, symbol suffix resolution, reconnect
  telegram_bot.py            Bot API, dedup, 429 handling, retries, outbox, dry-run, /status commands
  telegram_messages.py       message templates
  economic_calendar.py       ForexFactory weekly JSON / CSV / none, cached, UTC-normalised
  news_provider.py           Finnhub headlines (optional)
  macro_data.py              FRED yields (optional), DXY from MT5 (optional)
database/                    SQLAlchemy models; SQLite default, PostgreSQL via DATABASE_URL
backtesting/
  historical_data.py         MT5-export/generic CSV loader, resampling, MT5 downloader, synthetic data (tests only)
  backtest_engine.py         event-driven backtest reusing the live engine
  strategy_optimizer.py      walk-forward validation + fixed-parameter holdout
tests/                       78 tests (unit, no-look-ahead, integration, end-to-end, audit regressions)
```

### Decision pipeline (per symbol, on every closed M5 bar)

| Step | Timeframes | What happens |
|---|---|---|
| Hard gates | – | allowed session, spread ≤ max **and** ≤ 3× the day's median spread, no signals in the first 60 min after the weekly open or the last 120 min before the weekly close, enough history, daily/active/currency limits, cooldown, consecutive-loss pause |
| HTF bias | D1 25 %, H4 40 %, H1 35 % | EMA 50/200 position and slope, Supertrend, SMC structure → direction or **wait** |
| Setup | M15 | price **at** a valid OB/FVG, liquidity sweep, breakout-retest, or EMA pullback in trend |
| Confirmation | M5 (M1 optional) | CHOCH/BOS or engulfing/pin bar in the trade direction |
| Fundamentals | – | blackout around high-impact news (wider for gold tier-1 USD events); gold: one DXY/yield conflict = warning, both = reject |
| Risk plan | M15 ATR | SL beyond structure + 0.25 ATR + spread (1–3 ATR, wider ⇒ reject); TPs 1R/2R/3R (TP3 extended to liquidity); reject if opposing liquidity < 2R away |
| Score | – | 9 weighted components → 0–100; signal only if ≥ `MIN_SIGNAL_SCORE` (70) |

No trigger means no signal, whatever the score. This is the main defence against
"signals just to stay active".

### Important technical decisions

* **One engine for live and backtest.** `SignalEngine.evaluate` receives
  closed bars only. The backtester replays history through the same function,
  so backtest results describe the code that actually runs live.
* **No look-ahead.** Indicators are causal; a test recomputes every feature
  on truncated data and compares. Swings exist only after their confirmation
  bars. HTF bars are passed only once closed (tested), and trades are
  evaluated from the next bar.
* **Pessimistic outcome accounting.** If one bar touches SL and TP, the SL
  counts. Buys exit at bid, sells at ask. Slippage and commission are
  deducted in backtests.
* **One signal = one trade with partial exits.** TP1/TP2/TP3 are milestones,
  not three wins (default 50/30/20 % with BE after TP1). Win rate =
  wins ÷ resolved trades; cancelled signals are excluded from it but their R
  is included in net R.
* **"No data" ≠ "no news".** If the calendar cannot be loaded, new signals
  are **blocked** by default (`BLOCK_IF_CALENDAR_UNAVAILABLE=true`).
* **DST-correct history.** Vantage servers run on New York time + 7h, so the offset changes with US DST.
  In `ny_close` mode (auto-detected), each bar is converted with the offset valid on its own date.
  A constant offset would shift every bar on the other side of a DST change by one hour.
  Backtests also build H4/D1 bars from the New York close, like the broker does.
* **Broker time.** MT5 timestamps are in server time (Vantage: GMT+2/+3).
  The connector auto-detects the offset from live ticks (re-checked hourly for
  DST) and saves it to `data/mt5_server_offset.json`, so a restart while the
  market is closed still converts correctly. With no fresh tick and no saved
  value it raises an error instead of guessing.
* **Single instance.** `main.py run` holds an OS file lock (`data/bot.lock`);
  a second instance refuses to start, so duplicates can't come from two bots.
* **Weak SMC zones are scored down, not used.** OBs with low displacement,
  repeated touches, no FVG, or price already away from the zone (> 0.5 ATR)
  score lower, or are not counted as a setup at all.
* **Learning without auto-tuning.** `trade_review` reports only findings with
  ≥ 30 samples that hold in **both** earlier and later halves of the history.
  It stores them as `PENDING_REVIEW`. Parameters change only when you edit
  `.env`, ideally after a walk-forward run.

## 3. Setup on a Windows VPS (MT5 + Vantage)

1. Install the Vantage MT5 terminal, log in, and add all symbols to **Market Watch**.
   Note any suffix (e.g. `XAUUSD+`).
2. Install Python 3.11+ (64-bit). Then:
   ```
   git clone <repo> fxbot && cd fxbot
   python -m venv .venv && .venv\Scripts\activate
   pip install -r requirements.txt
   copy .env.example .env      (then edit it)
   ```
3. Create a Telegram bot with **@BotFather** → `TELEGRAM_BOT_TOKEN`. Send it a
   message and read your chat id from `https://api.telegram.org/bot<token>/getUpdates`
   → `TELEGRAM_CHAT_ID`.
4. Diagnostics: `python main.py diagnose` (full report, see §3a) or the quick `python main.py check`.
5. **Validate before going live** (see §4).
6. Run: `python main.py run`. To keep it running 24/5, register it as a service
   (e.g. NSSM: `nssm install fxbot <path>\.venv\Scripts\python.exe main.py run`,
   working directory = project folder) or a Task Scheduler task "at startup",
   restart on failure. The MT5 terminal must stay open and logged in.

Logs: `logs/system.log`, `logs/signals.log`. Without Telegram credentials all
messages go to `logs/telegram_dryrun.log`.

Telegram commands (from your chat only): `/status`, `/active`, `/report`, `/help`.

### 3a. Connection diagnostic (`python main.py diagnose`)

This is read-only and sends no orders. It checks, in order:
1. environment and credentials mode;
2. that the data source is the **official** MetaTrader5 package, not CSV, synthetic or mock data;
3. the trading lock, which makes `order_send` and every other trading call raise `PermissionError`, and the demo-only guard;
4. terminal connection, account (DEMO, Vantage, balance, equity, currency, leverage; login masked, holder name never read);
5. the gold symbol, including any suffix;
6. live bid/ask/spread and the server time offset;
7. XAUUSD M5/M15/H1/H4/D1 history (default 60 days, saved to `data/history/`): duplicates, invalid bars, future
   timestamps, weekend / daily-break / unexpected gaps, freshness;
8. consistency of broker M15/H1/H4/D1 bars with bars rebuilt from M5, which proves the time conversion and the
   backtest's H4/D1 alignment;
9. Telegram (`getMe` + one test message; skip it with `--no-telegram`);
10. the economic calendar;
11. one signal-engine pass per symbol on live data (nothing is sent or stored).

The redacted report is written to `logs/diagnostics/diagnostic_<time>.md`.

Safety guards that always apply:
- `MT5_REQUIRE_DEMO=true`: the bot refuses to connect to a non-demo account.
- `main.py run` refuses `DATA_PROVIDER=csv`; with `--offline`, Telegram is forced off.

## 4. Backtesting & validation workflow

```
python main.py download --symbol XAUUSD --days 730 --timeframes M5,M1
python main.py backtest --symbol XAUUSD --data data/history/XAUUSD_M5.csv --calendar data/calendar_hist.csv
python main.py walkforward --symbol XAUUSD --data data/history/XAUUSD_M5.csv --train-days 120 --test-days 30
```

* An MT5 "Export bars" file also works: pass `--server-offset 2` (or 3), since its times are server time.
* The bot does **not** ship a historical economic calendar (no free reliable
  archive was available to integrate). Without `--calendar` the backtest runs
  **without** the news filter, and the report says so. The template is `data/calendar_template.csv`.
* Gold DXY/yield context is not reproduced historically. In backtests the
  fundamentals component is neutral.
* Trust only the **out-of-sample** section of the walk-forward. A large gap
  between in-sample and out-of-sample results is flagged as likely overfitting.

## 5. What requires your input / external services

| Item | Needed for | How |
|---|---|---|
| Windows VPS + MT5 terminal (Vantage) | live data | `MetaTrader5` pip package is Windows-only |
| Telegram bot token + chat id | notifications | @BotFather |
| Historical M5/M1 data | backtests | `python main.py download` on the VPS |
| ForexFactory feed reachable | news blackout | default; unofficial feed, cached, rate-limited |
| `FRED_API_KEY` (free) | gold: US real/nominal yields | fred.stlouisfed.org |
| `DXY_SYMBOL` | gold: dollar index | only if your Vantage account offers a DXY CFD |
| `FINNHUB_API_KEY` (free tier) | headline alerts | finnhub.io |
| Historical calendar CSV | news filter inside backtests | export from a calendar provider you trust |

## 6. Testing

```
pytest -m "not slow"      # 75 fast tests (~5 s)
pytest -m slow            # end-to-end backtest, gold-gate scan, live-loop replay on synthetic data (~3 min)
```

Synthetic data is used **only** to test mechanics. The numbers it produces
say nothing about real performance and must never be quoted as results.

## 7. Roadmap

**Done (this iteration):** full architecture; MT5 read-only connector; MTF
TA/PA/SMC engine; news blackout; gold macro hooks; risk engine; Telegram with
dedup/retry/outbox/commands; SQLite tracking; daily report; trade review;
backtester; walk-forward; tests.

**Next, in priority order:**
1. Run `check` on the VPS. Confirm symbol names, spreads and server offset against MT5.
2. Download ≥ 2 years of M5 + M1 for every instrument. Run backtest + walk-forward per instrument
   and disable instruments whose out-of-sample expectancy is ≤ 0.
3. Run in production with Telegram for 2–4 weeks as a forward test. Compare live vs backtest statistics.
4. Add a historical calendar source so the news filter can be backtested.
5. Optional: per-instrument parameter profiles (only if walk-forward supports them), a correlation
   filter between EUR/GBP/AUD/NZD USD pairs, and a PostgreSQL migration if several bots share one DB.
