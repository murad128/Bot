# Live dashboard (read-only)

A small separate process that shows the bot's real data. It never trades, never changes settings,
never reads `.env`, and never connects to MT5 or Telegram.

```
cd C:\fxbot
.venv\Scripts\python.exe dashboard\server.py --bot C:\fxbot
```
Then open `http://127.0.0.1:8765`. Stop it with Ctrl+C. The bot keeps running independently.

## Secure access from your own computer
The server binds **127.0.0.1 only** and rejects foreign `Host` headers, so it is unreachable from the
internet. Do **not** open a firewall port for it. Use one of:

1. **RDP (no new port):** open the dashboard in a browser inside your existing RDP session on the VPS.
2. **SSH tunnel** (only if SSH to the VPS is already set up):
   `ssh -N -L 8765:127.0.0.1:8765 <user>@<vps>` on your computer, then open `http://127.0.0.1:8765` locally.

## What it shows and where it comes from
| Panel | Source |
|---|---|
| Instruments, pipeline stage, score, direction | Last evaluation line per instrument in `logs/system.log` (the bot logs a result when it changes; only the first failing filter is logged) |
| MT5 / bot activity | `MT5 connected` / disconnect / shutdown lines in `logs/system.log` |
| Telegram | Last sent notification and failed-message outbox in `bot.db` |
| News filter / calendar | `data/calendar_cache.json` and the bot's calendar refresh log lines; the test-suite event is flagged |
| Active signals, Entry/SL/TP1-3, MFE/MAE, HTF trend | `signals` table in `bot.db` (opened read-only) |
| SMC snapshot | `signal_snapshots` (L1.1b) when present |
| Performance, by instrument, recent outcomes | `signals` table, using the bot's own `compute_stats` |
| Learning | Counts from the learning tables (snapshots, settings versions, calendar archive, recommendations) |

Missing data is shown as **N/A**; every item shows how old it is. The score is a confluence measure, not a probability.
