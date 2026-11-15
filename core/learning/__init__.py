"""
Self-learning engine - data capture (L1.1a foundation, L1.1b signal-time snapshots).

Everything here only RECORDS information for later analysis. Nothing in this
package influences signal generation, scoring, risk, MT5 access or Telegram.
Every entry point is called from isolated hooks that catch their own errors.
LEARNING_CAPTURE=false (RuntimeConfig.learning_capture) turns all capture off.
"""
