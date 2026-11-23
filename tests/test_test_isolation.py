"""P0-P4 regression tests: the suite must never touch the real logs/data/backups, leak handles, or use
pandas APIs deprecated for pandas 4."""
import logging
import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL = [ROOT / "logs", ROOT / "data", ROOT / "backups"]


def _inside_repo_data(p) -> bool:
    p = Path(str(p).replace("sqlite:///", "")).resolve()
    return any(p == r or r in p.parents for r in REAL)


# ----------------------------------------------------------------------------- P0: isolation
def test_bare_config_never_points_at_real_logs_data_or_backups():
    from config import Config
    c = Config()                                   # the pattern that leaked before (type(cfg)())
    for path in (c.log_dir, c.database_url, c.runtime.backup_dir, c.csv_data_dir, c.news.calendar_csv_path):
        assert not _inside_repo_data(path), f"test config points into the real repository folder: {path}"


def test_hard_coded_base_dir_is_redirected_for_tests():
    import core.diagnostics
    import core.orchestrator
    import main
    for mod in (main, core.orchestrator, core.diagnostics):
        assert Path(mod.BASE_DIR).resolve() != ROOT, f"{mod.__name__}.BASE_DIR still points at the repository"


def test_cli_with_a_fresh_config_logs_outside_the_repository(monkeypatch, tmp_path):
    """Regression: main.main() with a fresh Config attached file handlers to the real logs/ folder."""
    import main
    from config import Config
    other = Config()
    other.database_url = f"sqlite:///{tmp_path / 'x.db'}"
    other.runtime.backup_dir = str(tmp_path / "backups")
    monkeypatch.setattr(main, "load_config", lambda: other)
    main.main(["backup", "--list"])
    files = [h.baseFilename for lg in (logging.getLogger(), logging.getLogger("signals"))
             for h in lg.handlers if hasattr(h, "baseFilename")]
    assert files, "setup_logging attached no file handler"
    assert not any(_inside_repo_data(f) for f in files), files


# ----------------------------------------------------------------------------- P2: logging handlers
def test_setup_logging_closes_old_handlers_and_never_duplicates(cfg, tmp_path):
    import main
    main.setup_logging(cfg)
    first_root = [h for h in logging.getLogger().handlers if hasattr(h, "baseFilename")]
    first_sig = list(logging.getLogger("signals").handlers)
    cfg.log_dir = str(tmp_path / "logs2")
    main.setup_logging(cfg)
    sig = logging.getLogger("signals").handlers
    assert len(sig) == 1 and sig[0] not in first_sig                       # replaced, not stacked
    assert len([h for h in logging.getLogger().handlers if hasattr(h, "baseFilename")]) == 1
    for h in first_root + first_sig:                                       # old file handles are closed
        assert h.stream is None or h.stream.closed


# ----------------------------------------------------------------------------- P1/P3: pandas 4
def test_no_pandas4_deprecated_dayofweek_in_the_code():
    deprecated = re.compile(r"\." + "dayof" + r"week\b")         # built in pieces so this file never matches
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in ROOT.rglob("*.py")
            if not any(part.startswith(".venv") or part == "site-packages" for part in p.parts)
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if deprecated.search(line)]
    assert not hits, f"use .day_of_week, pandas 4 removes the old attribute: {hits}"


def test_gap_classification_and_synthetic_data_raise_no_pandas_deprecation():
    from backtesting.historical_data import synthetic_ohlc
    from core.diagnostics import classify_gaps
    from core.domain import Timeframe
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.Pandas4Warning)
        df = synthetic_ohlc(n=3000, tf=Timeframe.M5, seed=1)
        gaps = classify_gaps(df, Timeframe.M5)
    assert gaps["weekend"], "the synthetic data spans weekends"


# ----------------------------------------------------------------------------- P4: ResourceWarnings are surfaced
def test_resource_warnings_are_not_silently_ignored(pytestconfig):
    filters = pytestconfig.getini("filterwarnings")
    assert any(f.strip().startswith(("always::ResourceWarning", "error::ResourceWarning")) for f in filters)
    assert any(f.strip() == "error::pandas.errors.Pandas4Warning" for f in filters)
