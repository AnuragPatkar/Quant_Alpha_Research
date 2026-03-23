"""
quant_alpha/utils/__init__.py
==============================
Shared utility layer for the entire quant_alpha package.

Provides a single import point so consumers never need to know which
sub-module each symbol lives in:

    from quant_alpha.utils import (
        setup_logging, load_parquet, save_parquet,
        time_execution, calculate_returns
    )

    # Also available:
    from quant_alpha.utils import timer, retry
    from quant_alpha.utils import safe_col
    from quant_alpha.utils import get_trading_days, align_dates
    from quant_alpha.utils import calculate_sharpe, calculate_max_drawdown

Sub-modules
-----------
    math_utils     — return calculation, Sharpe, Sortino, drawdown
    date_utils     — trading-day calendars, alignment, tz normalisation
    io_utils       — Parquet save/load with pyarrow → fastparquet fallback
    decorators     — @time_execution / @timer and @retry with backoff
    column_helpers — safe_col() for defensive column access
    preprocessing  — WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb
                     (import directly: from quant_alpha.utils.preprocessing import ...)

NOTE: WinsorisationScaler and SectorNeutralScaler live in
      quant_alpha/utils/preprocessing.py and are imported directly from there
      throughout the codebase. They are NOT re-exported here to avoid circular
      imports (preprocessing.py depends on config which bootstraps utils).
"""

from __future__ import annotations

# ── Logging ───────────────────────────────────────────────────────────────────
# Used everywhere as: from quant_alpha.utils import setup_logging
from config.logging_config import setup_logging          # noqa: F401
        

# ── Math / Finance ────────────────────────────────────────────────────────────
from .math_utils import (                           # noqa: F401
    calculate_returns,
    calculate_log_returns,
    calculate_sharpe,
    calculate_sortino,
    calculate_drawdown,
    calculate_max_drawdown,
)

# ── Date / Calendar ───────────────────────────────────────────────────────────
from .date_utils import (                           # noqa: F401
    get_trading_days,
    align_dates,
    get_previous_trading_day,
    next_trading_date,
    date_range,
    is_trading_day,
    get_market_calendar,
)

# ── I/O ───────────────────────────────────────────────────────────────────────
from .io_utils import save_parquet, load_parquet    # noqa: F401

# ── Decorators ────────────────────────────────────────────────────────────────
# time_execution is the canonical name; timer is the alias.
# generate_predictions.py uses:  from quant_alpha.utils import save_parquet, time_execution
# decorators.py defines timer = time_execution as an alias
from .decorators import time_execution, timer, retry   # noqa: F401

# ── Column helpers ────────────────────────────────────────────────────────────
from .column_helpers import safe_col                # noqa: F401


__all__ = [
    # Logging
    "setup_logging",
    # Math
    "calculate_returns",
    "calculate_log_returns",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_drawdown",
    "calculate_max_drawdown",
    # Date
    "get_trading_days",
    "align_dates",
    "get_previous_trading_day",
    "next_trading_date",
    "date_range",
    "is_trading_day",
    "get_market_calendar",
    # I/O
    "save_parquet",
    "load_parquet",
    # Decorators
    "time_execution",
    "timer",
    "retry",
    # Columns
    "safe_col",
]