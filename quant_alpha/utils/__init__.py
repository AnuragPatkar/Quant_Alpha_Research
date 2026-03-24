"""
Core Utility Subsystem
======================

Provides a unified integration point for cross-cutting quantitative utilities, 
abstracting underlying sub-module complexities from the high-level algorithmic 
trading and feature engineering engines.

Purpose
-------
This module exposes an aggregated API for mathematical operations, temporal 
alignments, robust data serialization, and execution profiling. By consolidating 
these foundational primitives, it enforces strict standardized operational 
procedures across both the research and production environments.

Role in Quantitative Workflow
-----------------------------
Serves as the foundational dependency layer for all high-order systems 
(e.g., Portfolio Allocators, Signal Generators). It guarantees consistency 
in critical calculations like objective scaling, look-ahead bias prevention 
via strict date alignment, and fault-tolerant disk I/O.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Structural matrix manipulations and cross-sectional indices.
- **PyArrow/FastParquet**: High-performance columnar memory persistence.
- **Numba**: Targeted JIT compilation execution paths (explicitly excluded 
  from global exports to mitigate initialization overhead).
"""

from __future__ import annotations

# ── Logging Subsystem ─────────────────────────────────────────────────────────
# Exposes centralized logging configuration to standardize execution telemetry
from config.logging_config import setup_logging          # noqa: F401
        

# ── Quantitative Financial Mathematics ────────────────────────────────────────
# Standardized performance metrics, risk attribution, and continuous vector operators
from .math_utils import (                           # noqa: F401
    calculate_returns,
    calculate_log_returns,
    calculate_sharpe,
    calculate_sortino,
    calculate_drawdown,
    calculate_max_drawdown,
)

# ── Temporal Alignment & Market Calendars ─────────────────────────────────────
# Strict chronological boundaries preventing look-ahead bias and holiday artifacts
from .date_utils import (                           # noqa: F401
    get_trading_days,
    align_dates,
    get_previous_trading_day,
    next_trading_date,
    date_range,
    is_trading_day,
    get_market_calendar,
)

# ── Data Persistence & I/O ────────────────────────────────────────────────────
# Fault-tolerant columnar serialization engines supporting resilient data lakes
from .io_utils import save_parquet, load_parquet    # noqa: F401

# ── Aspect-Oriented Execution Controls ────────────────────────────────────────
# High-order instrumentation for performance profiling and exponential backoff
from .decorators import time_execution, timer, retry   # noqa: F401

# ── Feature Schema Safeguards ─────────────────────────────────────────────────
# Defensive matrix access primitives mitigating malformed temporal schema faults
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