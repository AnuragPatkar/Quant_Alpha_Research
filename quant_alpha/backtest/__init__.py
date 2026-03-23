"""
quant_alpha.backtest
====================
Event-driven backtesting engine, performance metrics, and trade attribution.

Public API
----------
    from quant_alpha.backtest import BacktestEngine
    from quant_alpha.backtest import compute_metrics, print_metrics_report
    from quant_alpha.backtest import SimpleAttribution, FactorAttribution
"""

from .engine      import BacktestEngine
from .metrics     import compute_metrics, print_metrics_report
from .attribution import SimpleAttribution, FactorAttribution
from .utils       import set_style, format_currency

__all__ = [
    "BacktestEngine",
    "compute_metrics",
    "print_metrics_report",
    "SimpleAttribution",
    "FactorAttribution",
    "set_style",
    "format_currency",
]