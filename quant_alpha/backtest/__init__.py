"""
Backtest Simulation Subsystem
=============================

Provides event-driven historical simulation, performance attribution, 
and institutional-grade risk metrics.

Purpose
-------
This module exposes the unified public API for historical strategy evaluation, 
accounting for realistic transaction costs, slippage, and portfolio limits.

Role in Quantitative Workflow
-----------------------------
Serves as the rigorous ex-post validation layer, transforming alpha signals 
and optimized weights into simulated equity curves to evaluate out-of-sample 
execution viability before capital deployment.
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