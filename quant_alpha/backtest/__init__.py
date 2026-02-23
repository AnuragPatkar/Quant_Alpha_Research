"""
Backtesting Package
Complete realistic backtesting infrastructure

Modules:
- engine: Main backtest loop
- portfolio: Portfolio construction
- execution: Trade execution simulation
- market_impact: Almgren-Chriss model
- metrics: Performance analytics
- attribution: Factor attribution
- risk_manager: Risk controls
- utils: Helper functions
"""

from .engine import BacktestEngine
from .portfolio import Portfolio
from .execution import ExecutionSimulator
from .market_impact import AlmgrenChrissImpact, SimpleImpactModel
from .metrics import PerformanceMetrics, print_metrics_report
from .attribution import FactorAttribution, SimpleAttribution
from .risk_manager import RiskManager
from .utils import (
    calculate_returns,
    cumulative_returns,
    align_dates,
    validate_backtest_data
)

__all__ = [
    'BacktestEngine',
    'Portfolio',
    'ExecutionSimulator',
    'AlmgrenChrissImpact',
    'SimpleImpactModel',
    'PerformanceMetrics',
    'print_metrics_report',
    'FactorAttribution',
    'SimpleAttribution',
    'RiskManager',
    'calculate_returns',
    'cumulative_returns',
    'align_dates',
    'validate_backtest_data',
]