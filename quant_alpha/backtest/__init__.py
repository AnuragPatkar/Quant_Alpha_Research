"""
Backtest Module
===============
Portfolio simulation and performance analysis.

Classes:
    - Backtester: Main backtesting engine
    - BacktestResult: Results container
    - PerformanceMetrics: Performance calculations
    - RiskMetrics: Risk calculations
    - PortfolioAnalyzer: Portfolio analysis

Functions:
    - calculate_metrics: Quick metrics calculation
    - analyze_portfolio: Quick portfolio analysis
    - print_metrics_summary: Print formatted metrics
    - print_portfolio_summary: Print formatted portfolio stats
    
Author: Senior Quant Team
Version: 1.0.0
"""

from .engine import Backtester, BacktestResult
from .metrics import (
    PerformanceMetrics, 
    RiskMetrics, 
    calculate_metrics, 
    print_metrics_summary
)
from .portfolio import (
    PortfolioAnalyzer, 
    analyze_portfolio, 
    print_portfolio_summary
)

__all__ = [
    # Main classes
    'Backtester',
    'BacktestResult',
    'PerformanceMetrics',
    'RiskMetrics',
    'PortfolioAnalyzer',
    
    # Convenience functions
    'calculate_metrics',
    'analyze_portfolio',
    'print_metrics_summary',
    'print_portfolio_summary'
]

__version__ = "1.0.0"