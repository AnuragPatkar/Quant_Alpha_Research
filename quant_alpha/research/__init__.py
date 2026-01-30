"""
Research Module
===============
Research utilities and analysis tools for alpha models.

This module provides:
- Walk-forward validation (re-exported from models)
- Alpha decay analysis
- Factor analysis and correlations
- Statistical significance testing
- Regime analysis

Usage:
    >>> from quant_alpha.research import (
    ...     WalkForwardValidator,
    ...     analyze_alpha_decay,
    ...     calculate_factor_correlations,
    ...     test_statistical_significance
    ... )
"""

# Re-export WalkForwardTrainer as WalkForwardValidator for backward compatibility
from quant_alpha.models.trainer import (
    WalkForwardTrainer as WalkForwardValidator,
    WalkForwardResults,
    FoldResult,
    run_walk_forward_validation,
)

from .analysis import (
    analyze_alpha_decay,
    calculate_factor_correlations,
    calculate_factor_turnover,
    analyze_factor_returns,
    get_redundant_factors,
)

from .significance import (
    test_statistical_significance,
    calculate_t_statistic,
    bootstrap_confidence_interval,
    calculate_deflated_sharpe,
)

from .regime import (
    identify_market_regimes,
    analyze_regime_performance,
    calculate_regime_metrics,
)

__all__ = [
    # Validation (re-exported)
    'WalkForwardValidator',
    'WalkForwardResults',
    'FoldResult',
    'run_walk_forward_validation',
    
    # Analysis
    'analyze_alpha_decay',
    'calculate_factor_correlations',
    'calculate_factor_turnover',
    'analyze_factor_returns',
    'get_redundant_factors',
    
    # Significance testing
    'test_statistical_significance',
    'calculate_t_statistic',
    'bootstrap_confidence_interval',
    'calculate_deflated_sharpe',
    
    # Regime analysis
    'identify_market_regimes',
    'analyze_regime_performance',
    'calculate_regime_metrics',
]