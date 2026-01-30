"""
Feature Engineering Module for Quant Alpha Research.

This module provides:
- Base classes for factor construction
- Factor registry for managing all factors
- Individual factor implementations (momentum, mean reversion, etc.)
- Cross-sectional normalization utilities

Usage:
    >>> from quant_alpha.features import FactorRegistry, compute_all_features
    >>> registry = FactorRegistry()
    >>> features_df = compute_all_features(price_data)
"""

from .base import (
    BaseFactor,
    FactorInfo,
    FactorCategory,
    FactorValidationError,
)

from .registry import (
    FactorRegistry,
    compute_all_features,
    normalize_cross_section,
    winsorize_series,
)

from .momentum import (
    Momentum,
    MomentumRank,
    get_momentum_factors,
)

from .mean_reversion import (
    RSI,
    DistanceFromMA,
    ZScore,
    BollingerPosition,
    get_mean_reversion_factors,
)

from .microstructure import (
    Volatility,
    VolatilityRank,
    VolumeZScore,
    AmihudIlliquidity,
    get_microstructure_factors,
)

__all__ = [
    # Base classes
    'BaseFactor',
    'FactorInfo',
    'FactorCategory',
    'FactorValidationError',
    
    # Registry
    'FactorRegistry',
    'compute_all_features',
    'normalize_cross_section',
    'winsorize_series',
    
    # Momentum factors
    'Momentum',
    'MomentumRank',
    'get_momentum_factors',
    
    # Mean reversion factors
    'RSI',
    'DistanceFromMA',
    'ZScore',
    'BollingerPosition',
    'get_mean_reversion_factors',
    
    # Microstructure factors
    'Volatility',
    'VolatilityRank',
    'VolumeZScore',
    'AmihudIlliquidity',
    'get_microstructure_factors',
]