"""
Composite Feature Engineering Subsystem
=======================================

Provides advanced composite alpha factors capturing non-linear cross-asset 
interactions and macro-adjusted systematic regime indicators.

Purpose
-------
Aggregates technical, fundamental, and alternative datasets into high-level 
regime classifiers and context-aware predictive signals. These factors adjust 
their signal strength dynamically based on the broader macroeconomic environment.

Role in Quantitative Workflow
-----------------------------
Acts as the systemic overlay mapping local equity-specific alpha factors to 
global market regimes (e.g., Risk-On/Risk-Off). Essential for training 
regime-conditional decision trees.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Vectorized cross-asset condition tracking and multi-factor 
  equal-weight scalar combinations.
"""

from .macro_adjusted import (
    MacroAdjustedMomentum,
    OilCorrectedValue,
    RateEnvironmentScore,
    DollarAdjustedGrowth,
    RiskParityBlend,
)

from .system_health import (
    MarketRegimeScore,
    VolatilityRegime,
    CapitalFlowSignal,
    EconomicMomentumScore,
    PortfolioHealthIndex,
)

from .smart_signals import (
    MomentumVIXDivergence,
    ValueYieldCombo,
    QualityInDownturn,
    EarningsMacroAlignment,
    MultiAssetOpportunity,
)

__all__ = [
    # Macro-Adjusted
    'MacroAdjustedMomentum',
    'OilCorrectedValue',
    'RateEnvironmentScore',
    'DollarAdjustedGrowth',
    'RiskParityBlend',
    # System Health
    'MarketRegimeScore',
    'VolatilityRegime',
    'CapitalFlowSignal',
    'EconomicMomentumScore',
    'PortfolioHealthIndex',
    # Smart Signals
    'MomentumVIXDivergence',
    'ValueYieldCombo',
    'QualityInDownturn',
    'EarningsMacroAlignment',
    'MultiAssetOpportunity',
]
