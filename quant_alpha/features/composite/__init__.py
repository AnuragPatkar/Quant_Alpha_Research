"""
Composite Factors (15 Total)

Macro-Adjusted Factors (5):
- MacroAdjustedMomentum: Momentum scaled by volatility regime
- OilCorrectedValue: Value adjusted for commodity environment
- RateEnvironmentScore: Quality performance in current rate regime
- DollarAdjustedGrowth: Growth factors modulated by USD strength
- RiskParityBlend: Equal risk weight momentum + sentiment

System Health Factors (5):
- MarketRegimeScore: Bull/Bear/Sideways classification
- VolatilityRegime: Normal/Elevated/Crisis classification
- CapitalFlowSignal: Oil + USD combined flow indicator
- EconomicMomentumScore: Combined macro strength (0-100)
- PortfolioHealthIndex: Overall system health (0-100)

Smart Signals Factors (5):
- MomentumVIXDivergence: Detect euphoria or capitulation
- ValueYieldCombo: Value + rates + sentiment blend
- QualityInDownturn: Quality performance during stress
- EarningsMacroAlignment: Earnings vs macro alignment
- MultiAssetOpportunity: Cross-asset opportunity scoring (0-100)

Total: 15 factors
Inheritance: All inherit from AlternativeFactor
Registration: All auto-registered via @FactorRegistry.register()
Data Sources: Combines technical data with alternative data (5 macro/sentiment series)
Frequency: Daily, aligned with price data
Focus: Regime adaptation, multi-factor blending, cross-asset coordination
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
