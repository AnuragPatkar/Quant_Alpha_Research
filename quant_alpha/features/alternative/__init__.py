"""
Alternative Feature Engineering Subsystem
=========================================

Provides structural extraction algorithms mapping non-equity macroeconomic, 
sentiment, and inflationary signals into continuous cross-sectional alpha metrics.

Purpose
-------
Aggregates exogenous data structures (Commodities, Treasuries, VIX, FX) 
into mathematically normalized predictors, enabling regime-conditional modeling 
and systematic risk overlay adjustments.

Role in Quantitative Workflow
-----------------------------
Acts as the primary systematic overlay, injecting macroeconomic orthogonality 
into traditional equity-centric factor ensembles. These signals help identify 
structural regime shifts (e.g., Stagflation, Risk-Off capitulation).

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Vectorized signal normalization, cross-asset correlations, 
  and distributional Z-scoring.
"""

from .macro import (
    OilMomentum,
    USDStrength,
    YieldTrend,
    MacroEconomicScore,
)

from .sentiment import (
    VIXLevel,
    VIXMomentum,
    RiskOnOffSignal,
    VolatilityStressIndex,
)

from .inflation import (
    OilUSDRatio,
    YieldMomentum,
    InflationProxyScore,
    GrowthInflationMix,
)

__all__ = [
    # Macro
    'OilMomentum',
    'USDStrength',
    'YieldTrend',
    'MacroEconomicScore',
    # Sentiment
    'VIXLevel',
    'VIXMomentum',
    'RiskOnOffSignal',
    'VolatilityStressIndex',
    # Inflation
    'OilUSDRatio',
    'YieldMomentum',
    'InflationProxyScore',
    'GrowthInflationMix',
]
