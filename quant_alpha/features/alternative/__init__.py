"""
Alternative Data Factors (12 Total)

Macro Factors (4):
- OilMomentum: Oil price momentum (commodity cycle indicator)
- USDStrength: USD index normalized strength (currency sentiment)
- YieldTrend: 10Y yield trend (growth expectations)
- MacroEconomicScore: Composite macro health (0-100)

Sentiment Factors (4):
- VIXLevel: Implied volatility absolute level (fear gauge)
- VIXMomentum: Change in VIX (fear direction)
- RiskOnOffSignal: Binary risk appetite signal
- VolatilityStressIndex: Combined fear+stress (0-100)

Inflation Factors (4):
- OilUSDRatio: Oil-USD ratio (inflation vs currency)
- YieldMomentum: Rate of change in 10Y yields
- InflationProxyScore: Combined oil+rates inflation signal (0-100)
- GrowthInflationMix: Growth vs inflation balance

Total: 12 factors
Inheritance: All inherit from AlternativeFactor
Registration: All auto-registered via @FactorRegistry.register()
Data Sources: OIL.csv, SP500.csv, VIX.csv, USD.csv, US_10Y.csv (5 macro/sentiment series)
Frequency: Daily, 2000+ records each
Focus: Macro momentum, sentiment extremes, inflation expectations
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
