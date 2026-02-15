"""
Technical Features Module
Exposes sub-modules to ensure FactorRegistry auto-discovery works.
"""

# Momentum Factors (15 total)
from .momentum import (
    Return5D, Return10D, Return21D, Return63D, Return126D, Return252D,
    MomentumAcceleration10D, MomentumAcceleration21D,
    RSI14D, RSI21,
    MACD, MACDSignal,
    StochasticOscillator,
    WilliamsR,
    TSI,
    # Removed RateOfChange12D (redundant with RateOfChange20D)
    RateOfChange20D,
    ADX14
)

# Volatility Factors (10 total)
from .volatility import (
    # Removed Volatility5D (too short-term, noisy)
    # Removed Volatility10D (redundant with Volatility21D)
    Volatility21D, Volatility63D,
    # Removed Volatility126D (less useful, regime-like)
    GKVolatility21D, GKVolatility63D,
    ATR14, ATR21,
    VolatilityRatio,
    Skewness21D,
    Kurtosis21D
)

# Mean Reversion Factors (12 total)
from .mean_reversion import (
    # Removed DistSMA10D (too short-term, redundant with DistSMA21D)
    DistSMA21D, DistSMA50D, DistSMA200D,
    # Removed ZScore10D (too short-term, redundant with ZScore21D)
    ZScore21D, ZScore63D,
    MeanRevBBPosition, MeanRevBBWidth,
    MACrossover5_21, MACrossover21_63,
    PriceToHighLow52W,
    CCI
)

# Volume Factors (9 total)
from .volume import (
    # Removed VolumeZScore5D (too short-term, noisy)
    VolumeZScore21D,
    VolumeMA20Ratio,
    TurnoverRate,
    Amihud63D,
    # Removed VWAPDistance (redundant with price-based signals)
    # Removed OnBalanceVolumeSlope (OBV is lagging indicator)
    PriceVolumeCorr21D,
    ForceIndex14D,
    # Removed EaseOfMovement14 (derivative of ATR, somewhat redundant)
    ChaikinMoneyFlow21D,
    MoneyFlowIndex14,
    AccumulationDistribution
)

__all__ = [
    # Momentum (15 factors)
    'Return5D', 'Return10D', 'Return21D', 'Return63D', 'Return126D', 'Return252D',
    'MomentumAcceleration10D', 'MomentumAcceleration21D',
    'RSI14D', 'RSI21',
    'MACD', 'MACDSignal',
    'StochasticOscillator',
    'WilliamsR',
    'TSI',
    'RateOfChange20D',
    'ADX14',
    
    # Volatility (10 factors)
    'Volatility21D', 'Volatility63D',
    'GKVolatility21D', 'GKVolatility63D',
    'ATR14', 'ATR21',
    'VolatilityRatio',
    'Skewness21D',
    'Kurtosis21D',
    
    # Mean Reversion (12 factors)
    'DistSMA21D', 'DistSMA50D', 'DistSMA200D',
    'ZScore21D', 'ZScore63D',
    'MeanRevBBPosition', 'MeanRevBBWidth',
    'MACrossover5_21', 'MACrossover21_63',
    'PriceToHighLow52W',
    'CCI',
    
    # Volume (9 factors)
    'VolumeZScore21D',
    'VolumeMA20Ratio',
    'TurnoverRate',
    'Amihud63D',
    'PriceVolumeCorr21D',
    'ForceIndex14D',
    'ChaikinMoneyFlow21D',
    'MoneyFlowIndex14',
    'AccumulationDistribution'
]