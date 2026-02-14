"""
Technical Features Module
Exposes sub-modules to ensure FactorRegistry auto-discovery works.
"""

# Momentum Factors
from .momentum import (
    Return5D, Return10D, Return21D, Return63D, Return126D, Return252D,
    MomentumAcceleration10D, MomentumAcceleration21D,
    RSI14D, RSI21,
    MACD, MACDSignal,
    StochasticOscillator,
    WilliamsR,
    TSI,
    RateOfChange12D, RateOfChange20D,
    ADX14
)

# Volatility Factors
from .volatility import (
    Volatility5D, Volatility10D, Volatility21D, Volatility63D, Volatility126D,
    GKVolatility21D, GKVolatility63D,
    ATR14, ATR21,
    VolatilityRatio,
    Skewness21D,
    Kurtosis21D
)

# Mean Reversion Factors
from .mean_reversion import (
    DistSMA10D, DistSMA21D, DistSMA50D, DistSMA200D,
    ZScore10D, ZScore21D, ZScore63D,
    MeanRevBBPosition, MeanRevBBWidth,
    MACrossover5_21, MACrossover21_63,
    PriceToHighLow52W,
    CCI
)

# Volume Factors
from .volume import (
    VolumeZScore5D, VolumeZScore21D,
    VolumeMA20Ratio,
    TurnoverRate,
    Amihud63D,
    VWAPDistance,
    OnBalanceVolumeSlope,
    PriceVolumeCorr21D,
    ForceIndex14D,
    EaseOfMovement14,
    ChaikinMoneyFlow21D,
    MoneyFlowIndex14,
    AccumulationDistribution
)

__all__ = [
    # Momentum
    'Return5D', 'Return10D', 'Return21D', 'Return63D', 'Return126D', 'Return252D',
    'MomentumAcceleration10D', 'MomentumAcceleration21D',
    'RSI14D', 'RSI21',
    'MACD', 'MACDSignal',
    'StochasticOscillator',
    'WilliamsR',
    'TSI',
    'RateOfChange12D', 'RateOfChange20D',
    'ADX14',
    
    # Volatility
    'Volatility5D', 'Volatility10D', 'Volatility21D', 'Volatility63D', 'Volatility126D',
    'GKVolatility21D', 'GKVolatility63D',
    'ATR14', 'ATR21',
    'VolatilityRatio',
    'Skewness21D',
    'Kurtosis21D',
    
    # Mean Reversion
    'DistSMA10D', 'DistSMA21D', 'DistSMA50D', 'DistSMA200D',
    'ZScore10D', 'ZScore21D', 'ZScore63D',
    'MeanRevBBPosition', 'MeanRevBBWidth',
    'MACrossover5_21', 'MACrossover21_63',
    'PriceToHighLow52W',
    'CCI',
    
    # Volume
    'VolumeZScore5D', 'VolumeZScore21D',
    'VolumeMA20Ratio',
    'TurnoverRate',
    'Amihud63D',
    'VWAPDistance',
    'OnBalanceVolumeSlope',
    'PriceVolumeCorr21D',
    'ForceIndex14D',
    'EaseOfMovement14',
    'ChaikinMoneyFlow21D',
    'MoneyFlowIndex14',
    'AccumulationDistribution'
]