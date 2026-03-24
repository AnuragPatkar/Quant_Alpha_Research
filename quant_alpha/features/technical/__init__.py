"""
Technical Feature Engineering Subsystem
=======================================

Exposes mathematical derivations for price, volume, and momentum-based 
alpha factors, integrating seamlessly with the FactorRegistry discovery process.

Purpose
-------
Aggregates and standardizes the import path for all technical indicators, 
ensuring that the Singleton registry automatically detects and instantiates 
these modules during the feature generation pipeline.
"""

from .momentum import (
    Return5D, Return10D, Return21D, Return63D, Return126D, Return252D,
    MomentumAcceleration10D, MomentumAcceleration21D,
    RSI14D, RSI21,
    MACD, MACDSignal,
    StochasticOscillator,
    WilliamsR,
    TSI,
    RateOfChange20D,
    ADX14
)

from .volatility import (
    Volatility21D, Volatility63D,
    GKVolatility21D, GKVolatility63D,
    ATR14, ATR21,
    VolatilityRatio,
    Skewness21D,
    Kurtosis21D
)

from .mean_reversion import (
    DistSMA21D, DistSMA50D, DistSMA200D,
    ZScore21D, ZScore63D,
    MeanRevBBPosition, MeanRevBBWidth,
    MACrossover5_21, MACrossover21_63,
    PriceToHighLow52W,
    CCI
)

from .volume import (
    VolumeZScore21D,
    VolumeMA20Ratio,
    TurnoverRate,
    Amihud63D,
    PriceVolumeCorr21D,
    ForceIndex14D,
    ChaikinMoneyFlow21D,
    MoneyFlowIndex14,
    AccumulationDistribution
)

__all__ = [
    'Return5D', 'Return10D', 'Return21D', 'Return63D', 'Return126D', 'Return252D',
    'MomentumAcceleration10D', 'MomentumAcceleration21D',
    'RSI14D', 'RSI21',
    'MACD', 'MACDSignal',
    'StochasticOscillator',
    'WilliamsR',
    'TSI',
    'RateOfChange20D',
    'ADX14',
    
    'Volatility21D', 'Volatility63D',
    'GKVolatility21D', 'GKVolatility63D',
    'ATR14', 'ATR21',
    'VolatilityRatio',
    'Skewness21D',
    'Kurtosis21D',
    
    'DistSMA21D', 'DistSMA50D', 'DistSMA200D',
    'ZScore21D', 'ZScore63D',
    'MeanRevBBPosition', 'MeanRevBBWidth',
    'MACrossover5_21', 'MACrossover21_63',
    'PriceToHighLow52W',
    'CCI',
    
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