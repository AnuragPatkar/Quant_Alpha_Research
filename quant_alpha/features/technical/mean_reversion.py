"""
Mean Reversion Factors (Production Hardened)
Capture "Overbought" and "Oversold" conditions relative to trends and historical ranges.
Total Factors: 13

Includes:
1. Distance from MA (10, 21, 50, 200)
2. Price Z-Scores (10, 21, 63)
3. Bollinger Band Position & Width (Mean Reversion specific)
4. MA Crossovers (Spread)
5. 52-Week High/Low Position
6. CCI (Commodity Channel Index) 
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor

EPS = 1e-9

# ==================== 1. DISTANCE FROM MOVING AVERAGES ====================
# Formula: (Close - MA) / MA

# REMOVED: DistSMA10D (too short-term, redundant with DistSMA21D)
# @FactorRegistry.register()
# class DistSMA10D(TechnicalFactor):
#     def __init__(self, period=10):
#         super().__init__(name='dist_sma_10d', description='Distance from 10D SMA', lookback_period=period + 5)
#         self.period = period
#     
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
#         return (df['close'] - ma) /( ma + EPS )
    
@FactorRegistry.register()
class DistSMA21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='dist_sma_21d', description='Distance from 21D SMA', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x:x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)

@FactorRegistry.register()
class DistSMA50D(TechnicalFactor):
    def __init__(self, period=50):
        super().__init__(name='dist_sma_50d', description='Distance from 50D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x:x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)
    
@FactorRegistry.register()
class DistSMA200D(TechnicalFactor):
    """
    Distance from 200D SMA.
    Institutional Benchmark: Prices far below 200DMA are often 'Deep Value'.
    """
    def __init__(self, period=200):
        super().__init__(name='dist_sma_200d', description='Distance from 200D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x:x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)
    
# ================================ 2. Price Z-Scores ==============================
# Formula : (Close - Mean) / Stdev

# REMOVED: ZScore10D (too short-term mean reversion, high noise-to-signal)
# @FactorRegistry.register()
# class ZScore10D(TechnicalFactor):
#     def __init__(self, period=10):
#         super().__init__(name='zscore_10d', description='Price Z-Score 10D', lookback_period=period + 5)
#         self.period = period
# 
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         def calc_z(x):
#             return (x - x.rolling(self.period).mean()) / (x.rolling(self.period).std() + EPS)
#         return df.groupby('ticker')['close'].transform(calc_z)

@FactorRegistry.register()
class ZScore21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='zscore_21d', description='Price Z-Score 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return(x-x.rolling(self.period).mean())/(x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)
    
@FactorRegistry.register()
class ZScore63D(TechnicalFactor):
    def __init__(self, period=63):
        super().__init__(name='zscore_63d', description='Price Z-Score 63D', lookback_period=period + 5)
        self.period = period
 
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return(x-x.rolling(self.period).mean())/(x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)      

# ==================== 3. BOLLINGER BANDS (Mean Reversion Specific) ====================
# Prefixed 'mr_' to avoid conflict with Volatility module

@FactorRegistry.register()
class MeanRevBBPosition(TechnicalFactor):
    """Bollinger Band %B"""
    def __init__(self, period=20,std=2):
        super().__init__(name='mr_bb_pos', description='BB Position %B', lookback_period=period + 5)
        self.period = period
        self.std = std
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        def calc_bb(x):
            mean = x.rolling(self.period).mean()
            std = x.rolling(self.period).std()
            upper = mean + (self.std * std)
            lower = mean - (self.std * std)
            return (x-lower)/((upper-lower) + EPS )
        return df.groupby('ticker')['close'].transform(calc_bb)

@FactorRegistry.register()
class MeanRevBBWidth(TechnicalFactor):
    """Bollinger Band Width"""
    def __init__(self, period=20, std=2):
        super().__init__(name='mr_bb_width', description='BB Width', lookback_period=period + 5)
        self.period = period
        self.std = std
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        def calc_width(x):
            mean = x.rolling(self.period).mean()
            std = x.rolling(self.period).std()
            upper = mean + (self.std * std)
            lower = mean - (self.std * std)
            return (upper - lower) / (mean + EPS)
        return df.groupby('ticker')['close'].transform(calc_width)

# ==================== 4. MA CROSSOVERS ====================
@FactorRegistry.register()
class MACrossover5_21(TechnicalFactor):
    """Spread between 5D and 21D SMA """
    def __init__(self, fast=5, slow=21):
        super().__init__(name='ma_cross_5_21', description='MA 5-21 Spread', lookback_period= slow + 5)
        self.fast = fast
        self.slow = slow

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_fast = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.fast).mean())        
        ma_slow = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.slow).mean())
        return (ma_fast - ma_slow) / (ma_slow + EPS)
    
@FactorRegistry.register()
class MACrossover21_63(TechnicalFactor):
    """Spread between 21D and 63D SMA"""
    def __init__(self, fast=21, slow=63):
        super().__init__(name='ma_cross_21_63', description='MA 21-63 Spread', lookback_period=slow + 5)
        self.fast = fast
        self.slow = slow
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_fast = df.groupby('ticker')['close'].transform(lambda x: x.rolling(self.fast).mean())
        ma_slow = df.groupby('ticker')['close'].transform(lambda x: x.rolling(self.slow).mean())
        return (ma_fast - ma_slow) / (ma_slow + EPS)

# ==================== 5. PRICE POSITION IN 52-WEEK RANGE ====================
@FactorRegistry.register()
class PriceToHighLow52W(TechnicalFactor):
    """
    Stochastic position within 52-Week Range.
    1.0 = At 52W High, 0.0 = At 52W Low
    """
    def __init__(self, period=252):
        super().__init__(name='price_pos_52w',description='Position in 52W Range',lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        roll_high = df.groupby('ticker')['high'].transform(lambda x: x.rolling(self.period).max())
        roll_low = df.groupby('ticker')['low'].transform(lambda x: x.rolling(self.period).min())
        denom = roll_high - roll_low
        position = (df['close'] - roll_low) / (denom + EPS)
        
        return position.fillna(0)
    
# ==================== 6. CCI (Commodity Channel Index) ====================    
@FactorRegistry.register()
class CCI(TechnicalFactor):
    """
    Commodity Channel Index (20D).
    Safe implementation without risky index resets.
    """
    def __init__(self, period=20):
        super().__init__(name='cci_20', description='Commodity Channel Index', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_cci(x):
            # 1. Typical Price
            tp = (x['high'] + x['low'] + x['close']) / 3
            
            # 2. SMA of Typical Price
            sma_tp = tp.rolling(self.period).mean()

            # 3. Mean Deviation (MAD)
            # MAD =  Mean(|Price - SMA|)
            def mad(a):
                return np.mean(np.abs(a - np.mean(a)))
            
            mean_dev = tp.rolling(window = self.period).apply(mad,raw = True)

            # CCI calculation
            cci = (tp - sma_tp) / (mean_dev * 0.015 + EPS)
            return cci
        return df.groupby('ticker', group_keys=False).apply(calc_cci)
    



  


    


                  