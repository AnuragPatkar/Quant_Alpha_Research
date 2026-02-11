"""
Volatility Factors (Production Hardened)
Capture market risk, variance, tail risk, and intraday volatility.
Total Factors: 12

Fixes:
- VolatilityRatio: Replaced reset_index with transform() for safe alignment.
- GK Volatility: Uses correct High-Low-Close-Open formula.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor

EPS = 1e-9

# ==================== 1. STANDARD HISTORICAL VOLATILITY ====================
# Logic: Rolling Standard Deviation of Returns * Sqrt(252)

@FactorRegistry.register()
class Volatility5D(TechnicalFactor):
    def __init__(self, period=5):
        super().__init__(name='volatility_5d', description='1 Week volatility', lookback_period=period + 1)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252))

@FactorRegistry.register()
class Volatility10D(TechnicalFactor):
    def __init__(self, period=10):
        super().__init__(name='volatility_10d',description='2 Week volatility',lookback_period=period + 1)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:       
        return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252))

@FactorRegistry.register()
class Volatility21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='volatility_21d', description='1 Month Volatility', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
        )
    
@FactorRegistry.register()
class Volatility63D(TechnicalFactor):
    def __init__(self, period=63):
        super().__init__(name='volatility_63d', description='3 Month Volatility', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
        )

@FactorRegistry.register()
class Volatility126D(TechnicalFactor):
    def __init__(self, period=126):
        super().__init__(name='volatility_126d', description='6 Month Volatility', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(
            lambda x: x.pct_change().rolling(window=self.period).std() * np.sqrt(252)
        )
    
# ==================== 2. GARMAN-KLASS VOLATILITY ====================
# More efficient estimator using Open, High, Low, Close
@FactorRegistry.register()
class GKVolatility21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='gk_vol_21',description='Garman-Klass Volatility',lookback_period=period )
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series :
        # Pre-calculate logs safely (Vectorized)
        log_hl = np.log((df['high'] / df['low']).replace(0,np.nan))
        log_co = np.log((df['close'] / df['open']).replace(0,np.nan))
        
        # Raw Varience per day
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

        # Rolling Mean of Variance -> Sqrt -> Annualize
        # Using transform to ensure idex alignment
        return gk_var.groupby(df['ticker']).transform(lambda x: np.sqrt(x.rolling(window=self.period).mean()) * np.sqrt(252))
    
@FactorRegistry.register()
class GKVolatility63D(TechnicalFactor):
    def __init__(self, period = 63):
        super().__init__(name='gk_val_63',description='Garman-Klass Volatility (63D)',lookback_period=period )
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series :
        log_hl = np.log((df['high'] / df['low']).replace(0, np.nan))
        log_co = np.log((df['close'] / df['open']).replace(0, np.nan))
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        return gk_var.groupby(df['ticker']).transform(
            lambda x: np.sqrt(x.rolling(window=self.period).mean()) * np.sqrt(252)
        )
    
# ==================== 3. ATR (AVERAGE TRUE RANGE) ====================
@FactorRegistry.register()
class ATR14(TechnicalFactor):
    def __init__(self, period=14):
        super().__init__(name='atr_14', description='ATR 14 Normalized', lookback_period=period +1 )
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df.groupby('ticker')['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
       
        # Using transform for rolling calculation
        atr = tr.groupby(df['ticker']).transform(lambda x: x.rolling(window=self.period).mean())
        return atr / (df['close'] + EPS)
    
@FactorRegistry.register()
class ATR21(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='atr_21', description='ATR 21 Normalized', lookback_period=period + 1)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        prev_close = df.groupby('ticker')['close'].shift(1)
        
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.groupby(df['ticker']).transform(lambda x: x.rolling(window=self.period).mean())
        return atr / (df['close'] + EPS)
    
# ==================== 4. REGIME INDICATORS (RATIO, SKEW, KURT) ====================
@FactorRegistry.register()
class VolatilityRatio(TechnicalFactor):
    """
    Ratio of Short Term Vol (5D) to Long Term Vol (21D).
    Safe Implementation using .transform()
    """
    def __init__(self, short_period=5, long_period=21):
        super().__init__(name='vol_ratio_5_21', description='Vol Ratio 5D/21D', lookback_period=long_period + 1)
        self.short = short_period
        self.long = long_period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        # Use transform instead of rolling().std() directly to maintain index alignment
        vol_short = df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.short).std())
        vol_long = df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.long).std())
        return vol_short / (vol_long + EPS)
    
@FactorRegistry.register()
class Skewness21D(TechnicalFactor):
    """
    Rolling Skewness (21D)
    Measures asymmetry of return distribution (Crash Risk).
    """
    def __init__(self, period=21):
        super().__init__(name='skew_21d', description='Rolling Skewness 21D', lookback_period=period + 1)
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(lambda x: x.pct_change().rolling(window=self.period).skew()).fillna(0)

@FactorRegistry.register()
class Kurtosis21D(TechnicalFactor):
    """
    Rolling Kurtosis (21D)
    Measures 'Fat Tails' (Extreme Event Risk).
    """
    def __init__(self, period=21):
        super().__init__(name='kurt_21d', description='Rolling Kurtosis 21D', lookback_period=period + 1)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].transform(lambda x:x.pct_change().rolling(window=self.period).kurt()).fillna(0)

    
    
    
