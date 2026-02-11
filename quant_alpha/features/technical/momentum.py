"""
Momentum Factors (Production Hardened)
Capture price momentum, acceleration, and oscillation effects.
Total Factors: 15

Fixes:
- Zero Division Handling (Added epsilon 1e-9)
- Data Leakage Prevention (Strict Grouping on Shifts)
- Index Alignment (Replaced .apply with .transform where possible)
- NaN Handling (Neutral fills for flat markets)
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor

# EPSILON to prevent Divison by Zero
EPS = 1e-9

# ==================== 1. SIMPLE RETURNS (Rate of Change) ====================
# Logic: pct_change automatically handles grouping alignment if called on groupby object
@FactorRegistry.register()
class Return5D(TechnicalFactor):
    def __init__(self, period = 5):
        super().__init__(name='return_5d',description='5-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return10D(TechnicalFactor):
    def __init__(self,period = 10):
        super().__init__(name='return_10d',description='10-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return21D(TechnicalFactor):
    def __init__(self,period = 21):
        super().__init__(name='return_21d',description='21-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame ) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)
    
@FactorRegistry.register()
class Return63D(TechnicalFactor):
    def __init__(self,period = 63):
        super().__init__(name='return_63d',description='63-day return',lookback_period=period + 1)
        self.period = period    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return126D(TechnicalFactor):
    def __init__(self, period=126):
        super().__init__(name='return_126d', description='126-day return', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)

@FactorRegistry.register()
class Return252D(TechnicalFactor):
    def __init__(self, period=252):
        super().__init__(name='return_252d', description='252-day return', lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)
   
# ==================== 2. MOMENTUM ACCELERATION ====================
@FactorRegistry.register()
class MomentumAcceleration10D(TechnicalFactor):
    def __init__(self, period=10):
        super().__init__(name='mom_accel_10d', description='10-day momentum acceleration', lookback_period=period * 2)
        self.period = period
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        # 1. Calculate Returns per ticker
        ret = df.groupby('ticker')['close'].pct_change(self.period) 

        # 2. Calculate Diff of Returns per ticker (Critical: Group again to prevent leakage)
        # We use the original df['ticker'] to group the calculated series
        return ret.groupby(df['ticker']).diff(self.period)
    
@FactorRegistry.register()
class MomentumAcceleration21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='mom_accel_21d', description='21-day momentum acceleration', lookback_period=period * 2)
        self.period = period
        
    def compute(self, df:pd.DataFrame)->pd.Series:
        ret = df.groupby('ticker')['close'].pct_change(self.period)
        return ret.groupby(df['ticker']).diff(self.period)

# ==================== 3. RSI & MACD ====================
@FactorRegistry.register()
class RSI14D(TechnicalFactor):
    def __init__(self, period=14):
        super().__init__(name='rsi_14d', description=f'RSI {period}', lookback_period=period * 3)
        self.period = period
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        def calculate_rsi_transform(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
            avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

            # Add Epsilon to prevent divison by zero
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        # Use transform to maintain index alignment
        return df.groupby('ticker')['close'].transform(calculate_rsi_transform)

@FactorRegistry.register()
class RSI21(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='rsi_21',description=f'RSI {period}',lookback_period=period * 3)
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calculate_rsi_transform(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
            avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

            # Add Epsilon to prevent divison by zero
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        # Use transform to maintain index alignment
        return df.groupby('ticker')['close'].transform(calculate_rsi_transform)

@FactorRegistry.register()
class MACD(TechnicalFactor):
    def __init__(self, fast=12, slow=26):
        super().__init__(name=f'macd_{fast}_{slow}',description='MACD Main Line',lookback_period=slow + 10)
        self.fast = fast
        self.slow = slow
    
    def compute(self, df:pd.DataFrame) -> pd.Series :
        # Use transform to keep series aligned with original DF
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        return ema_fast - ema_slow
    
@FactorRegistry.register()
class MACDSignal(TechnicalFactor):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(name=f'macd_signal_{fast}_{slow}_{signal}',description='MACD Signal Line',lookback_period=slow +signal+ 10)
        self.fast = fast
        self.slow = slow
        self.signal = signal    
    
    def compute(self, df:pd.DataFrame) -> pd.Series :
        # 1. Calc MACD Series
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        macd = ema_fast - ema_slow

        # 2. Calc Signal
        # Important: We group the *calculated MACD series* by the original ticker column
        # This avoids .loc indexing risks
        signal = macd.groupby(df['ticker']).transform(lambda x: x.ewm(span=self.signal, adjust=False).mean())
        return signal

# ==================== StochasticOscillator ====================
@FactorRegistry.register()
class StochasticOscillator(TechnicalFactor):
    """Stochastic %K"""
    def __init__(self, period=14):
        super().__init__(name='stoch_k', description='Stochastic Oscillator %K', lookback_period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        low_min = df.groupby('ticker')['low'].transform(lambda x: x.rolling(window=self.period).min())
        high_max = df.groupby('ticker')['high'].transform(lambda x: x.rolling(window=self.period).max())
        
        denom = (high_max - low_min).replace(0, np.nan)
        k = 100 * ((df['close'] - low_min) / denom)
        
        return k.fillna(50)
    
# ==================== WilliamsR ====================
@FactorRegistry.register()
class WilliamsR(TechnicalFactor):
    """Williams %R"""
    def __init__(self, period=14):
        super().__init__(name='williams_r', description='Williams %R', lookback_period=period)      
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Vectorized Approach via Transform (Safe)
        low_min = df.groupby('ticker')['low'].transform(lambda x: x.rolling(window=self.period).min())
        high_max = df.groupby('ticker')['high'].transform(lambda x: x.rolling(window=self.period).max())

        denom = (high_max - low_min).replace(0, np.nan)
        wr = -100 * ((high_max - df['close']) / denom)
        
        return wr.fillna(-50) # Neutral fill for Williams %R
    
# =========================== TSI ============================
@FactorRegistry.register()
class TSI(TechnicalFactor):
    """True Strength Index (TSI)"""
    def __init__(self, long_period=25,short_period=13):
        super().__init__(name='tsi', description='True Strength Index', lookback_period= long_period + short_period )
        self.long_period = long_period
        self.short_period = short_period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_tsi_transform(x):
            diff = x.diff()

            smooth1 = diff.ewm(span=self.long_period, adjust=False).mean()
            smooth2 = smooth1.ewm(span=self.short_period, adjust=False).mean()

            abs_diff = diff.abs()
            abs_smooth1 = abs_diff.ewm(span=self.long_period, adjust=False).mean()
            abs_smooth2 = abs_smooth1.ewm(span=self.short_period, adjust=False).mean()

            denom = abs_smooth2.replace(0,np.nan)
            return 100 * (smooth2 / denom)
        
        return df.groupby('ticker')['close'].transform(calc_tsi_transform).fillna(0)
        

     