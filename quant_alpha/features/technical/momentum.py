"""
Momentum Factors
================
Quantitative signals capturing price velocity, acceleration, and oscillation dynamics.

Purpose
-------
This module constructs alpha factors based on the **Momentum** anomaly, which posits
that assets which have performed well in the past will continue to perform well
in the near future (Jegadeesh & Titman, 1993). It includes:
1. **Time-Series Momentum**: Raw returns over various lookback horizons.
2. **Acceleration**: The second derivative of price (convexity).
3. **Oscillators**: Mean-reverting momentum indicators (RSI, Stochastic, TSI)
   identifying overbought/oversold conditions.
4. **Trend Strength**: Direction-agnostic metrics like ADX.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    registry = FactorRegistry()
    mom_factor = registry.get('rsi_14d')
    signals = mom_factor.compute(market_data_df)

Importance
----------
- **Alpha Generation**: Momentum is one of the most robust and pervasive style
  factors in asset pricing, offering significant risk-adjusted returns.
- **Signal Diversity**: Combinations of fast (5D) and slow (252D) momentum
  signals allow for multi-frequency strategy construction.
- **Regime Identification**: Indicators like ADX help distinguish trending regimes
  from mean-reverting chopping markets.

Tools & Frameworks
------------------
- **Pandas**: Efficient `groupby` and `ewm` operations for time-series smoothing.
- **NumPy**: Vectorized arithmetic for oscillator normalization.
- **FactorRegistry**: Decorator-based registration for pipeline integration.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor
from quant_alpha.utils.column_helpers import safe_col

# Machine epsilon for numerical stability (prevents DivisionByZero)
EPS = 1e-9

# ==================== 1. SIMPLE RETURNS (Rate of Change) ====================
# Metric: Discrete returns over $N$ periods.
# Formula: $$ R_t = \frac{P_t}{P_{t-n}} - 1 $$

@FactorRegistry.register()
class Return5D(TechnicalFactor):
    def __init__(self, period = 5):
        super().__init__(name='return_5d',description='5-day return',lookback_period=period + 1)
        self.period = period
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Optimization: pct_change on groupby object preserves index alignment
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
    r"""
    10-Day Momentum Acceleration.
    
    Measures the rate of change of momentum (Convexity/Second Derivative).
    Formula:
    $$ A_t = R_{t} - R_{t-n} $$
    """
    def __init__(self, period=10):
        super().__init__(name='mom_accel_10d', description='10-day momentum acceleration', lookback_period=period * 2)
        self.period = period
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        # 1. First Derivative: Calculate Returns per ticker
        ret = df.groupby('ticker')['close'].pct_change(self.period) 

        # 2. Second Derivative: Calculate Diff of Returns per ticker
        # Critical: Must group by ticker again on the Series to prevent cross-asset data leakage.
        return ret.groupby(df['ticker']).diff(self.period)
    
@FactorRegistry.register()
class MomentumAcceleration21D(TechnicalFactor):
    """21-Day Momentum Acceleration."""
    def __init__(self, period=21):
        super().__init__(name='mom_accel_21d', description='21-day momentum acceleration', lookback_period=period * 2)
        self.period = period
        
    def compute(self, df:pd.DataFrame)->pd.Series:
        ret = df.groupby('ticker')['close'].pct_change(self.period)
        return ret.groupby(df['ticker']).diff(self.period)

# ==================== 3. RSI & MACD ====================

@FactorRegistry.register()
class RSI14D(TechnicalFactor):
    r"""
    Relative Strength Index (RSI) - 14 Day.
    
    Momentum oscillator measuring the speed and change of price movements.
    Formula:
    $$ RSI = 100 - \frac{100}{1 + RS} $$
    """
    def __init__(self, period=14):
        super().__init__(name='rsi_14d', description=f'RSI {period}', lookback_period=period * 3)
        self.period = period
    
    def compute(self, df:pd.DataFrame)->pd.Series:
        def calculate_rsi_transform(x):
            delta = x.diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            # Wilder's Smoothing (EWMA with com = period - 1)
            avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
            avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

            # Numerical Stability: Add Epsilon to prevent division by zero
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        # Vectorization: Use transform to maintain strict index alignment with input DF
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
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))
        
        return df.groupby('ticker')['close'].transform(calculate_rsi_transform)

@FactorRegistry.register()
class MACD(TechnicalFactor):
    r"""
    Moving Average Convergence Divergence (MACD) - Main Line.
    
    Trend-following momentum indicator.
    Formula:
    $$ MACD = EMA_{fast} - EMA_{slow} $$
    """
    def __init__(self, fast=12, slow=26):
        super().__init__(name=f'macd_{fast}_{slow}',description='MACD Main Line',lookback_period=slow + 10)
        self.fast = fast
        self.slow = slow
    
    def compute(self, df:pd.DataFrame) -> pd.Series :
        # Optimization: Use transform to keep series aligned with original DF
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        return ema_fast - ema_slow
    
@FactorRegistry.register()
class MACDSignal(TechnicalFactor):
    r"""
    MACD Signal Line.
    
    The EMA of the MACD Line, acting as a trigger for buy/sell signals.
    Formula:
    $$ Signal = EMA_{signal\_period}(MACD) $$
    """
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(name=f'macd_signal_{fast}_{slow}_{signal}',description='MACD Signal Line',lookback_period=slow +signal+ 10)
        self.fast = fast
        self.slow = slow
        self.signal = signal    
    
    def compute(self, df:pd.DataFrame) -> pd.Series :
        # 1. Calculate MACD Series
        ema_fast = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.fast, adjust=False).mean())
        ema_slow = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=self.slow, adjust=False).mean())
        macd = ema_fast - ema_slow

        # 2. Calculate Signal Line
        # Note: Group the *calculated MACD series* by the original ticker column to avoid .loc indexing risks
        signal = macd.groupby(df['ticker']).transform(lambda x: x.ewm(span=self.signal, adjust=False).mean())
        return signal

# ==================== StochasticOscillator ====================

@FactorRegistry.register()
class StochasticOscillator(TechnicalFactor):
    r"""
    Stochastic Oscillator %K.
    
    Compares a particular closing price to a range of its prices over a certain period.
    
    Formula:
    $$ \%K = \frac{C - L_n}{H_n - L_n} \times 100 $$
    """
    def __init__(self, period=14):
        super().__init__(name='stoch_k', description='Stochastic Oscillator %K', lookback_period=period)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc(g):
            lo = safe_col(g, "low")
            hi = safe_col(g, "high")
            if lo.isna().all() or hi.isna().all():
                return pd.Series(np.nan, index=g.index)

            low_min = lo.rolling(window=self.period).min()
            high_max = hi.rolling(window=self.period).max()
            
            denom = (high_max - low_min).replace(0, np.nan)
            k = 100 * ((g['close'] - low_min) / denom)
            return k

        k_series = df.groupby('ticker', group_keys=False).apply(calc, include_groups=False)
        return k_series.fillna(50) # Neutral fill
    
# ==================== WilliamsR ====================

@FactorRegistry.register()
class WilliamsR(TechnicalFactor):
    r"""
    Williams %R.
    
    Momentum indicator that is the inverse of the Fast Stochastic Oscillator.
    Formula:
    $$ \%R = \frac{H_n - C}{H_n - L_n} \times -100 $$
    """
    def __init__(self, period=14):
        super().__init__(name='williams_r', description='Williams %R', lookback_period=period)      
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc(g):
            lo = safe_col(g, "low")
            hi = safe_col(g, "high")
            if lo.isna().all() or hi.isna().all():
                return pd.Series(np.nan, index=g.index)

            low_min = lo.rolling(window=self.period).min()
            high_max = hi.rolling(window=self.period).max()

            denom = (high_max - low_min).replace(0, np.nan)
            wr = -100 * ((high_max - g['close']) / denom)
            return wr

        wr_series = df.groupby('ticker', group_keys=False).apply(calc, include_groups=False)
        return wr_series.fillna(-50) # Neutral fill
    
# =========================== TSI ============================

@FactorRegistry.register()
class TSI(TechnicalFactor):
    r"""
    True Strength Index (TSI).
    
    A variation of the double smoothed momentum indicator.
    Formula:
    $$ TSI = 100 \times \frac{EMA(EMA(\Delta P))}{EMA(EMA(|\Delta P|))} $$
    """
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


# ==================== 4. RATE OF CHANGE (ROC) ====================

@FactorRegistry.register()
class RateOfChange20D(TechnicalFactor):
    r"""
    Rate of Change (ROC) - 20 Day.
    
    Momentum oscillator measuring the percentage change in price.
    Formula:
    $$ ROC = \frac{Price_t - Price_{t-n}}{Price_{t-n}} $$
    """
    def __init__(self, period=20):
        super().__init__(name='roc_20d', description='Rate of Change 20D', lookback_period=period + 1)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('ticker')['close'].pct_change(self.period)


# ==================== 5. AVERAGE DIRECTIONAL INDEX (ADX) ====================

@FactorRegistry.register()
class ADX14(TechnicalFactor):
    """
    Average Directional Index (ADX) - 14 Day.
    
    Measures trend strength regardless of direction.
    Range: 0-100 (higher = stronger trend)
    """
    def __init__(self, period=14):
        super().__init__(name='adx_14', description='Average Directional Index 14', lookback_period=period * 2)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_adx_group(group):
            # Pre-computation: Isolate series for cleaner syntax
            high = safe_col(group, "high")
            low = safe_col(group, "low")
            close = group['close']
            
            if high.isna().all():
                return pd.Series(np.nan, index=group.index)

            # True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up = high.diff()
            down = -low.diff()
            
            plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
            minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)
            
            # Smoothed values (SMA)
            tr_sum = tr.rolling(self.period).sum()
            plus_sum = plus_dm.rolling(self.period).sum()
            minus_sum = minus_dm.rolling(self.period).sum()
            
            # DI values
            plus_di = 100 * plus_sum / (tr_sum + EPS)
            minus_di = 100 * minus_sum / (tr_sum + EPS)
            
            # DX
            di_diff = (plus_di - minus_di).abs()
            di_total = plus_di + minus_di
            dx = 100 * di_diff / (di_total + EPS)
            
            # ADX Calculation: EMA of DX
            adx = dx.ewm(span=self.period, adjust=False).mean()
            
            return adx
        
        # FutureWarning fix: Add include_groups=False to avoid including grouping columns
        return df.groupby('ticker', group_keys=False).apply(calc_adx_group, include_groups=False)



     