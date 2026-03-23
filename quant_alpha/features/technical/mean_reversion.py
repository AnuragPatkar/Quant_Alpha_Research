"""
Mean Reversion Factors
======================
Quantitative signals identifying "Overbought" and "Oversold" conditions relative to statistical central tendencies.

Purpose
-------
This module constructs alpha factors based on the **Mean Reversion** hypothesis, which posits that
asset prices and historical returns eventually revert to their long-term mean (Stationarity).
These factors quantify the normalized deviation of price from various statistical anchors
(Moving Averages, Bollinger Bands, Range Extremes).

Theoretical underpinning: Assumes price processes exhibit Ornstein-Uhlenbeck properties over specific horizons.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    registry = FactorRegistry()
    mr_factor = registry.get('dist_sma_50d')
    signals = mr_factor.compute(market_data_df)

Importance
----------
- **Counter-Trend Alpha**: Identifies extension exhaustion points where the probability favors
  a reversal (e.g., $Z > 2.0$ implies a 95% confidence interval breach).
- **Regime Detection**: Volatility metrics like Band Width signal transition
  phases between range-bound (mean reverting) and trending regimes.
- **Statistical Robustness**: Uses Z-scoring and normalized distances to ensure
  comparability across assets with different price magnitudes.

Tools & Frameworks
------------------
- **Pandas**: Efficient `rolling` window operations for statistical aggregation.
- **NumPy**: Vectorized arithmetic for normalization and spread calculation.
- **FactorRegistry**: Decorator-based registration for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor
from quant_alpha.utils.column_helpers import safe_col

EPS = 1e-9  # Machine epsilon for numerical stability

# ==================== 1. DISTANCE FROM MOVING AVERAGES ====================
# Metric: Percentage deviation from the trend line.
# Formula: $$ D_t = \frac{P_t - MA_t}{MA_t} $$

@FactorRegistry.register()
class DistSMA21D(TechnicalFactor):
    """
    Distance from 21-Day Simple Moving Average.
    
    Captures short-term mean reversion potential via percentage deviation.
    Formula: $$ D_t = \frac{P_t - \text{SMA}_{21}}{\text{SMA}_{21}} $$
    """
    def __init__(self, period=21):
        super().__init__(name='dist_sma_21d', description='Distance from 21D SMA', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)

@FactorRegistry.register()
class DistSMA50D(TechnicalFactor):
    """
    Distance from 50-Day Simple Moving Average.
    
    Intermediate-term trend deviation.
    Formula: $$ D_t = \frac{P_t - \text{SMA}_{50}}{\text{SMA}_{50}} $$
    """
    def __init__(self, period=50):
        super().__init__(name='dist_sma_50d', description='Distance from 50D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)
    
@FactorRegistry.register()
class DistSMA200D(TechnicalFactor):
    """
    Distance from 200-Day Simple Moving Average.
    
    Institutional Benchmark: Deviations from the 200DMA often signal long-term
    valuation extremes. Prices far below are considered 'Deep Value' (Oversold),
    while prices far above indicate 'Euphoria' (Overbought).
    
    Formula: $$ D_t = \frac{P_t - \text{SMA}_{200}}{\text{SMA}_{200}} $$
    """
    def __init__(self, period=200):
        super().__init__(name='dist_sma_200d', description='Distance from 200D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)
    
# ================================ 2. Price Z-Scores ==============================
# Metric: Standardized statistical deviation.
# Formula: $$ Z_t = \frac{P_t - \mu_t}{\sigma_t} $$

@FactorRegistry.register()
class ZScore21D(TechnicalFactor):
    """
    Price Z-Score (21-Day).
    
    Measures how many standard deviations the current price is from its 21-day mean.
    Assumption: Short-term returns are normally distributed.
    """
    def __init__(self, period=21):
        super().__init__(name='zscore_21d', description='Price Z-Score 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_z(x):
            # Z = (Value - Mean) / StdDev
            return (x - x.rolling(self.period).mean()) / (x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)
    
@FactorRegistry.register()
class ZScore63D(TechnicalFactor):
    """
    Price Z-Score (63-Day/Quarterly).
    
    Measures quarterly mean reversion tendencies.
    """
    def __init__(self, period=63):
        super().__init__(name='zscore_63d', description='Price Z-Score 63D', lookback_period=period + 5)
        self.period = period
 
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return (x - x.rolling(self.period).mean()) / (x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)      

# ==================== 3. BOLLINGER BANDS (Mean Reversion Specific) ====================

@FactorRegistry.register()
class MeanRevBBPosition(TechnicalFactor):
    """
    Bollinger Band %B (Position).
    
    Quantifies the price position relative to the bands.
    - $\%B > 1.0$: Price above upper band (Overbought).
    - $\%B < 0.0$: Price below lower band (Oversold).
    
    Formula:
    $$ \%B = \frac{Price - Lower}{Upper - Lower} $$
    """
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
            
            # Scaling: Position within the bandwidth
            return (x - lower) / ((upper - lower) + EPS)
        return df.groupby('ticker')['close'].transform(calc_bb)

@FactorRegistry.register()
class MeanRevBBWidth(TechnicalFactor):
    """
    Bollinger Band Width.
    
    Measures volatility expansion/contraction (The Squeeze).
    
    Formula:
    $$ Width = \frac{Upper - Lower}{Middle} $$
    """
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
            
            # Normalized width relative to price level
            return (upper - lower) / (mean + EPS)
        return df.groupby('ticker')['close'].transform(calc_width)

# ==================== 4. MA CROSSOVERS ====================

@FactorRegistry.register()
class MACrossover5_21(TechnicalFactor):
    """
    Moving Average Spread (5D vs 21D).
    
    Calculates the normalized percentage distance between Fast and Slow MAs.
    Extreme spreads often precede mean reversion.
    
    Formula:
    $$ Spread = \frac{MA_{fast} - MA_{slow}}{MA_{slow}} $$
    """
    def __init__(self, fast=5, slow=21):
        super().__init__(name='ma_cross_5_21', description='MA 5-21 Spread', lookback_period=slow + 5)
        self.fast = fast
        self.slow = slow

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_fast = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.fast).mean())        
        ma_slow = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.slow).mean())
        return (ma_fast - ma_slow) / (ma_slow + EPS)
    
@FactorRegistry.register()
class MACrossover21_63(TechnicalFactor):
    """
    Moving Average Spread (21D vs 63D).
    
    Intermediate-term divergence signal.
    Formula: $$ Spread = \frac{MA_{21} - MA_{63}}{MA_{63}} $$
    """
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
    Stochastic Position within 52-Week Range.
    
    Determines where the current price sits relative to its annual range.
    
    Formula:
    $$ K = \frac{Close - Low_{52w}}{High_{52w} - Low_{52w}} $$
    
    Range:
    - 1.0: Trading at 52-Week High.
    - 0.0: Trading at 52-Week Low.
    """
    def __init__(self, period=252):
        super().__init__(name='price_pos_52w', description='Position in 52W Range', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_pos(group):
            high = safe_col(group, "high")
            low = safe_col(group, "low")
            if high.isna().all():
                high = group['close']
                low = group['close']

            roll_high = high.rolling(self.period).max()
            roll_low = low.rolling(self.period).min()
            denom = roll_high - roll_low
            position = (group['close'] - roll_low) / (denom + EPS)
            return position

        pos_series = df.groupby('ticker', group_keys=False).apply(calc_pos, include_groups=False)
        return pos_series.fillna(0)
    
# ==================== 6. CCI (Commodity Channel Index) ====================    

@FactorRegistry.register()
class CCI(TechnicalFactor):
    """
    Commodity Channel Index (20D).
    
    Measures the deviation of the Typical Price from its SMA, normalized by
    Mean Absolute Deviation (MAD).
    
    Formula:
    $$ CCI = \frac{TP - SMA_{TP}}{0.015 \times \text{MeanDev}} $$
    """
    def __init__(self, period=20):
        super().__init__(name='cci_20', description='Commodity Channel Index', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        def calc_cci(x):
            high = safe_col(x, "high")
            low = safe_col(x, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=x.index)

            # 1. Typical Price
            tp = (high + low + x['close']) / 3
            
            # 2. SMA of Typical Price
            sma_tp = tp.rolling(self.period).mean()

            # 3. Mean Deviation (MAD)
            # MAD = Mean(|Price - SMA|)
            # Note: apply() is O(N * W) which is slower than vectorized operations, but required for MAD.
            def mad(a):
                return np.mean(np.abs(a - np.mean(a)))
            
            mean_dev = tp.rolling(window=self.period).apply(mad, raw=True)

            # 4. CCI Calculation (0.015 is Lambert's Constant)
            cci = (tp - sma_tp) / (mean_dev * 0.015 + EPS)
            return cci
        # FutureWarning fix: Add include_groups=False to avoid including grouping columns
        return df.groupby('ticker', group_keys=False).apply(calc_cci, include_groups=False)
    



  


    


                  