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

EPS = 1e-9

@FactorRegistry.register()
class DistSMA21D(TechnicalFactor):
    """
    Distance from 21-Day Simple Moving Average.
    
    Captures short-term mean reversion potential via percentage deviation.
    Formula: $$ D_t = \frac{P_t - \text{SMA}_{21}}{\text{SMA}_{21}} $$
    """
    def __init__(self, period: int = 21):
        """
        Initializes the short-term deviation boundary matrix.
        
        Args:
            period (int): Lookback parameter bounding the moving average. Defaults to 21.
        """
        super().__init__(name='dist_sma_21d', description='Distance from 21D SMA', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates discrete spatial variations bounding historical prices relative to the structural trend.
        
        Args:
            df (pd.DataFrame): Systemic raw historical execution limits.
            
        Returns:
            pd.Series: Continuous distribution mapped identically across explicit execution targets.
        """
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)

@FactorRegistry.register()
class DistSMA50D(TechnicalFactor):
    """
    Distance from 50-Day Simple Moving Average.
    
    Intermediate-term trend deviation.
    Formula: $$ D_t = \frac{P_t - \text{SMA}_{50}}{\text{SMA}_{50}} $$
    """
    def __init__(self, period: int = 50):
        """
        Initializes the intermediate deviation limits.
        
        Args:
            period (int): Lookback evaluation horizon. Defaults to 50.
        """
        super().__init__(name='dist_sma_50d', description='Distance from 50D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes strictly normalized mean-reversion potentials bounding historical trends.
        
        Args:
            df (pd.DataFrame): Foundational evaluating bounds.
            
        Returns:
            pd.Series: Evaluated parameter mapping sequence deviations.
        """
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
    def __init__(self, period: int = 200):
        """
        Initializes the institutional macro baseline.
        
        Args:
            period (int): Structural baseline configuration limit. Defaults to 200.
        """
        super().__init__(name='dist_sma_200d', description='Distance from 200D SMA', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Quantifies macroscopic euphoric constraints bounded strictly by structural parameters.
        
        Args:
            df (pd.DataFrame): Time-series matrix extraction limits.
            
        Returns:
            pd.Series: Extracted structural deviation coordinates.
        """
        ma = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=self.period).mean())
        return (df['close'] - ma) / (ma + EPS)
    

@FactorRegistry.register()
class ZScore21D(TechnicalFactor):
    """
    Price Z-Score (21-Day).
    
    Measures how many standard deviations the current price is from its 21-day mean.
    Assumption: Short-term returns are normally distributed.
    """
    def __init__(self, period: int = 21):
        """
        Initializes parametric standard score boundaries.
        
        Args:
            period (int): Structural temporal extraction sequence. Defaults to 21.
        """
        super().__init__(name='zscore_21d', description='Price Z-Score 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts statistical variance evaluating bounded spatial limits continuously.
        
        Args:
            df (pd.DataFrame): Bounding evaluation limits.
            
        Returns:
            pd.Series: Cross-sectional spatial variances extracted safely.
        """
        def calc_z(x):
            return (x - x.rolling(self.period).mean()) / (x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)
    
@FactorRegistry.register()
class ZScore63D(TechnicalFactor):
    """
    Price Z-Score (63-Day/Quarterly).
    
    Measures quarterly mean reversion tendencies.
    """
    def __init__(self, period: int = 63):
        """
        Initializes explicit long-term deviation matrices limits.
        
        Args:
            period (int): The trailing lookback execution parameter. Defaults to 63.
        """
        super().__init__(name='zscore_63d', description='Price Z-Score 63D', lookback_period=period + 5)
        self.period = period
 
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates structurally bound normal limits assessing trailing parameters identically.
        
        Args:
            df (pd.DataFrame): Matrix bounding limit structures.
            
        Returns:
            pd.Series: Strictly distributed deviation sequences.
        """
        def calc_z(x):
            return (x - x.rolling(self.period).mean()) / (x.rolling(self.period).std() + EPS)
        return df.groupby('ticker')['close'].transform(calc_z)      


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
    def __init__(self, period: int = 20, std: float = 2.0):
        """
        Initializes geometric bounded bandwidth mapping configuration.
        
        Args:
            period (int): Bounding simple moving average baseline length. Defaults to 20.
            std (float): Boundary constraint weighting empirical distributions. Defaults to 2.0.
        """
        super().__init__(name='mr_bb_pos', description='BB Position %B', lookback_period=period + 5)
        self.period = period
        self.std = std
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts absolute statistical boundary representations strictly projecting parameters structurally.
        
        Args:
            df (pd.DataFrame): Evaluation spatial coordinates.
            
        Returns:
            pd.Series: Scaled continuous representation of local bandwidth probabilities.
        """
        def calc_bb(x):
            mean = x.rolling(self.period).mean()
            std = x.rolling(self.period).std()
            upper = mean + (self.std * std)
            lower = mean - (self.std * std)
            
            # Systematically scales positional coordinates evaluating structural bandwidth boundaries
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
    def __init__(self, period: int = 20, std: float = 2.0):
        """
        Initializes the dynamic absolute variance scaler limit bounds.
        
        Args:
            period (int): Length parameter defining structural standard distribution length. Defaults to 20.
            std (float): Discrete multiplier isolating trailing spatial coordinates. Defaults to 2.0.
        """
        super().__init__(name='mr_bb_width', description='BB Width', lookback_period=period + 5)
        self.period = period
        self.std = std
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates raw standard deviations scaled linearly against median trend vectors securely.
        
        Args:
            df (pd.DataFrame): Explicitly evaluating array parameter definitions.
            
        Returns:
            pd.Series: Mapped strictly bounding variance limits mathematically.
        """
        def calc_width(x):
            mean = x.rolling(self.period).mean()
            std = x.rolling(self.period).std()
            upper = mean + (self.std * std)
            lower = mean - (self.std * std)
            
            return (upper - lower) / (mean + EPS)
        return df.groupby('ticker')['close'].transform(calc_width)


@FactorRegistry.register()
class MACrossover5_21(TechnicalFactor):
    """
    Moving Average Spread (5D vs 21D).
    
    Calculates the normalized percentage distance between Fast and Slow MAs.
    Extreme spreads often precede mean reversion.
    
    Formula:
    $$ Spread = \frac{MA_{fast} - MA_{slow}}{MA_{slow}} $$
    """
    def __init__(self, fast: int = 5, slow: int = 21):
        """
        Initializes dual sequence convergence tracking arrays.
        
        Args:
            fast (int): The aggressive front-running sequence configuration length. Defaults to 5.
            slow (int): The trailing structural baseline limit sequence. Defaults to 21.
        """
        super().__init__(name='ma_cross_5_21', description='MA 5-21 Spread', lookback_period=slow + 5)
        self.fast = fast
        self.slow = slow

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates temporal crossover variations measuring momentum decay dynamically.
        
        Args:
            df (pd.DataFrame): Source matrices enforcing historical limits.
            
        Returns:
            pd.Series: Divergence mappings cleanly extracted.
        """
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
    def __init__(self, fast: int = 21, slow: int = 63):
        """
        Initializes structural trailing macro crossover limits.
        
        Args:
            fast (int): Lead moving execution bounds. Defaults to 21.
            slow (int): Lag boundary mapping execution targets. Defaults to 63.
        """
        super().__init__(name='ma_cross_21_63', description='MA 21-63 Spread', lookback_period=slow + 5)
        self.fast = fast
        self.slow = slow
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts scaled differential structures tracking temporal macro overlaps.
        
        Args:
            df (pd.DataFrame): Mathematical bounding arrays scaling sequences natively.
            
        Returns:
            pd.Series: Linear distributions mapped identically.
        """
        ma_fast = df.groupby('ticker')['close'].transform(lambda x: x.rolling(self.fast).mean())
        ma_slow = df.groupby('ticker')['close'].transform(lambda x: x.rolling(self.slow).mean())
        return (ma_fast - ma_slow) / (ma_slow + EPS)


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
    def __init__(self, period: int = 252):
        """
        Initializes the spatial limits binding global statistical price arrays.
        
        Args:
            period (int): Defines temporal bounds encapsulating empirical trading years. Defaults to 252.
        """
        super().__init__(name='price_pos_52w', description='Position in 52W Range', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates bounded percentage limits standardizing sequence price coordinates precisely.
        
        Args:
            df (pd.DataFrame): Evaluating limits modeling execution targets natively.
            
        Returns:
            pd.Series: Continuously mapped limits isolating distribution parameters cleanly.
        """
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
    

@FactorRegistry.register()
class CCI(TechnicalFactor):
    """
    Commodity Channel Index (20D).
    
    Measures the deviation of the Typical Price from its SMA, normalized by
    Mean Absolute Deviation (MAD).
    
    Formula:
    $$ CCI = \frac{TP - SMA_{TP}}{0.015 \times \text{MeanDev}} $$
    """
    def __init__(self, period: int = 20):
        """
        Initializes dynamic channel extraction algorithms.
        
        Args:
            period (int): Length parameter limiting standard moving deviations. Defaults to 20.
        """
        super().__init__(name='cci_20', description='Commodity Channel Index', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates strict oscillator boundaries explicitly evaluating statistical channel geometries.
        
        Args:
            df (pd.DataFrame): Fundamental pricing frames structuring arrays strictly.
            
        Returns:
            pd.Series: Continuous vector limits bounding dynamic variations uniformly.
        """
        def calc_cci(x):
            high = safe_col(x, "high")
            low = safe_col(x, "low")
            if high.isna().all():
                return pd.Series(np.nan, index=x.index)

            tp = (high + low + x['close']) / 3
            
            sma_tp = tp.rolling(self.period).mean()

            # Extracts Mean Absolute Deviation utilizing localized applying mapping loops natively.
            def mad(a):
                return np.mean(np.abs(a - np.mean(a)))
            
            mean_dev = tp.rolling(window=self.period).apply(mad, raw=True)

            cci = (tp - sma_tp) / (mean_dev * 0.015 + EPS)
            return cci
        return df.groupby('ticker', group_keys=False).apply(calc_cci, include_groups=False)
    



  


    


                  