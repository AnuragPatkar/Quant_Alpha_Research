"""
Volume & Liquidity Factors
==========================
Quantitative signals capturing trading activity, liquidity constraints, and institutional flow.

Purpose
-------
This module constructs alpha factors quantifying the magnitude and quality of trading
activity. It moves beyond simple volume analysis to measure:
1.  **Liquidity Risk**: Amihud Illiquidity Ratio estimates price impact cost.
2.  **Smart Money Flow**: Accumulation/Distribution and Chaikin Money Flow (CMF)
    track institutional positioning.
3.  **Trend Confirmation**: Price-Volume correlation validates whether price moves
    are supported by participation.
4.  **Anomalies**: Z-Scores identify statistically significant volume spikes often
    associated with news events or earnings surprises.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    registry = FactorRegistry()
    liq_factor = registry.get('amihud_63d')
    signals = liq_factor.compute(market_data_df)

Importance
----------
- **Execution Quality**: Liquidity metrics (Turnover, Amihud) are critical for
  capacity analysis and transaction cost modeling ($TC \propto \text{Illiquidity}$).
- **Signal Confirmation**: Volume often leads price. Divergences between price
  momentum and flow oscillators (MFI, CMF) can signal trend exhaustion.
- **Regime Detection**: High volume volatility often precedes volatility in price,
  serving as a leading indicator for regime shifts.

Tools & Frameworks
------------------
- **Pandas**: Efficient rolling window aggregations and `groupby` transformations.
- **NumPy**: Vectorized arithmetic for oscillator derivation and correlation.
- **FactorRegistry**: Decorator-based registration for pipeline integration.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor

EPS = 1e-9  # Machine epsilon for numerical stability

# ==================== 1. VOLUME ANOMALIES (Z-SCORE) ====================

# REMOVED: VolumeZScore5D (5-day volume z-score extremely noisy, unreliable)
# @FactorRegistry.register()
# class VolumeZScore5D(TechnicalFactor):
#     def __init__(self, period=5):
#         super().__init__(name='vol_zscore_5d', description='Volume Z-Score 5D', lookback_period=period + 5)
#         self.period = period
#     
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         def calc_z(x):
#             return (x - x.rolling(window=self.period).mean()) / (x.rolling(window=self.period).std() + EPS)
#         return df.groupby('ticker')['volume'].transform(calc_z)

@FactorRegistry.register()
class VolumeZScore21D(TechnicalFactor):
    """
    Volume Z-Score (21D).
    
    Standardizes volume to detect statistical anomalies.
    
    Formula:
    $$ Z_t = \frac{V_t - \mu_{21}}{\sigma_{21}} $$
    """
    def __init__(self, period=21):
        super().__init__(name='vol_zscore_21d', description='Volume Z-Score 21D', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_z(x):
            # Standard Score calculation with epsilon for stability
            return (x - x.rolling(window=self.period).mean()) / (x.rolling(window=self.period).std() + EPS)
        return df.groupby('ticker')['volume'].transform(calc_z)
    
# ==================== 2. RELATIVE VOLUME (RATIO) ====================

@FactorRegistry.register()
class VolumeMA20Ratio(TechnicalFactor):
    """
    Relative Volume Ratio (20D).
    
    Compares current volume to its recent average.
    Formula: $$ R_t = \frac{V_t}{\text{SMA}(V_t, 20)} $$
    """
    def __init__(self, period=20):
        super().__init__(name='vol_ma20_ratio', description='Volume MA20 Ratio', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_vol = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=self.period).mean())
        return df['volume'] / (ma_vol + EPS)
    
# ==================== 3. LIQUIDITY & TURNOVER ====================

@FactorRegistry.register()
class TurnoverRate(TechnicalFactor):
    """
    Logarithmic Dollar Volume (Liquidity Proxy).
    
    Estimates the average daily turnover in dollar terms.
    Formula: $$ L_t = \ln(\text{SMA}(P_t \times V_t, 21)) $$
    """
    def __init__(self, period=21):
        super().__init__(name='turnover_rate', description='Log Dollar Volume (Liquidity)', lookback_period = period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        dollar_vol = df['close'] * df['volume']
        # Log-transform smoothes the distribution of dollar volume across caps
        return dollar_vol.groupby(df['ticker']).transform(lambda x: np.log(x.rolling(window=self.period).mean() + EPS))
 
@FactorRegistry.register()
class Amihud63D(TechnicalFactor):
    """
    Amihud Illiquidity Ratio (63D).
    
    Measures the price impact per unit of volume. High values indicate low liquidity
    (large price moves on small volume).
    
    Formula:
    $$ ILLIQ_T = \frac{1}{N} \sum_{t=1}^{N} \frac{|R_t|}{P_t V_t} $$
    """
    def __init__(self, period=63):
        super().__init__(name='amihud_63d', description='Amihud Illiquidity 63D', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        dollar_vol = (df['close'] * df['volume']) + EPS
        dollar_illiq = df['close'].pct_change().abs() / dollar_vol
        # Scaling factor (1e6) adjusts values to a readable range
        return dollar_illiq.groupby(df['ticker']).transform(lambda x: x.rolling(window=self.period).mean() * 1e6)
    
# ==================== 4. SMART MONEY & INSTITUTIONAL ====================

# REMOVED: VWAPDistance (redundant with price-based technical signals)
# @FactorRegistry.register()
# class VWAPDistance(TechnicalFactor):
#     """Distance from 21-Day Rolling VWAP"""
#     def __init__(self, period=21):
#         super().__init__(name='vwap_dist_21d', description='Distance from 21D VWAP', lookback_period=period + 5)
#         self.period = period
# 
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         def calc_vwap_dist(x):
#             pv = x['close'] * x['volume']
#             cum_pv = pv.rolling(window=self.period).sum()
#             cum_vol = x['volume'].rolling(window=self.period).sum()
#             vwap = cum_pv / (cum_vol + EPS)
#             return (x['close'] / vwap) - 1.0
# 
#         return df.groupby('ticker', group_keys=False).apply(calc_vwap_dist)
# 
# REMOVED: OnBalanceVolumeSlope (OBV is lagging indicator, marginal predictive power)
# @FactorRegistry.register()
# class OnBalanceVolumeSlope(TechnicalFactor):
#     """OBV Slope (Rate of Change)"""
#     def __init__(self, period=10):
#         super().__init__(name='obv_slope_10d', description='OBV Rate of Change 10D', lookback_period=period + 5)
#         self.period = period
# 
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         def calc_obv_slope(x):
#             change = x['close'].diff()
#             direction = np.where(change > 0, 1, -1)
#             direction[change == 0] = 0
#             
#             obv = (direction * x['volume']).cumsum()
#             avg_vol = x['volume'].rolling(window=self.period).mean()
#             slope = obv.diff(self.period) / (avg_vol * self.period + EPS)
#             return slope
# 
#         return df.groupby('ticker', group_keys=False).apply(calc_obv_slope)
    
# ==================== 5. CORRELATION ====================

@FactorRegistry.register()
class PriceVolumeCorr21D(TechnicalFactor):
    """
    Price-Volume Correlation (21D).
    
    Measures the linear relationship between price changes and volume levels.
    - Positive $\\rho$: Volume expands on rallies (Bullish).
    - Negative $\\rho$: Volume expands on sell-offs (Bearish/Panic).
    """
    def __init__(self, period=21):
        super().__init__(name='price_vol_corr_21d', description='Price-Volume Correlation 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_corr(x):
            # Rolling Pearson Correlation
            return x['close'].rolling(window=self.period).corr(x['volume'])
            
        return df.groupby('ticker', group_keys=False).apply(calc_corr).fillna(0)

# ==================== 6. FORCE & EFFICIENCY ====================

@FactorRegistry.register()
class ForceIndex14D(TechnicalFactor):
    """
    Elder's Force Index (14D).
    
    Combines price movement magnitude and volume to measure buying/selling pressure.
    
    Formula:
    $$ FI = \text{EMA}(\Delta P \times V, 14) $$
    """
    def __init__(self, period=14):
        super().__init__(name='force_index_14d', description='Elder Force Index 14D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_force(x):
            # Raw Force = \Delta P * V
            raw_force = x['close'].diff() * x['volume']
            
            # EMA Smoothing to reduce noise
            ema_force = raw_force.ewm(span=self.period, adjust=False).mean()
            
            # Normalize by average dollar volume to scale it
            norm_factor = (x['close'] * x['volume']).rolling(window=self.period).mean()
            return ema_force / (norm_factor + EPS)

        return df.groupby('ticker', group_keys=False).apply(calc_force)

# REMOVED: EaseOfMovement14 (derivative of ATR/price range, conceptually covered by ATR)
# @FactorRegistry.register()
# class EaseOfMovement14(TechnicalFactor):
#     """
#     Ease of Movement (EMV)
#     High EMV = Price rising on low volume (Low resistance).
#     """
#     def __init__(self, period=14):
#         super().__init__(name='emv_14', description='Ease of Movement 14D', lookback_period=period + 5)
#         self.period = period
# 
#     def compute(self, df: pd.DataFrame) -> pd.Series:
#         def calc_emv(x):
#             # Midpoint Move
#             high_low_mid = (x['high'] + x['low']) / 2
#             mid_move = high_low_mid.diff()
#             
#             # Box Ratio: Volume / Range
#             box_ratio = (x['volume'] + EPS) / ((x['high'] - x['low']) + EPS)
#             
#             # Raw EMV
#             emv = mid_move / box_ratio
#             
#             # Smoothed EMV
#             return emv.rolling(window=self.period).mean() * 1e8 # Scale up
# 
#         return df.groupby('ticker', group_keys=False).apply(calc_emv)
                                                        
# ==================== 7. CLASSIC FLOW OSCILLATORS ====================

@FactorRegistry.register()
class ChaikinMoneyFlow21D(TechnicalFactor):
    """
    Chaikin Money Flow (CMF).
    
    Measures the accumulation/distribution line over a specific period.
    Captures flow relative to the high-low range.
    """
    def __init__(self, period=21):
        super().__init__(name='cmf_21d', description='Chaikin Money Flow 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_cmf(x):
            # 1. Money Flow Multiplier (MFM): Position of Close within High-Low range
            range_len = (x['high'] - x['low']) + EPS
            mf_multiplier = ((x['close'] - x['low']) - (x['high'] - x['close'])) / range_len
            
            # 2. Money Flow Volume (MFV)
            mf_volume = mf_multiplier * x['volume']
            
            # 3. CMF = \frac{\sum MFV}{\sum V}
            rolling_mf_vol = mf_volume.rolling(window=self.period).sum()
            rolling_vol = x['volume'].rolling(window=self.period).sum()
            
            return rolling_mf_vol / (rolling_vol + EPS)

        return df.groupby('ticker', group_keys=False).apply(calc_cmf)

@FactorRegistry.register()
class MoneyFlowIndex14(TechnicalFactor):
    """
    Money Flow Index (MFI).
    
    Momentum indicator that uses price and volume to predict overbought or oversold conditions.
    Often referred to as volume-weighted RSI.
    """
    def __init__(self, period=14):
        super().__init__(name='mfi_14', description='Money Flow Index 14D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_mfi(x):
            # 1. Typical Price
            tp = (x['high'] + x['low'] + x['close']) / 3
            
            # 2. Raw Money Flow (Typical Price * Volume)
            raw_mf = tp * x['volume']
            
            # 3. Separate Positive and Negative Flow based on price direction
            tp_diff = tp.diff()
            
            pos_flow = raw_mf.where(tp_diff > 0, 0)
            neg_flow = raw_mf.where(tp_diff < 0, 0)
            
            # 4. Rolling Sums (Ratio Calculation)
            pos_mf_sum = pos_flow.rolling(window=self.period).sum()
            neg_mf_sum = neg_flow.rolling(window=self.period).sum()
            
            # 5. Money Ratio & MFI
            mr = pos_mf_sum / (neg_mf_sum + EPS)
            mfi = 100 - (100 / (1 + mr))
            
            return mfi

        return df.groupby('ticker', group_keys=False).apply(calc_mfi)


# ==================== 8. ACCUMULATION/DISTRIBUTION LINE ====================

@FactorRegistry.register()
class AccumulationDistribution(TechnicalFactor):
    """
    Accumulation/Distribution Line (A/D).
    
    Cumulative measure of volume flow based on the close price's location within the
    daily High-Low range. Acts as a leading indicator for price reversals.
    
    Logic:
    1. Money Flow Multiplier (MFM) = [(Close - Low) - (High - Close)] / (High - Low)
    2. Money Flow Volume (MFV) = MFM * Volume
    3. A/D Line = Cumulative sum of MFV
    """
    def __init__(self):
        super().__init__(name='ad_line', description='Accumulation/Distribution Line', lookback_period=2)
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_ad(group):
            high = group['high']
            low = group['low']
            close = group['close']
            volume = group['volume']
            
            # 1. Money Flow Multiplier (MFM) - Quantification of buying/selling pressure within bar
            # Handles division by zero: if High = Low, MFM = 0
            range_hl = high - low
            mfm = np.where(
                range_hl != 0,
                ((close - low) - (high - close)) / range_hl,
                0
            )
            
            # 2. Money Flow Volume (MFV) - Volume adjusted by pressure
            mfv = mfm * volume
            
            # 3. Cumulative A/D Line - Running total
            ad_line = pd.Series(mfv).cumsum()
            
            # Normalize by dividing by volume scaling
            ad_normalized = ad_line / (volume.sum() + EPS)
            
            return ad_normalized
        
        return df.groupby('ticker', group_keys=False).apply(calc_ad)
