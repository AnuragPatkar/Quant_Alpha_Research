"""
Volume & Liquidity Factors (Production Hardened)
Capture trading activity, liquidity shocks, and smart money flow.
Total Factors: 12

Key Insights:
- Volume Z-Score: Detects statistical volume anomalies.
- Amihud: Measures price impact per unit of volume (Illiquidity).
- Price-Vol Corr: Does volume confirm the trend?
- VWAP: Institutional benchmark for fair value.
- Force Index: Combines price movement and volume strength.
- Ease of Movement: Identifies 'Easy' moves (low resistance).
- CMF & MFI: Classic flow oscillators for divergence detection.
"""

import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import TechnicalFactor

EPS = 1e-9  # Prevent Division by Zero

# ==================== 1. VOLUME ANOMALIES (Z-SCORE) ====================

@FactorRegistry.register()
class VolumeZScore5D(TechnicalFactor):
    def __init__(self, period=5):
        super().__init__(name='vol_zscore_5d', description='Volume Z-Score 5D', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return (x - x.rolling(window=self.period).mean()) / (x.rolling(window=self.period).std() + EPS)
        return df.groupby('ticker')['volume'].transform(calc_z)

@FactorRegistry.register()
class VolumeZScore21D(TechnicalFactor):
    def __init__(self, period=21):
        super().__init__(name='vol_zscore_21d', description='Volume Z-Score 21D', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_z(x):
            return (x - x.rolling(window=self.period).mean()) / (x.rolling(window=self.period).std() + EPS)
        return df.groupby('ticker')['volume'].transform(calc_z)
    
# ==================== 2. RELATIVE VOLUME (RATIO) ====================

@FactorRegistry.register()
class VolumeMA20Ratio(TechnicalFactor):
    """Ratio of Volume to 20D Moving Average"""
    def __init__(self, period=20):
        super().__init__(name='vol_ma20_ratio', description='Volume MA20 Ratio', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        ma_vol = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=self.period).mean())
        return df['volume'] / (ma_vol + EPS)
    
# ==================== 3. LIQUIDITY & TURNOVER ====================

@FactorRegistry.register()
class TurnoverRate(TechnicalFactor):
    """Proxy for Turnover: Log(Close * Volume)"""
    def __init__(self, period=21):
        super().__init__(name='turnover_rate', description='Log Dollar Volume (Liquidity)', lookback_period = period + 5)
        self.period = period
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        dollar_vol = df['close'] * df['volume']
        return dollar_vol.groupby(df['ticker']).transform(lambda x: np.log(x.rolling(window=self.period).mean() + EPS))
 
@FactorRegistry.register()
class Amihud63D(TechnicalFactor):
    """
    Amihud Illiquidity Ratio (63D)
    High Value = Illiquid (Small volume moves price a lot)
    """
    def __init__(self, period=63):
        super().__init__(name='amihud_63d', description='Amihud Illiquidity 63D', lookback_period=period + 5)
        self.period = period
    
    def compute(self, df:pd.DataFrame) -> pd.Series:
        dollar_vol = (df['close'] * df['volume']) + EPS
        dollar_illiq = df['close'].pct_change().abs() / dollar_vol
        return dollar_illiq.groupby(df['ticker']).transform(lambda x: x.rolling(window=self.period).mean() * 1e6)
    
# ==================== 4. SMART MONEY & INSTITUTIONAL ====================

@FactorRegistry.register()
class VWAPDistance(TechnicalFactor):
    """Distance from 21-Day Rolling VWAP"""
    def __init__(self, period=21):
        super().__init__(name='vwap_dist_21d', description='Distance from 21D VWAP', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_vwap_dist(x):
            pv = x['close'] * x['volume']
            cum_pv = pv.rolling(window=self.period).sum()
            cum_vol = x['volume'].rolling(window=self.period).sum()
            vwap = cum_pv / (cum_vol + EPS)
            return (x['close'] / vwap) - 1.0

        return df.groupby('ticker', group_keys=False).apply(calc_vwap_dist)

@FactorRegistry.register()
class OnBalanceVolumeSlope(TechnicalFactor):
    """OBV Slope (Rate of Change)"""
    def __init__(self, period=10):
        super().__init__(name='obv_slope_10d', description='OBV Rate of Change 10D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_obv_slope(x):
            change = x['close'].diff()
            direction = np.where(change > 0, 1, -1)
            direction[change == 0] = 0
            
            obv = (direction * x['volume']).cumsum()
            avg_vol = x['volume'].rolling(window=self.period).mean()
            slope = obv.diff(self.period) / (avg_vol * self.period + EPS)
            return slope

        return df.groupby('ticker', group_keys=False).apply(calc_obv_slope)
    
# ==================== 5. CORRELATION ====================

@FactorRegistry.register()
class PriceVolumeCorr21D(TechnicalFactor):
    """Correlation between Price and Volume (21D)"""
    def __init__(self, period=21):
        super().__init__(name='price_vol_corr_21d', description='Price-Volume Correlation 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_corr(x):
            return x['close'].rolling(window=self.period).corr(x['volume'])
            
        return df.groupby('ticker', group_keys=False).apply(calc_corr).fillna(0)

# ==================== 6. FORCE & EFFICIENCY ====================

@FactorRegistry.register()
class ForceIndex14D(TechnicalFactor):
    """
    Elder's Force Index (14D)
    Force = PriceChange * Volume. Smoothed by EMA.
    """
    def __init__(self, period=14):
        super().__init__(name='force_index_14d', description='Elder Force Index 14D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_force(x):
            # Change * Volume
            raw_force = x['close'].diff() * x['volume']
            
            # EMA Smoothing
            ema_force = raw_force.ewm(span=self.period, adjust=False).mean()
            
            # Normalize by average dollar volume to scale it
            norm_factor = (x['close'] * x['volume']).rolling(window=self.period).mean()
            return ema_force / (norm_factor + EPS)

        return df.groupby('ticker', group_keys=False).apply(calc_force)

@FactorRegistry.register()
class EaseOfMovement14(TechnicalFactor):
    """
    Ease of Movement (EMV)
    High EMV = Price rising on low volume (Low resistance).
    """
    def __init__(self, period=14):
        super().__init__(name='emv_14', description='Ease of Movement 14D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_emv(x):
            # Midpoint Move
            high_low_mid = (x['high'] + x['low']) / 2
            mid_move = high_low_mid.diff()
            
            # Box Ratio: Volume / Range
            box_ratio = (x['volume'] + EPS) / ((x['high'] - x['low']) + EPS)
            
            # Raw EMV
            emv = mid_move / box_ratio
            
            # Smoothed EMV
            return emv.rolling(window=self.period).mean() * 1e8 # Scale up

        return df.groupby('ticker', group_keys=False).apply(calc_emv)
                                                        
# ==================== 7. CLASSIC FLOW OSCILLATORS ====================

@FactorRegistry.register()
class ChaikinMoneyFlow21D(TechnicalFactor):
    """
    Chaikin Money Flow (CMF)
    Close near High = Accumulation.
    Close near Low = Distribution.
    """
    def __init__(self, period=21):
        super().__init__(name='cmf_21d', description='Chaikin Money Flow 21D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_cmf(x):
            # 1. Money Flow Multiplier
            range_len = (x['high'] - x['low']) + EPS
            mf_multiplier = ((x['close'] - x['low']) - (x['high'] - x['close'])) / range_len
            
            # 2. Money Flow Volume
            mf_volume = mf_multiplier * x['volume']
            
            # 3. CMF = Sum(MF Vol) / Sum(Vol)
            rolling_mf_vol = mf_volume.rolling(window=self.period).sum()
            rolling_vol = x['volume'].rolling(window=self.period).sum()
            
            return rolling_mf_vol / (rolling_vol + EPS)

        return df.groupby('ticker', group_keys=False).apply(calc_cmf)

@FactorRegistry.register()
class MoneyFlowIndex14(TechnicalFactor):
    """
    Money Flow Index (MFI) - The "Volume-Weighted RSI".
    """
    def __init__(self, period=14):
        super().__init__(name='mfi_14', description='Money Flow Index 14D', lookback_period=period + 5)
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.Series:
        def calc_mfi(x):
            # 1. Typical Price
            tp = (x['high'] + x['low'] + x['close']) / 3
            
            # 2. Raw Money Flow
            raw_mf = tp * x['volume']
            
            # 3. Positive vs Negative Flow
            tp_diff = tp.diff()
            
            pos_flow = raw_mf.where(tp_diff > 0, 0)
            neg_flow = raw_mf.where(tp_diff < 0, 0)
            
            # 4. Rolling Sums
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
    Accumulation/Distribution Line (A/D Line)
    Combines price position within bar + volume
    More robust than OBV for catch reversals
    
    Formula:
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
            
            # 1. Money Flow Multiplier (MFM)
            # Handles division by zero: if High = Low, MFM = 0
            range_hl = high - low
            mfm = np.where(
                range_hl != 0,
                ((close - low) - (high - close)) / range_hl,
                0
            )
            
            # 2. Money Flow Volume (MFV)
            mfv = mfm * volume
            
            # 3. Cumulative A/D Line
            ad_line = pd.Series(mfv).cumsum()
            
            # Normalize by dividing by volume scaling
            ad_normalized = ad_line / (volume.sum() + EPS)
            
            return ad_normalized
        
        return df.groupby('ticker', group_keys=False).apply(calc_ad)
