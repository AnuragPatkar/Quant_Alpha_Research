import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class OilUSDRatio(AlternativeFactor):
    """
    Oil-USD Ratio: Commodity vs Currency strength.
    Optimized for RangeIndex (Flat DataFrame).
    """
    def __init__(self):
        super().__init__(name='alt_oil_usd_ratio', description='Oil-USD Ratio Inflation Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'usd_close'}.issubset(df.columns):
            logger.warning(f"❌ {self.name}: Missing columns")
            return pd.Series(0, index=df.index)
        
        # Simple ratio calculation
        ratio = df['oil_close'] / df['usd_close'].replace(0, np.nan)
        
        # 252-day rolling mean for normalization
        if 'ticker' in df.columns:
            ratio_ma = ratio.groupby(df['ticker']).transform(lambda x: x.rolling(window=252, min_periods=63).mean())
            signal = (ratio / ratio_ma - 1) * 100
            return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        else:
            ratio_ma = ratio.rolling(window=252, min_periods=63).mean()
            signal = (ratio / ratio_ma - 1) * 100
            return signal.rolling(5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class YieldMomentum(AlternativeFactor):
    """
    Yield Momentum: Change in growth expectations.
    """
    def __init__(self):
        super().__init__(name='alt_yield_momentum', description='Yield Momentum (Growth Expectations)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 21-day momentum (approx 1 month)
        if 'ticker' in df.columns:
            momentum = df.groupby('ticker')['us_10y_close'].pct_change(21) * 100
            return momentum.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        else:
            momentum = df['us_10y_close'].pct_change(21) * 100
            return momentum.rolling(5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class InflationProxyScore(AlternativeFactor):
    """
    Blends Oil (Real Inflation) and Yields (Expected Inflation).
    Scale: 0-100
    """
    def __init__(self):
        super().__init__(name='alt_inflation_proxy', description='Inflation Proxy (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'us_10y_close'}.issubset(df.columns):
            return pd.Series(50, index=df.index)
        
        # 1. Oil Momentum Component (60% weight)
        if 'ticker' in df.columns:
            oil_mom = df.groupby('ticker')['oil_close'].pct_change(21).groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())
            y_min = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(252, min_periods=63).min())
            y_max = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(252, min_periods=63).max())
        else:
            oil_mom = df['oil_close'].pct_change(21).rolling(5, min_periods=1).mean()
            y_min = df['us_10y_close'].rolling(252, min_periods=63).min()
            y_max = df['us_10y_close'].rolling(252, min_periods=63).max()
            
        oil_score = (oil_mom * 100).clip(-50, 50) + 50 
        
        # Range normalization (0 to 100)
        yield_score = ((df['us_10y_close'] - y_min) / (y_max - y_min + 1e-6)) * 100
        
        combined = (oil_score * 0.6 + yield_score * 0.4).clip(0, 100)
        return combined.fillna(50)

@FactorRegistry.register()
class GrowthInflationMix(AlternativeFactor):
    """
    Z-Score of Growth (Yields) vs Inflation (Oil).
    Positive = Goldilocks, Negative = Stagflation risk.
    """
    def __init__(self):
        super().__init__(name='alt_growth_inflation_mix', description='Growth vs Inflation Balance')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'us_10y_close', 'oil_close'}.issubset(df.columns):
            return pd.Series(0, index=df.index)
        
        # Growth vs Inflation signals
        if 'ticker' in df.columns:
            growth_sig = df.groupby('ticker')['us_10y_close'].pct_change(21)
            infl_sig = df.groupby('ticker')['oil_close'].pct_change(21)
            spread = growth_sig - infl_sig
            spread_mean = spread.groupby(df['ticker']).transform(lambda x: x.rolling(63, min_periods=21).mean())
            spread_std = spread.groupby(df['ticker']).transform(lambda x: x.rolling(63, min_periods=21).std())
        else:
            growth_sig = df['us_10y_close'].pct_change(21)
            infl_sig = df['oil_close'].pct_change(21)
            spread = growth_sig - infl_sig
            spread_mean = spread.rolling(63, min_periods=21).mean()
            spread_std = spread.rolling(63, min_periods=21).std()
        
        z_mix = (spread - spread_mean) / (spread_std + 1e-6)
        
        return z_mix.clip(-3, 3).fillna(0)