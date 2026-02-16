import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class OilMomentum(AlternativeFactor):
    """
    Oil Price Momentum: Inflation and Growth signal.
    """
    def __init__(self):
        super().__init__(name='alt_oil_momentum', description='Oil Price Momentum (21D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'oil_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'oil_close'")
            return pd.Series(0, index=df.index)
        
        # 21-day momentum with 5-day smoothing to avoid price spikes
        if 'ticker' in df.columns:
            momentum = df.groupby('ticker')['oil_close'].pct_change(21) * 100
            return momentum.groupby(df['ticker']).transform(lambda x: x.rolling(window=5, min_periods=1).mean()).fillna(0)
        else:
            momentum = df['oil_close'].pct_change(21) * 100
            return momentum.rolling(window=5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class USDStrength(AlternativeFactor):
    """
    USD Index Strength: Export headwind/tailwind indicator.
    """
    def __init__(self):
        super().__init__(name='alt_usd_strength', description='USD Index Strength (Normalized)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'usd_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Deviation from 20-day mean
        if 'ticker' in df.columns:
            ma_20 = df.groupby('ticker')['usd_close'].transform(lambda x: x.rolling(window=20, min_periods=5).mean())
        else:
            ma_20 = df['usd_close'].rolling(window=20, min_periods=5).mean()
            
        strength = ((df['usd_close'] - ma_20) / (ma_20.replace(0, np.nan) + 1e-6)) * 100
        return strength.fillna(0)

@FactorRegistry.register()
class YieldTrend(AlternativeFactor):
    """
    10Y Treasury Yield Trend: Growth and Discount Rate signal.
    """
    def __init__(self):
        super().__init__(name='alt_yield_trend', description='10Y Yield Momentum (63D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 63-day (quarterly) momentum in yields
        # Rising yields = growth expectations UP or inflation UP
        if 'ticker' in df.columns:
            yield_momentum = df.groupby('ticker')['us_10y_close'].pct_change(63) * 100
        else:
            yield_momentum = df['us_10y_close'].pct_change(63) * 100
            
        return yield_momentum.fillna(0)

@FactorRegistry.register()
class MacroEconomicScore(AlternativeFactor):
    """
    Macro Health Score (0-100): Composite of Oil, USD, and Yields.
    """
    def __init__(self):
        super().__init__(name='alt_macro_score', description='Macroeconomic Health Score (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'usd_close', 'us_10y_close']
        if not all(col in df.columns for col in required):
            return pd.Series(50, index=df.index)
        
        # Calculate components
        # Helper for rolling z-score per ticker
        def z_score(series, window):
            return (series - series.rolling(window, min_periods=21).mean()) / (series.rolling(window, min_periods=21).std() + 1e-6)
        
        if 'ticker' in df.columns:
            usd_z = df.groupby('ticker')['usd_close'].transform(lambda x: z_score(x, 252))
            yield_z = df.groupby('ticker')['us_10y_close'].transform(lambda x: z_score(x.pct_change(63), 252))
            oil_z = df.groupby('ticker')['oil_close'].transform(lambda x: z_score(x.pct_change(21), 252))
        else:
            usd_z = z_score(df['usd_close'], 252)
            yield_z = z_score(df['us_10y_close'].pct_change(63), 252)
            oil_z = z_score(df['oil_close'].pct_change(21), 252)

        # Composite: In many regimes, Rising Oil + Falling USD + Rising Yields = Strong Macro
        # Note: USD is often inverse to equity strength, so we subtract its strength
        composite = (oil_z - usd_z + yield_z) / 3
        
        # Map to 0-100 (Center 50, Std Dev 15)
        score = 50 + (composite * 15)
        
        return score.clip(lower=0, upper=100).fillna(50)