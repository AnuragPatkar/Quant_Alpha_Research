import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class VIXLevel(AlternativeFactor):
    def __init__(self):
        super().__init__(name='alt_vix_level', description='VIX Level (Fear Score)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # VIX Level is an absolute value, no time-series op needed
        # Just normalize element-wise
        vix_norm = ((df['vix_close'] - 10) / 30).clip(0, 1)
        return vix_norm.fillna(0)

@FactorRegistry.register()
class VIXMomentum(AlternativeFactor):
    def __init__(self):
        super().__init__(name='alt_vix_momentum', description='VIX 5D Momentum')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Group by ticker for correct time-series change
        if 'ticker' in df.columns:
            return df.groupby('ticker')['vix_close'].pct_change(5).fillna(0) * 100
        else:
            return df['vix_close'].pct_change(5).fillna(0) * 100

@FactorRegistry.register()
class RiskOnOffSignal(AlternativeFactor):
    def __init__(self):
        super().__init__(name='alt_risk_on_off', description='Binary Risk Sentiment')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'vix_close', 'sp500_close'}.issubset(df.columns):
            return pd.Series(0, index=df.index)
        
        vix_calm = df['vix_close'] < 20
        # Rolling mean must be grouped by ticker
        if 'ticker' in df.columns:
            sp500_ma = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(50).mean())
        else:
            sp500_ma = df['sp500_close'].rolling(50).mean()
            
        market_uptrend = df['sp500_close'] > sp500_ma
        
        signal = (vix_calm & market_uptrend).astype(int)
        return signal.fillna(0)

@FactorRegistry.register()
class VolatilityStressIndex(AlternativeFactor):
    def __init__(self):
        super().__init__(name='alt_volatility_stress', description='Volatility Stress Index')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(50, index=df.index)
        
        vix_level_score = ((df['vix_close'] - 10) / 30).clip(0, 1) * 70
        if 'ticker' in df.columns:
            vix_mom = df.groupby('ticker')['vix_close'].pct_change(5).clip(0, 1) * 30
        else:
            vix_mom = df['vix_close'].pct_change(5).clip(0, 1) * 30
        
        stress = vix_level_score + vix_mom
        return stress.fillna(50)