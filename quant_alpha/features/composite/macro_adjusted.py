import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MacroAdjustedMomentum(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_macro_momentum', description='Momentum modulated by VIX')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Stock-specific close and Market VIX
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # 1. Individual Stock Momentum (21-day)
        # Groupby 'ticker' is CRITICAL here to avoid mixing stock prices
        momentum = df.groupby('ticker')['close'].pct_change(21) * 100
        
        # 2. VIX Adjustment (Inverse volatility weight)
        # We use transform to broadcast market-wide VIX back to all tickers
        vix_smooth = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean()).clip(10, 40)
        adjustment = 25 / vix_smooth
        
        return momentum * adjustment

@FactorRegistry.register()
class OilCorrectedValue(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_oil_value', description='Value signals filtered by oil environment')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'pe_ratio'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Oil regime (Market-wide)
        oil_median = df.groupby('ticker')['oil_close'].transform(lambda x: x.rolling(252).median())
        oil_regime = np.where(df['oil_close'] < oil_median, 1.0, 0.5)
        
        # Value Signal: Inverse PE (Earnings Yield proxy)
        # clipping to avoid extreme values from low earnings
        value_signal = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 100)
        
        return value_signal * oil_regime

@FactorRegistry.register()
class RateEnvironmentScore(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_rate_quality', description='Quality adjusted by interest rates')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'us_10y_close', 'roe'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Rate weight: Higher rates = Lower weight for high-duration quality
        rate_weight = 1 - (df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(21).mean()).clip(0, 5) / 5)
        
        # Quality: ROE (Stock specific)
        quality_signal = df.groupby('ticker')['roe'].ffill().clip(-1, 1)
        
        return quality_signal * rate_weight

@FactorRegistry.register()
class DollarAdjustedGrowth(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_dollar_growth', description='Growth adjusted for USD strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'usd_close', 'earnings_growth'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # USD regime (Market-wide)
        usd_median = df.groupby('ticker')['usd_close'].transform(lambda x: x.rolling(252).median())
        usd_regime = np.where(df['usd_close'] < usd_median, 1.0, 0.7)
        
        # Growth Signal (Stock specific)
        growth_signal = df.groupby('ticker')['earnings_growth'].ffill().clip(-1, 1)
        
        return growth_signal * usd_regime

@FactorRegistry.register()
class RiskParityBlend(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_risk_parity', description='Momentum + Sentiment Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Stock-specific Momentum Rank
        mom = df.groupby('ticker')['close'].pct_change(21)
        # FIX: Group by date to rank stocks against each other on that day
        mom_rank = mom.groupby(df['date']).rank(pct=True)
        
        # Market Sentiment Rank (VIX inverted)
        vix_rank = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(252).rank(pct=True))
        sentiment_rank = 1 - vix_rank
        
        return (mom_rank + sentiment_rank) / 2