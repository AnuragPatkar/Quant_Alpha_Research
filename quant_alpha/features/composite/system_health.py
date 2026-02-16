import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MarketRegimeScore(CompositeFactor):
    """
    Classifies market as Bull (1), Bear (-1), or Sideways (0).
    Robustified with min_periods to avoid early-data NaNs.
    """
    def __init__(self):
        super().__init__(name='comp_regime', description='Market Regime Classification')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'sp500_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 1. Structural Trend (Long-term)
        # Using 200-day SMA as the anchor
        ma200 = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200, min_periods=50).mean())
        price_above_ma = df['sp500_close'] > ma200
        
        # 2. Cyclical Momentum (Medium-term)
        returns_21d = df.groupby('ticker')['sp500_close'].pct_change(21)
        
        # Logic: 
        # Bull: Price > MA200 AND Returns > 0
        # Bear: Price < MA200 AND Returns < 0
        # Sideways: Everything else
        regime = pd.Series(0, index=df.index)
        regime[price_above_ma & (returns_21d > 0)] = 1
        regime[~price_above_ma & (returns_21d < 0)] = -1
        
        return regime.ffill().fillna(0)

@FactorRegistry.register()
class VolatilityRegime(CompositeFactor):
    """
    Categorical Volatility Score.
    0: Low Vol (Risk-on), 1: Elevated (Caution), 2: High Vol (Risk-off/Crisis)
    """
    def __init__(self):
        super().__init__(name='comp_vol_regime', description='Volatility Regime Levels')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Using a 5-day smooth to avoid single-day VIX spikes
        vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        # np.select is faster and cleaner for multiple categories
        conditions = [
            (vix < 17), # Normal/Quiet
            (vix >= 17) & (vix < 28), # Elevated
            (vix >= 28) # Crisis
        ]
        choices = [0, 1, 2]
        
        return pd.Series(np.select(conditions, choices, default=1), index=df.index)

@FactorRegistry.register()
class CapitalFlowSignal(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_capital_flow', description='Oil & USD Flow Indicator')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'usd_close'}.issubset(df.columns):
            return pd.Series(50, index=df.index)
        
        # We look for consensus in momentum
        # Normalize 21d momentum by its 63d volatility for a 'Z-score' like feel
        oil_mom = df.groupby('ticker')['oil_close'].pct_change(21)
        usd_mom = df.groupby('ticker')['usd_close'].pct_change(21)
        
        # Combine and scale to 0-100. (Mean of 0 momentum = 50 score)
        combined = (oil_mom + usd_mom) / 2
        flow_score = ((combined + 0.05) / 0.10).clip(0, 1) * 100
        
        return flow_score.fillna(50)

@FactorRegistry.register()
class EconomicMomentumScore(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_econ_momentum', description='Cross-Asset Macro Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        req = ['us_10y_close', 'oil_close', 'usd_close']
        if not all(col in df.columns for col in req):
            return pd.Series(50, index=df.index)
        
        # Yields Up + Oil Up - USD Up (USD is usually a headwind for global growth)
        y_mom = df.groupby('ticker')['us_10y_close'].pct_change(21)
        o_mom = df.groupby('ticker')['oil_close'].pct_change(21)
        u_mom = df.groupby('ticker')['usd_close'].pct_change(21)
        
        composite = (y_mom + o_mom - u_mom) / 3
        return ((composite + 0.04) / 0.08).clip(0, 1).fillna(0.5) * 100

@FactorRegistry.register()
class PortfolioHealthIndex(CompositeFactor):
    """
    The Ultimate 'Master Switch'.
    Blends Price, Vol, and Macro into a 0-100 'Safety' score.
    """
    def __init__(self):
        super().__init__(name='comp_health_index', description='Global Portfolio Health Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Components
        regime = self.compute_sub_score(df, 'regime') * 25 # Bull = 25, Sideways = 0
        vol = self.compute_sub_score(df, 'vol') * 25    # Low Vol = 25, Crisis = 0
        macro = self.compute_sub_score(df, 'macro') * 50 # Strong Macro = 50
        
        health = regime + vol + macro
        return health.groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).clip(0, 100).fillna(50)

    def compute_sub_score(self, df, type):
        if type == 'regime':
            # MarketRegimeScore helper
            ma = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200).mean())
            return (df['sp500_close'] > ma).astype(float)
        elif type == 'vol':
            # Invert VIX: 15 is good (1.0), 35 is bad (0.0)
            vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean())
            return ((35 - vix) / 20).clip(0, 1)
        elif type == 'macro':
            # Economic Momentum normalized to 0.0-1.0
            y_mom = df.groupby('ticker')['us_10y_close'].pct_change(21)
            return ((y_mom + 0.02) / 0.04).clip(0, 1)