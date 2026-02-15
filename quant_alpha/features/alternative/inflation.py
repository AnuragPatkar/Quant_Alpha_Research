"""
Inflation & Growth Expectations Alternative Factors (4 Factors)
Focus: Commodity prices, yield curves, inflation proxies.

Factors (Ranked by Importance):
1. Oil-USD Ratio -> Inflation expectations proxy (commodity vs currency)
2. Yield Momentum -> Change in growth expectations signal
3. Inflation Proxy Score -> Combined oil+rates inflation indicator
4. Growth-Inflation Mix -> Forward expectations blend
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class OilUSDRatio(AlternativeFactor):
    """
    Oil-USD Ratio: Commodity strength vs currency strength.
    Formula: (Oil / USD) normalized by 252-day rolling mean
    
    Why: Oil rising while USD strong = Inflation beating currency (watch out).
    Oil falling while USD rising = Deflation or demand weakness.
    Captures term premium and real vs nominal inflation expectations.
    Key for portfolio hedging and sector rotation.
    """
    def __init__(self):
        super().__init__(name='alt_oil_usd_ratio', description='Oil-USD Ratio (Inflation Signal)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'usd_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing oil or USD data")
            return pd.Series(np.nan, index=df.index)
        
        # Calculate ratio
        ratio = df['oil_close'] / df['usd_close']
        
        # Normalize by 252-day rolling mean (annual baseline)
        ratio_ma = ratio.rolling(252).mean()
        ratio_normalized = (ratio / ratio_ma - 1) * 100  # % deviation
        
        return ratio_normalized.fillna(0)


@FactorRegistry.register()
class YieldMomentum(AlternativeFactor):
    """
    Yield Momentum: Rate of change in 10Y yields (growth expectations).
    Formula: (Yield[t] - Yield[t-21]) / Yield[t-21] * 100
    
    Why: Rising yields = Growth expectations up, inflation priced in.
    Falling yields = Recession fears or savings period demand.
    Leads equity markets by 1-2 weeks on average (forward indicator).
    Critical for growth factor timing.
    """
    def __init__(self):
        super().__init__(name='alt_yield_momentum', description='Yield Momentum (21D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'us_10y_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # Change momentum in yields (21-day)
        momentum = df['us_10y_close'].pct_change(21) * 100
        return momentum.fillna(0)


@FactorRegistry.register()
class InflationProxyScore(AlternativeFactor):
    """
    Inflation Proxy Score: Combined signal from oil + rates (0-100 scale).
    Formula: (Oil_momentum + Yield_level) normalized to 0-100
    
    Why: Oil price = commodity inflation trend.
    Yield level = Market-priced expected inflation.
    Combined captures both commodity and financial market inflation signals.
    """
    def __init__(self):
        super().__init__(name='alt_inflation_proxy', description='Inflation Proxy Score (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'us_10y_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing oil or yield data")
            return pd.Series(np.nan, index=df.index)
        
        # Oil momentum component (0-50 scale)
        oil_mom = df['oil_close'].pct_change(21).fillna(0)
        oil_score = (oil_mom * 100).clip(lower=-50, upper=50) + 50  # Shift to 0-100
        
        # Yield level component (0-50 scale, capped at 5-3%)
        yield_level = df['us_10y_close'].clip(lower=2, upper=5)
        yield_score = ((yield_level - 2) / 3) * 50  # 2% = 0, 5% = 50
        
        # Combined score
        inflation_score = (oil_score * 0.6 + yield_score * 0.4).clip(lower=0, upper=100)
        return inflation_score.fillna(50)


@FactorRegistry.register()
class GrowthInflationMix(AlternativeFactor):
    """
    Growth-Inflation Mix: Balance between growth (rates) and inflation (oil) expectations.
    Formula: (Yield_momentum + Oil_momentum) / 2 normalized
    
    Why: When growth > inflation = Goldilocks scenario (growth not choking on inflation).
    When inflation > growth = Stagflation risk (bad for equities).
    When both low = Secular stagnation concerns.
    Regime indicator for portfolio construction.
    """
    def __init__(self):
        super().__init__(name='alt_growth_inflation_mix', description='Growth-Inflation Mix Balance')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['us_10y_close', 'oil_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing yield or oil data")
            return pd.Series(np.nan, index=df.index)
        
        # Growth signal: Yield momentum (positive = growth expectations up)
        yield_mom = df['us_10y_close'].pct_change(21)
        
        # Inflation signal: Oil momentum (positive = inflation up)
        oil_mom = df['oil_close'].pct_change(21)
        
        # Mix: Ratio of growth to inflation
        # Normalize to -100 to +100 scale (positive = growth > inflation)
        mix = (yield_mom - oil_mom) / (yield_mom.abs() + oil_mom.abs()).clip(lower=0.001)
        mix_normalized = mix * 100
        
        return mix_normalized.fillna(0).clip(lower=-100, upper=100)
