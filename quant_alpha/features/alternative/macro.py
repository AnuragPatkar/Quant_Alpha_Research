"""
Macroeconomic Alternative Factors (4 Factors)
Focus: Oil prices, bond yields, currency strength, market index levels.

Factors (Ranked by Importance):
1. Oil Momentum -> Oil price momentum (commodity cycle signal)
2. USD Strength -> Dollar index level trend (export/growth signal)
3. Yield Trend -> 10Y yield momentum (growth expectations)
4. Macro Economic Score -> Composite macro health indicator
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class OilMomentum(AlternativeFactor):
    """
    Oil Price Momentum: Change in crude oil prices over 21 days.
    Formula: (Current Oil - Oil 21D ago) / Oil 21D ago * 100
    
    Why: Oil prices signal inflation expectations and growth sentiment.
    Higher oil = Inflationary environment, stronger economy.
    Economic indicator with 2-3 quarter lead on corporate earnings.
    """
    def __init__(self):
        super().__init__(name='alt_oil_momentum', description='Oil Price Momentum (21D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'oil_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'oil_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # 21-day momentum
        momentum = df['oil_close'].pct_change(21) * 100
        return momentum.bfill()


@FactorRegistry.register()
class USDStrength(AlternativeFactor):
    """
    USD Index Strength: Dollar index level normalized to rolling mean.
    Formula: (USD Index - 20D MA) / 20D MA * 100
    
    Why: Strong dollar = Headwind for US exporters, capital flows.
    Weak dollar = Inflation boost, EM asset appreciation.
    Leading indicator for stock rotation (large cap vs small cap).
    """
    def __init__(self):
        super().__init__(name='alt_usd_strength', description='USD Index Strength (Normalized)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'usd_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'usd_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # Normalize to rolling mean
        ma_20 = df['usd_close'].rolling(window=20, min_periods=1).mean()
        strength = ((df['usd_close'] - ma_20) / ma_20) * 100
        return strength.fillna(0)


@FactorRegistry.register()
class YieldTrend(AlternativeFactor):
    """
    10Y Treasury Yield Trend: Change in long-term yields (growth signal).
    Formula: (Current Yield - Yield 63D ago) / Yield 63D ago * 100
    
    Why: Rising yields = Growth expectations improving, inflation rising.
    Falling yields = Recession fears, flight to safety.
    Lead indicator for equity risk premium and duration rotation.
    """
    def __init__(self):
        super().__init__(name='alt_yield_trend', description='10Y Yield Momentum (63D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'us_10y_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # 63-day (3-month) momentum in yields
        yield_momentum = df['us_10y_close'].diff(63)
        return yield_momentum.bfill()


@FactorRegistry.register()
class MacroEconomicScore(AlternativeFactor):
    """
    Macro Economic Health Score: Composite of Oil, USD, Yields (0-100 scale).
    Formula: normalize(oil_momentum) + normalize(usd_strength) + normalize(yield_trend)
    
    Why: Holistic macro health indicator combining inflation, growth, and sentiment.
    Higher score = Favorable macro backdrop for equities.
    Lower score = Defensive macro environment.
    """
    def __init__(self):
        super().__init__(name='alt_macro_score', description='Macroeconomic Health Score (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'usd_close', 'us_10y_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing macro data columns")
            return pd.Series(np.nan, index=df.index)
        
        # Normalize each component to z-score
        oil_mom = df['oil_close'].pct_change(21)
        usd_norm = ((df['usd_close'] - df['usd_close'].rolling(20).mean()) / df['usd_close'].rolling(20).std())
        yield_mom = df['us_10y_close'].diff(63)
        
        # Normalize to 0-100 scale
        oil_z = (oil_mom - oil_mom.rolling(63).mean()) / (oil_mom.rolling(63).std() + 1e-8)
        usd_z = usd_norm
        yield_z = (yield_mom - yield_mom.rolling(63).mean()) / (yield_mom.rolling(63).std() + 1e-8)
        
        # Composite: average of components, scale to 0-100
        composite = (oil_z + usd_z + yield_z) / 3
        score = 50 + (composite * 10)  # Center at 50, scale by 10
        
        return score.clip(lower=0, upper=100).fillna(50)
