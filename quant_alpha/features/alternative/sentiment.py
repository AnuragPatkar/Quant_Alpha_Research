"""
Market Sentiment Alternative Factors (4 Factors)
Focus: VIX volatility index, market fear gauge, risk sentiment.

Factors (Ranked by Importance):
1. VIX Level -> Absolute fear index (0-100 scale)
2. VIX Momentum -> Change in fear (trend direction)
3. Market Risk On/Off -> Binary risk sentiment signal
4. Volatility Stress Index -> Combined fear + stress measure
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class VIXLevel(AlternativeFactor):
    """
    VIX Absolute Level: Normalized fear gauge.
    Formula: (VIX - 10) / 30 -> Normalize to 0-1 scale (10=calm, 40=panicked)
    
    Why: VIX captures investor fear and option-implied volatility.
    <15 = Complacency, 15-20 = Normal, 20-30 = Elevated, >30 = Crisis.
    Leading indicator for market drawdowns (negative correlation).
    """
    def __init__(self):
        super().__init__(name='alt_vix_level', description='VIX Absolute Level (Normalized)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'vix_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # Normalize VIX: (vix - 10) / 30, capped at 0-1
        vix_norm = (df['vix_close'] - 10) / 30
        return vix_norm.clip(lower=0, upper=1)


@FactorRegistry.register()
class VIXMomentum(AlternativeFactor):
    """
    VIX Momentum: Change in fear index (5-day momentum).
    Formula: (VIX[t] - VIX[t-5]) / VIX[t-5] * 100
    
    Why: Rising VIX = Fear increasing (bearish), Falling VIX = Fear declining (bullish).
    Captures fear direction, not just absolute level.
    Useful for identifying panic stages vs. sustained anxiety.
    """
    def __init__(self):
        super().__init__(name='alt_vix_momentum', description='VIX Momentum (5D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'vix_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # 5-day momentum in VIX
        momentum = df['vix_close'].pct_change(5) * 100
        return momentum.fillna(0)


@FactorRegistry.register()
class RiskOnOffSignal(AlternativeFactor):
    """
    Risk On/Off Sentiment: Binary signal based on VIX and S&P500 momentum.
    Formula: 1 if (VIX < 20 AND SP500_momentum > 0) else 0
    
    Why: Captures market appetite for risky assets vs. safe havens.
    Risk-On (1) = Favor growth stocks, small caps, cyclicals.
    Risk-Off (0) = Favor utilities, bonds, defensive sectors.
    Rotation signal for asset allocation decisions.
    """
    def __init__(self):
        super().__init__(name='alt_risk_on_off', description='Risk On/Off Sentiment Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['vix_close', 'sp500_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing VIX or SP500 data")
            return pd.Series(np.nan, index=df.index)
        
        # Conditions for risk-on
        low_vix = df['vix_close'] < 20
        positive_momentum = df['sp500_close'].pct_change(5) > 0
        
        # Binary signal: 1 = risk-on, 0 = risk-off
        signal = (low_vix & positive_momentum).astype(int)
        return signal.fillna(0)


@FactorRegistry.register()
class VolatilityStressIndex(AlternativeFactor):
    """
    Volatility Stress Index: Combined measure of fear and stress (0-100 scale).
    Formula: VIX_level * 100 + (VIX_momentum if positive else -VIX_momentum)
    
    Why: Captures both magnitude (VIX level) and direction (momentum) of stress.
    Higher score = More stressed market (both fearful AND deteriorating).
    Lower score = Calm and stable market environment.
    """
    def __init__(self):
        super().__init__(name='alt_volatility_stress', description='Volatility Stress Index (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing 'vix_close' column")
            return pd.Series(np.nan, index=df.index)
        
        # VIX level component (0-100 scale)
        vix_level = (df['vix_close'].rolling(5).mean() - 10).clip(lower=0, upper=100)
        
        # VIX momentum component
        vix_mom = df['vix_close'].pct_change(5)
        momentum_stress = vix_mom.rolling(5).mean().fillna(0) * 20  # Scale to 0-20
        
        # Combine: Level + Momentum, capped at 100
        stress_index = vix_level + momentum_stress.abs()
        return stress_index.clip(lower=0, upper=100).fillna(50)
