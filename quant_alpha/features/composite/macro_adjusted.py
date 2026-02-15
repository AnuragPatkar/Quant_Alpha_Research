"""
Macro-Adjusted Composite Factors (5 Factors)
Focus: Blending technical signals with macro context for better regime adaptation.

Factors (Ranked by Importance):
1. MacroAdjustedMomentum -> Momentum modulated by volatility regime
2. OilCorrectedValue -> Value signals filtered by commodity environment
3. RateEnvironmentScore -> Quality performance in current rate regime
4. DollarAdjustedGrowth -> Growth adjusted for currency strength
5. RiskParityBlend -> Equal weight technical momentum + sentiment balance
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class MacroAdjustedMomentum(CompositeFactor):
    """
    Macro-Adjusted Momentum: Price momentum scaled by volatility regime.
    Formula: Momentum * (1 / VIX_level) * 100
    
    Why: High VIX suppresses momentum playability (noise too high).
    Low VIX amplifies momentum edge (clean signals).
    Adjusts risk exposure to market regime automatically.
    """
    def __init__(self):
        super().__init__(name='comp_macro_momentum', description='Macro-Adjusted Momentum Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['sp500_close', 'vix_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing SP500 or VIX data")
            return pd.Series(np.nan, index=df.index)
        
        # Base momentum (21-day)
        momentum = df['sp500_close'].pct_change(21) * 100
        
        # VIX adjustment factor (1/VIX normalized)
        vix_normalized = df['vix_close'].rolling(5).mean().clip(lower=10, upper=40)
        adjustment = 25 / vix_normalized  # Higher VIX = lower adjustment
        
        # Adjusted momentum
        adjusted = momentum * adjustment
        return adjusted.fillna(0)


@FactorRegistry.register()
class OilCorrectedValue(CompositeFactor):
    """
    Oil-Corrected Value: Value signals adjusted for oil/inflation environment.
    Formula: Value_signal * (1 if Oil < 50th percentile else 0.5)
    
    Why: Value performs better in low-inflation environments (oil supply ample).
    In high-oil regime, value gets hurt by inflation expectations.
    Auto-weights value exposure to commodity environment.
    """
    def __init__(self):
        super().__init__(name='comp_oil_value', description='Oil-Corrected Value Factor')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'oil_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing oil data")
            return pd.Series(np.nan, index=df.index)
        
        # Oil regime: 1 if below median, 0.5 if above
        oil_median = df['oil_close'].rolling(252).median()
        oil_regime = (df['oil_close'] < oil_median).astype(float) * 0.5 + 0.5
        
        # Value Proxy: Inverse P/E if available, else 0
        if 'pe_ratio' in df.columns:
            value_signal = -df['pe_ratio'].clip(0, 100)
        else:
            value_signal = pd.Series(0, index=df.index)
        
        # Apply oil correction
        corrected = value_signal * oil_regime
        return pd.Series(corrected, index=df.index).fillna(0)


@FactorRegistry.register()
class RateEnvironmentScore(CompositeFactor):
    """
    Rate Environment Score: Quality factor performance in current rate regime.
    Formula: (1 - Yield_level/5) * QualityScore (inverted yield as quality multiple)
    
    Why: Lower rates favor durational/quality stocks (low growth, high quality).
    Higher rates favor value/cyclical (higher discount).
    Adapts quality weight to rate environment.
    """
    def __init__(self):
        super().__init__(name='comp_rate_quality', description='Rate-Adjusted Quality Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing yield data")
            return pd.Series(np.nan, index=df.index)
        
        # Rate regime weight for quality (lower rates = more weight to quality)
        rate_weight = 1 - (df['us_10y_close'].rolling(21).mean().clip(lower=0, upper=5) / 5)
        
        # Quality Proxy: ROE if available
        if 'roe' in df.columns:
            quality_signal = df['roe'].clip(-1, 1)
        else:
            quality_signal = pd.Series(0, index=df.index)
        
        # Rate-adjusted quality
        adjusted = quality_signal * rate_weight
        return pd.Series(adjusted, index=df.index).fillna(0)


@FactorRegistry.register()
class DollarAdjustedGrowth(CompositeFactor):
    """
    Dollar-Adjusted Growth: Growth factors modulated by USD strength.
    Formula: GrowthFactor * (1 if USD < 50th percentile else 0.7)
    
    Why: Strong dollar headwind for growth stocks (earnings translation loss).
    Weak dollar tailwind for growth (more comfortable valuation).
    Adjusts growth exposure to currency regime.
    """
    def __init__(self):
        super().__init__(name='comp_dollar_growth', description='Dollar-Adjusted Growth Factor')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'usd_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing USD data")
            return pd.Series(np.nan, index=df.index)
        
        # USD regime: Strong = lower growth weight
        usd_median = df['usd_close'].rolling(252).median()
        usd_regime = (df['usd_close'] < usd_median).astype(float) * 0.3 + 0.7
        
        # Growth Proxy: Earnings Growth
        if 'earnings_growth' in df.columns:
            growth_signal = df['earnings_growth'].clip(-1, 1)
        else:
            growth_signal = pd.Series(0, index=df.index)
        
        # Dollar-adjusted growth
        adjusted = growth_signal * usd_regime
        return pd.Series(adjusted, index=df.index).fillna(0)


@FactorRegistry.register()
class RiskParityBlend(CompositeFactor):
    """
    Risk Parity Blend: Equal risk weight between momentum + sentiment.
    Formula: (Momentum_normalized + RiskOnOff_normalized) / 2
    
    Why: Combines bull-market signals (momentum) with risk-off signals (VIX).
    Reduces single-factor regime risk.
    Natural diversification across signal types.
    """
    def __init__(self):
        super().__init__(name='comp_risk_parity', description='Risk Parity Momentum+Sentiment Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['sp500_close', 'vix_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing SP500 or VIX data")
            return pd.Series(np.nan, index=df.index)
        
        # Momentum component (normalized 0-1)
        momentum = df['sp500_close'].pct_change(21)
        momentum_norm = (momentum - momentum.rolling(63).mean()) / momentum.rolling(63).std()
        momentum_norm = (momentum_norm + 3) / 6  # Clip to ~0-1 range
        
        # Sentiment component (VIX inverted - lower VIX = higher sentiment)
        vix_norm = 1 - ((df['vix_close'] - 10) / 30).clip(lower=0, upper=1)
        
        # Equal weight blend
        blend = (momentum_norm + vix_norm) / 2
        return blend.fillna(0.5)
