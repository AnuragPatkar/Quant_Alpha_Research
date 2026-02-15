"""
Smart Signals Composite Factors (5 Factors)
Focus: Complex cross-asset combinations, convergence/divergence signals, sophisticated blends.

Factors (Ranked by Importance):
1. MomentumVIXDivergence -> Detect euphoria or capitulation
2. ValueYieldCombo -> Value + rates + sentiment blend
3. QualityInDownturn -> Quality performance during stress
4. EarningsMacroAlignment -> Earnings expectations vs macro
5. MultiAssetOpportunity -> Cross-asset opportunity scoring
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class MomentumVIXDivergence(CompositeFactor):
    """
    Momentum-VIX Divergence: Detect market extremes (euphoria or capitulation).
    Formula: Momentum vs VIX trend ratio (positive divergence = warning signal)
    
    Why: When momentum rising but VIX rising = Fear despite strength (warning).
    When momentum falling but VIX falling = Relief despite weakness (bounce setup).
    Identifies unsustainable moves and potential reversals.
    """
    def __init__(self):
        super().__init__(name='comp_div_momentum_vix', description='Momentum-VIX Divergence Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['sp500_close', 'vix_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing SP500 or VIX data")
            return pd.Series(np.nan, index=df.index)
        
        # Momentum direction (positive trend = 1, negative = -1)
        returns = df['sp500_close'].pct_change(21)
        mom_trend = (returns > 0).astype(float) * 2 - 1
        
        # VIX trend (falling VIX = 1, rising = -1)
        vix_change = df['vix_close'].pct_change(21)
        vix_trend = (vix_change < 0).astype(float) * 2 - 1
        
        # Divergence signal: 1 if aligned, -1 if diverged
        divergence = (mom_trend * vix_trend)
        
        # Smooth and normalize
        divergence_smooth = divergence.rolling(21).mean()
        return divergence_smooth.fillna(0)


@FactorRegistry.register()
class ValueYieldCombo(CompositeFactor):
    """
    Value-Yield Combo: Blend value signals with rate environment.
    Formula: (Value_score + (1 - Yield_level/5) ) / 2 normalized
    
    Why: Value works better in low-rate environments (higher multiples expansion).
    In high-rate environment, value still attracts but with lower multiples.
    Combines fundamentals (value) with macro context (yields).
    """
    def __init__(self):
        super().__init__(name='comp_value_yield', description='Value-Yield Combination Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing yield data")
            return pd.Series(np.nan, index=df.index)
        
        # Rate environment component (lower rates = better for value multiple expansion)
        yield_level = df['us_10y_close'].rolling(21).mean().clip(lower=0, upper=5)
        rate_score = 1 - (yield_level / 5)
        
        # Value Proxy: Inverse P/E if available
        if 'pe_ratio' in df.columns:
            value_norm = (-df['pe_ratio']).rolling(63).rank(pct=True)
        else:
            # Neutral if missing
            value_norm = pd.Series(0.5, index=df.index)
        
        # Combine
        combo = (value_norm + rate_score) / 2
        return combo.fillna(0.5)


@FactorRegistry.register()
class QualityInDownturn(CompositeFactor):
    """
    Quality in Downturn: Quality factor performance during high-stress periods.
    Formula: Quality_signal * (if VIX > 20 then 1.5 else 1.0)
    
    Why: Quality (stable earnings, low debt) outperforms during crashes.
    This factor amplifies quality exposure when VIX elevated.
    Anti-cyclical positioning for downside protection.
    """
    def __init__(self):
        super().__init__(name='comp_quality_stress', description='Quality in Downturn Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing VIX data")
            return pd.Series(np.nan, index=df.index)
        
        # Stress indicator
        vix = df['vix_close'].rolling(5).mean()
        stress_multiplier = (vix > 20).astype(float) * 0.5 + 1.0
        
        # Quality proxy: Low volatility of returns (inverse of price volatility)
        returns_vol = df['sp500_close'].pct_change().rolling(63).std()
        quality_signal = 1 / (returns_vol + 0.01)  # Inverse volatility
        quality_signal = (quality_signal - quality_signal.mean()) / quality_signal.std()
        
        # Amplify quality during stress
        quality_amplified = quality_signal * stress_multiplier
        
        return quality_amplified.fillna(0)


@FactorRegistry.register()
class EarningsMacroAlignment(CompositeFactor):
    """
    Earnings-Macro Alignment: Match earnings expectations with macro reality.
    Formula: (Earnings_growth_trend + Macro_growth_trend) / 2
    
    Why: When earnings growth aligns with macro growth = Sustainable (stay long).
    When earnings exceed macro = Valuation disconnect (watch for cut).
    When macro exceeds earnings = Economic growth not monetized (opportunity).
    """
    def __init__(self):
        super().__init__(name='comp_earnings_macro', description='Earnings-Macro Alignment Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns or 'oil_close' not in df.columns:
            logger.warning(f"❌ {self.name}: Missing macro data")
            return pd.Series(np.nan, index=df.index)
        
        # Macro growth proxy: Yield momentum (rising = growth expectations up)
        macro_growth = df['us_10y_close'].pct_change(21)
        macro_growth_norm = (macro_growth.rolling(63).mean() + 0.05) / 0.10
        
        # Earnings Growth
        if 'earnings_growth' in df.columns:
            earnings_growth = df['earnings_growth']
        else:
            earnings_growth = pd.Series(0, index=df.index)
        earnings_growth_norm = (earnings_growth.rolling(63).mean() + 0.05) / 0.10
        
        # Alignment: How correlated are they?
        # Create alignment score (1 = perfectly aligned, -1 = diverged)
        rolling_corr = earnings_growth.rolling(63).corr(macro_growth)
        
        # Weighted alignment: Use both correlation and absolute levels
        alignment = (rolling_corr + (macro_growth_norm + earnings_growth_norm) / 2) / 2
        
        return alignment.fillna(0)


@FactorRegistry.register()
class MultiAssetOpportunity(CompositeFactor):
    """
    Multi-Asset Opportunity: Cross-asset opportunity scoring (bonds, equities, commodities).
    Formula: (Oil_opportunity + Yield_opportunity + Currency_opportunity) / 3
    
    Why: When all assets agree (Oil, Rates, USD) = High conviction trade.
    When assets diverge = Transition/rotation period (lower conviction).
    Measures cross-asset momentum consensus.
    """
    def __init__(self):
        super().__init__(name='comp_multi_asset', description='Multi-Asset Opportunity Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        required = ['oil_close', 'us_10y_close', 'usd_close']
        if not all(col in df.columns for col in required):
            logger.warning(f"❌ {self.name}: Missing multi-asset data")
            return pd.Series(np.nan, index=df.index)
        
        # Oil opportunity (commodity momentum)
        oil_mom = df['oil_close'].pct_change(21)
        oil_opp = (oil_mom > 0).astype(float) * 2 - 1
        
        # Yield opportunity (growth momentum)
        yield_mom = df['us_10y_close'].pct_change(21)
        yield_opp = (yield_mom > 0).astype(float) * 2 - 1
        
        # Currency opportunity (USD strength)
        usd_mom = df['usd_close'].pct_change(21)
        usd_opp = (usd_mom > 0).astype(float) * 2 - 1
        
        # Consensus score (smooth over 21 days)
        consensus = (oil_opp + yield_opp + usd_opp) / 3
        consensus_smooth = consensus.rolling(21).mean()
        
        # Convert to 0-100 opportunity score
        opp_score = ((consensus_smooth + 1) / 2) * 100
        
        return opp_score.fillna(50)
