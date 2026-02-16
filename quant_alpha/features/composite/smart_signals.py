import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MomentumVIXDivergence(CompositeFactor):
    """
    Detects euphoria/capitulation. 
    Logic: (Stock Mom - VIX Mom). 
    High value = Clean rally. Low value = Risky/Hidden fear.
    """
    def __init__(self):
        super().__init__(name='comp_div_momentum_vix', description='Momentum-VIX Divergence Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # 1. Individual Stock Momentum
        mom = df.groupby('ticker')['close'].pct_change(21)
        
        # 2. VIX Momentum (Normalized to be comparable)
        vix_mom = df.groupby('ticker')['vix_close'].pct_change(21)
        
        # Divergence: If price goes up (+) but VIX also goes up (+), signal drops (Danger)
        # If price goes up (+) and VIX goes down (-), signal stays high (Strong)
        signal = mom - vix_mom
        
        # FIX: Group by ticker for rolling calculation
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)

@FactorRegistry.register()
class ValueYieldCombo(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_value_yield', description='Value-Yield Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns or 'pe_ratio' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Yield Component: Inverted (Lower rates = Higher multiple support)
        yield_smooth = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(21, min_periods=5).mean()).clip(0.1, 5)
        rate_score = 1 - (yield_smooth / 5)
        
        # Value Component: Cross-sectional rank of inverse P/E
        # replace(0) and clip to avoid inf
        inv_pe = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 200)
        # FIX: Group by date for cross-sectional rank
        value_rank = inv_pe.groupby(df['date']).rank(pct=True)
        
        return (value_rank + rate_score) / 2

@FactorRegistry.register()
class QualityInDownturn(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_quality_stress', description='Quality under VIX Stress')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for core quality columns or fallback
        roe = df.get('roe', pd.Series(0, index=df.index))
        debt = df.get('debt_to_equity', pd.Series(1, index=df.index))
        vix = df.get('vix_close', pd.Series(20, index=df.index))
        
        # Quality Proxy: High ROE, Low Debt
        quality = roe.clip(-0.5, 0.5) - (debt.clip(0, 5) * 0.1)
        
        # Stress Multiplier: Amplify if VIX is high (above its 63-day mean)
        vix_ma = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(63, min_periods=5).mean())
        stress_trigger = np.where(vix > vix_ma, 1.5, 1.0)
        
        return (quality * stress_trigger).fillna(0)

@FactorRegistry.register()
class EarningsMacroAlignment(CompositeFactor):
    """
    FIXED: Per-ticker correlation to ensure variation.
    Detects if a specific stock's earnings trend is in sync with macro yields.
    """
    def __init__(self):
        super().__init__(name='comp_earnings_macro', description='Earnings-Macro Alignment Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check if required columns exist
        # We prefer pe_ratio for daily correlation, fallback to earnings_growth
        target_col = 'pe_ratio' if 'pe_ratio' in df.columns else 'earnings_growth'
        
        if 'us_10y_close' not in df.columns or target_col not in df.columns:
            logger.warning(f"❌ {self.name}: Missing macro or earnings data")
            return pd.Series(0, index=df.index)
        
        # 1. Macro signal (Yields are the same for everyone, we take 21-day change)
        # Group by ticker to align indices correctly in the apply
        macro_momentum = df.groupby('ticker')['us_10y_close'].pct_change(21)
        
        # 2. Vectorized Grouped Correlation
        # Alignment = Correlation between stock valuation/earnings and macro yields
        # Using PE Ratio provides daily variation, preventing "identical values" from constant earnings data
        alignment = df.groupby('ticker', group_keys=False).apply(
            lambda x: x[target_col].rolling(63, min_periods=10).corr(macro_momentum.loc[x.index])
        )
        
        # 3. Final cleanup: replace NaNs with 0 (neutral)
        return alignment.fillna(0)

@FactorRegistry.register()
class MultiAssetOpportunity(CompositeFactor):
    def __init__(self):
        super().__init__(name='comp_multi_asset', description='Oil-Yield-USD Consensus')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        assets = ['oil_close', 'us_10y_close', 'usd_close']
        if not all(col in df.columns for col in assets):
            return pd.Series(50, index=df.index)
        
        # Get binary direction of each asset
        # 1 if price > 21-day MA, else -1
        consensus = 0
        for col in assets:
            ma = df.groupby('ticker')[col].transform(lambda x: x.rolling(21).mean())
            consensus += np.where(df[col] > ma, 1, -1)
        
        # Map -3 to +3 range into 0 to 100
        # +3 (all bullish) -> 100, -3 (all bearish) -> 0
        opp_score = ((consensus / 3) + 1) * 50
        return pd.Series(opp_score, index=df.index).groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).fillna(50)