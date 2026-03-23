"""
Market Sentiment & Volatility Regime Factors
============================================
Quantitative indicators derived from implied volatility surfaces (VIX) and
broad market technical structure (S&P 500) to gauge investor sentiment.

Purpose
-------
This module constructs alternative alpha factors that serve as proxies for
market fear, complacency, and systemic stress. These signals are pivotal for
identifying "Risk-On" vs. "Risk-Off" (RORO) regimes, allowing portfolio
optimizers to dynamically adjust beta exposure or activate tail-risk hedges.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry

    # Compute volatility stress score
    df_sentiment = FactorRegistry.compute("alt_volatility_stress", market_data_df)

Importance
----------
- **Regime Filtering**: The `RiskOnOffSignal` acts as a coarse-grained filter,
  preventing long-only strategies from deploying capital during high-volatility
  downtrends (preservation of capital).
- **Convexity Detection**: `VIXMomentum` captures the velocity of fear, often
  signaling the onset of a liquidity crisis before price impacts fully materialize.
- **Tail Risk Management**: `VolatilityStressIndex` provides a composite scalar
  for sizing defensive positions.

Tools & Frameworks
------------------
- **Pandas**: Vectorized time-series operations (`pct_change`, `rolling`, `transform`).
- **NumPy**: Efficient numerical clipping and boolean logic handling.
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class VIXLevel(AlternativeFactor):
    """
    Normalized VIX Level: Absolute measure of implied volatility.

    Mathematical Formulation:
        $$ Score_t = \\text{clip}\\left( \\frac{VIX_t - 10}{30}, 0, 1 \\right) $$

    Interpretation:
        - **0.0**: VIX $\le$ 10 (Extreme Complacency).
        - **1.0**: VIX $\ge$ 40 (Extreme Panic / Capitulation).
    """
    def __init__(self):
        super().__init__(name='alt_vix_level', description='VIX Level (Fear Score)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Min-Max Normalization: Maps VIX range [10, 40] to [0, 1].
        # This provides a bounded feature suitable for ML models.
        vix_norm = ((df['vix_close'] - 10) / 30).clip(0, 1)
        if 'ticker' in df.columns:
            return vix_norm.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(np.nan)
        else:
            return vix_norm.rolling(5, min_periods=1).mean().fillna(np.nan)

@FactorRegistry.register()
class VIXMomentum(AlternativeFactor):
    """
    VIX Momentum: The velocity of fear.

    Logic:
        Calculates the 5-day percentage change in the VIX index.
        Rapid spikes in VIX ("Vol of Vol") often precede equity market drawdowns.
    """
    def __init__(self):
        super().__init__(name='alt_vix_momentum', description='VIX 5D Momentum')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # 5-day Rate of Change ($O(N)$).
        # Grouping ensures time-series integrity if dataframe contains multiple tickers.
        if 'ticker' in df.columns:
            mom = df.groupby('ticker')['vix_close'].pct_change(5) * 100
            return mom.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(np.nan)
        else:
            mom = df['vix_close'].pct_change(5) * 100
            return mom.rolling(5, min_periods=1).mean().fillna(np.nan)

@FactorRegistry.register()
class RiskOnOffSignal(AlternativeFactor):
    """
    Binary Regime Filter ("Risk-On" vs "Risk-Off").

    Logic:
        A "Risk-On" environment is defined by the intersection of:
        1. **Low Volatility**: VIX < 20.
        2. **Bullish Trend**: S&P 500 Price > 50-Day Moving Average.

    Output:
        - **1**: Favorable conditions for Beta exposure.
        - **0**: Defensive / Hedging regime.
    """
    def __init__(self):
        super().__init__(name='alt_risk_on_off', description='Binary Risk Sentiment')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'vix_close', 'sp500_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Condition 1: Volatility below psychological threshold
        vix_calm = df['vix_close'] < 20
        
        # Condition 2: Structural Uptrend (Price > 50D MA)
        if 'ticker' in df.columns:
            sp500_ma = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(50).mean())
        else:
            sp500_ma = df['sp500_close'].rolling(50).mean()
            
        market_uptrend = df['sp500_close'] > sp500_ma
        
        # Intersection of conditions
        signal = (vix_calm & market_uptrend).astype(int)
        return signal.fillna(np.nan)

@FactorRegistry.register()
class VolatilityStressIndex(AlternativeFactor):
    """
    Composite Stress Index (0-100).

    Aggregation Logic:
        - **Level Component (70 pts)**: Normalized VIX level. Captures static fear.
        - **Momentum Component (30 pts)**: Positive VIX velocity. Captures shock intensity.
    
    Interpretation:
        Higher values indicate acute market stress and potential liquidity gaps.
    """
    def __init__(self):
        super().__init__(name='alt_volatility_stress', description='Volatility Stress Index')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # 1. Level Score: 0 to 70 points
        vix_level_score = ((df['vix_close'] - 10) / 30).clip(0, 1) * 70
        
        # 2. Momentum Score: 0 to 30 points
        # Note: Only POSITIVE momentum contributes to stress (clipped at 0).
        if 'ticker' in df.columns:
            vix_mom = df.groupby('ticker')['vix_close'].pct_change(5).clip(0, 1) * 30
        else:
            vix_mom = df['vix_close'].pct_change(5).clip(0, 1) * 30
        
        stress = vix_level_score + vix_mom
        return stress.fillna(np.nan)