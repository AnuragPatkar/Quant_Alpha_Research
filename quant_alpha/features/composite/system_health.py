"""
System Health & Regime Classification
=====================================
Composite indicators for detecting broad market regimes and systemic risk levels.

Purpose
-------
This module aggregates technical and macro-economic primitives into high-level
state classifiers. These factors answer binary or categorical questions about
the market environment (e.g., "Is the market in a Bull trend?", "Is Volatility
at crisis levels?"). They serve as the "Traffic Light" system for portfolio
allocation, enabling dynamic beta scaling and risk-off switching.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry

    # Compute portfolio health index (0-100)
    df_health = FactorRegistry.compute("comp_health_index", market_data_df)

Importance
----------
- **Tail Risk Mitigation**: `VolatilityRegime` and `PortfolioHealthIndex` allow
  strategies to deleverage *before* realized variance destroys capital.
- **Trend Filtering**: `MarketRegimeScore` acts as a regime filter, preventing
  mean-reversion strategies from fighting strong trends or momentum strategies
  from churning in sideways markets.
- **Macro-Awareness**: Incorporates non-equity signals (Yields, Oil, USD) to
  assess the durability of equity rallies.

Tools & Frameworks
------------------
- **Pandas**: Grouped rolling window operations for trend detection.
- **NumPy**: `np.select` for efficient categorical vectorization.
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MarketRegimeScore(CompositeFactor):
    """
    Trend-Based Regime Classifier.

    Logic:
        Classifies the market environment based on the intersection of
        Long-Term Structure (200-Day SMA) and Medium-Term Momentum (21-Day Return).

    Categories:
        - **Bull (+1)**: Price > 200 SMA $\land$ 21d Return > 0.
        - **Bear (-1)**: Price < 200 SMA $\land$ 21d Return < 0.
        - **Sideways (0)**: Conflicting signals (e.g., Mean Reversion regime).
    """
    def __init__(self):
        super().__init__(name='comp_regime', description='Market Regime Classification')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'sp500_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # 1. Structural Trend (Long-term) ($O(N)$)
        # The 200-day Simple Moving Average (SMA) is the standard institutional proxy for secular trend.
        ma200 = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200, min_periods=50).mean())
        price_above_ma = df['sp500_close'] > ma200
        
        # 2. Cyclical Momentum (Medium-term) ($O(N)$)
        # 1-month lookback captures the immediate tactical direction.
        returns_21d = df.groupby('ticker')['sp500_close'].pct_change(21)
        
        # Classification Logic
        regime = pd.Series(0, index=df.index)
        regime[price_above_ma & (returns_21d > 0)] = 1
        regime[~price_above_ma & (returns_21d < 0)] = -1
        
        # Forward fill to ensure continuity between data points
        return regime.ffill().fillna(np.nan)

@FactorRegistry.register()
class VolatilityRegime(CompositeFactor):
    """
    Categorical Volatility Score.

    Purpose:
        Discretizes the continuous VIX index into actionable risk buckets based
        on historical distribution thresholds.

    Regimes:
        - **0 (Normal)**: VIX < 17. Favorable for leverage and carry strategies.
        - **1 (Elevated)**: 17 $\le$ VIX < 28. Increased hedging required.
        - **2 (Crisis)**: VIX $\ge$ 28. Capital preservation mode (Risk-Off).
    """
    def __init__(self):
        super().__init__(name='comp_vol_regime', description='Volatility Regime Levels')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'vix_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Smoothing: 5-day rolling mean ($O(N)$) to filter out intraday flash crashes.
        vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        # Vectorized Conditional Selection ($O(N)$)
        conditions = [
            (vix < 17), # Normal/Quiet
            (vix >= 17) & (vix < 28), # Elevated
            (vix >= 28) # Crisis
        ]
        choices = [0, 1, 2]
        
        # Default to 1 (Elevated) if undefined
        return pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)

@FactorRegistry.register()
class CapitalFlowSignal(CompositeFactor):
    """
    Cross-Asset Capital Flow Indicator.
    
    Logic:
        Measures the consensus momentum between Commodities (Oil) and Currency (USD).
        These assets often drive global liquidity cycles.
        
    Interpretation:
        - **High Score**: Strong flows into real assets and USD (Inflationary/Tightening).
        - **Low Score**: Outflows (Deflationary/Liquidity Injection).
    """
    def __init__(self):
        super().__init__(name='comp_capital_flow', description='Oil & USD Flow Indicator')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'usd_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        # Calculate 21-day momentum for asset classes
        # FutureWarning fix: Use ffill() before pct_change() to handle NaN values properly.
        oil_mom = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21)
        usd_mom = df.groupby('ticker')['usd_close'].ffill().groupby(df['ticker']).pct_change(21)
        
        # Composite Signal: Average Momentum
        combined = (oil_mom + usd_mom) / 2
        
        # Normalization: Scale roughly [-5%, +5%] range to [0, 100]
        flow_score = ((combined + 0.05) / 0.10).clip(0, 1) * 100
        
        if 'ticker' in df.columns:
            return flow_score.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(np.nan)
        return flow_score.rolling(5, min_periods=1).mean().fillna(np.nan)

@FactorRegistry.register()
class EconomicMomentumScore(CompositeFactor):
    """
    Macro-Economic Strength Composite.
    
    Logic:
        Aggregates price signals from Bond Yields, Oil, and USD to proxy
        real-time economic growth expectations.
        
    Formula:
        $$ Signal = \frac{\Delta Yields + \Delta Oil - \Delta USD}{3} $$
    """
    def __init__(self):
        super().__init__(name='comp_econ_momentum', description='Cross-Asset Macro Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        req = ['us_10y_close', 'oil_close', 'usd_close']
        if not all(col in df.columns for col in req):
            return pd.Series(np.nan, index=df.index)
        
        # Component Logic:
        # 1. Rising Yields -> Growth expectations or Inflation (Positive)
        # 2. Rising Oil -> Demand growth (Positive)
        # 3. Rising USD -> Global liquidity tightening (Negative headwind)
        # FutureWarning fix: Use ffill() before pct_change() to handle NaN values properly.
        y_mom = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21)
        o_mom = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21)
        u_mom = df.groupby('ticker')['usd_close'].ffill().groupby(df['ticker']).pct_change(21)
        
        # Equal-weighted composite
        composite = (y_mom + o_mom - u_mom) / 3
        
        # Min-Max Scaling to 0-100 range (assuming +/- 4% monthly moves as bounds)
        score = ((composite + 0.04) / 0.08).clip(0, 1).fillna(np.nan) * 100
        if 'ticker' in df.columns:
            return score.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())
        return score.rolling(5, min_periods=1).mean().fillna(np.nan)

@FactorRegistry.register()
class PortfolioHealthIndex(CompositeFactor):
    """
    Global Portfolio Health Score (0-100).
    
    Purpose:
        The primary "Risk-On / Risk-Off" (RORO) master switch.
        Aggregates Technicals, Volatility, and Macro into a single scalar.
        
    Weights:
        - **Macro (50%)**: Economic Momentum.
        - **Regime (25%)**: S&P 500 Trend.
        - **Volatility (25%)**: VIX Level.
    """
    def __init__(self):
        super().__init__(name='comp_health_index', description='Global Portfolio Health Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Validation: Check for required columns to avoid KeyErrors
        required = ['sp500_close', 'vix_close', 'us_10y_close']
        if not all(c in df.columns for c in required):
            return pd.Series(np.nan, index=df.index)

        # Component Calculation (Sum of Parts)
        regime = self.compute_sub_score(df, 'regime') * 25  # Max 25 pts
        vol = self.compute_sub_score(df, 'vol') * 25     # Max 25 pts
        macro = self.compute_sub_score(df, 'macro') * 50   # Max 50 pts
        
        health = regime + vol + macro
        
        # Smoothing: 5-day rolling average to prevent signal flicker
        return health.groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).clip(0, 100).fillna(np.nan)

    def compute_sub_score(self, df, type):
        """Helper to normalize sub-components to [0, 1] scale."""
        if type == 'regime':
            # Trend: 1.0 if > 200 SMA, else 0.0
            ma = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200).mean())
            return (df['sp500_close'] > ma).astype(float)
        elif type == 'vol':
            # Volatility: Inverted Scale.
            # VIX <= 15 -> 1.0 (Good)
            # VIX >= 35 -> 0.0 (Bad)
            vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean())
            return ((35 - vix) / 20).clip(0, 1)
        elif type == 'macro':
            # Macro: 10Y Yield Momentum (Proxy for growth expectations)
            # Normalized assuming +/- 2% monthly change range
            # FutureWarning fix: Use ffill() before pct_change() to handle NaN values properly.
            y_mom = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21)
            return ((y_mom + 0.02) / 0.04).clip(0, 1)