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
        """Initializes the structural market regime classifier map."""
        super().__init__(name='comp_regime', description='Market Regime Classification')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates discrete boundaries cleanly matching market states accurately.

        Args:
            df (pd.DataFrame): Foundational execution bounds containing structural S&P 500 limits.

        Returns:
            pd.Series: Flawlessly tracked deviation states classifying standard market regimes.
        """
        if 'sp500_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Evaluates secular trajectory structures defining 200-day systemic institutional baselines
        ma200 = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200, min_periods=50).mean())
        price_above_ma = df['sp500_close'] > ma200
        
        # Captures localized 1-month cyclical momentum boundaries tracking immediate tactical flow
        returns_21d = df.groupby('ticker')['sp500_close'].pct_change(21)
        
        # Structurally maps intersection logic discretizing continuous metrics into ordinal states
        regime = pd.Series(0, index=df.index)
        regime[price_above_ma & (returns_21d > 0)] = 1
        regime[~price_above_ma & (returns_21d < 0)] = -1
        
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
        """Initializes the absolute localized volatility risk state classifier."""
        super().__init__(name='comp_vol_regime', description='Volatility Regime Levels')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates discrete statistical risk buckets natively bounding systemic implied variances.

        Args:
            df (pd.DataFrame): Evaluation dimensional limits safely tracking index volatility proxies.

        Returns:
            pd.Series: Categorical numerical mappings identifying explicit variance topologies.
        """
        if 'vix_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Extrapolates smoothed risk metrics evaluating moving boundaries cleanly avoiding intraday spikes
        vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        conditions = [
            (vix < 17), 
            (vix >= 17) & (vix < 28), 
            (vix >= 28) 
        ]
        choices = [0, 1, 2]
        
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
        """Initializes the cross-asset inflation/liquidity flow boundary tracker."""
        super().__init__(name='comp_capital_flow', description='Oil & USD Flow Indicator')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes aggregate directional consensus natively scaling broad systemic monetary velocities.

        Args:
            df (pd.DataFrame): Bounding evaluation parameters safely resolving global asset pairs.

        Returns:
            pd.Series: Normalized composite score bounded seamlessly evaluating current liquidity patterns.
        """
        if not {'oil_close', 'usd_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        # Synchronously derives localized moving boundaries extracting standardized tracking velocities
        oil_mom = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21)
        usd_mom = df.groupby('ticker')['usd_close'].ffill().groupby(df['ticker']).pct_change(21)
        
        combined = (oil_mom + usd_mom) / 2
        
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
        """Initializes the multidimensional economic trajectory baseline model."""
        super().__init__(name='comp_econ_momentum', description='Cross-Asset Macro Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates strictly mapped growth expectations directly aggregating core systematic matrices.

        Args:
            df (pd.DataFrame): Mapped evaluations successfully structurally tracked natively spanning yields, energy, FX.

        Returns:
            pd.Series: Unified probabilistic scalar boundary bounded strictly evaluating economic vitality.
        """
        req = ['us_10y_close', 'oil_close', 'usd_close']
        if not all(col in df.columns for col in req):
            return pd.Series(np.nan, index=df.index)
        
        # Identifies structural changes tracking dynamic macro conditions logically seamlessly.
        y_mom = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21)
        o_mom = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21)
        u_mom = df.groupby('ticker')['usd_close'].ffill().groupby(df['ticker']).pct_change(21)
        
        composite = (y_mom + o_mom - u_mom) / 3
        
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
        """Initializes the master portfolio health orchestration variable safely."""
        super().__init__(name='comp_health_index', description='Global Portfolio Health Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Executes comprehensive systemic structural matrices bounding explicitly optimal deployment limits.

        Args:
            df (pd.DataFrame): Target evaluation matrices correctly tracked systematically.

        Returns:
            pd.Series: Mathematical execution constraints tracking generalized optimal conditions.
        """
        required = ['sp500_close', 'vix_close', 'us_10y_close']
        if not all(c in df.columns for c in required):
            return pd.Series(np.nan, index=df.index)

        regime = self.compute_sub_score(df, 'regime') * 25
        vol = self.compute_sub_score(df, 'vol') * 25
        macro = self.compute_sub_score(df, 'macro') * 50
        
        health = regime + vol + macro
        
        return health.groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).clip(0, 100).fillna(np.nan)

    def compute_sub_score(self, df, type):
        """
        Normalizes specific categorical sub-components explicitly safely to strict evaluation bounds.

        Args:
            df (pd.DataFrame): Mapped evaluations successfully structurally tracked.
            type (str): Explicit mathematical routing condition boundary.

        Returns:
            pd.Series: Linearly standardized matrix outputs precisely bounding localized constraints.
        """
        if type == 'regime':
            ma = df.groupby('ticker')['sp500_close'].transform(lambda x: x.rolling(200).mean())
            return (df['sp500_close'] > ma).astype(float)
        elif type == 'vol':
            vix = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean())
            return ((35 - vix) / 20).clip(0, 1)
        elif type == 'macro':
            y_mom = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21)
            return ((y_mom + 0.02) / 0.04).clip(0, 1)