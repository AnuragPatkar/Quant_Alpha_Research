"""
Inflation & Macro-Regime Factors
================================
Quantitative indicators for tracking inflation expectations, cost-push pressures,
and macro-economic regimes (Goldilocks vs. Stagflation).

Purpose
-------
This module constructs alternative alpha factors derived from cross-asset relationships
between Commodities (Oil), Fixed Income (10Y Treasury Yields), and Currencies (USD Index).
These factors serve as critical inputs for regime-switching models and risk management
overlays, helping to identify environments where traditional equity correlations break down.

Usage
-----
These factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry
    import quant_alpha.features.alternative.inflation  # Triggers registration

    # Compute specific inflation factor
    df_signals = FactorRegistry.compute("alt_oil_usd_ratio", market_data_df)

Importance
----------
- **Regime Identification**: Inflation is a primary driver of asset class correlation shifts.
  Detecting transitions (e.g., Low $\to$ High Inflation) allows for dynamic beta adjustment.
- **Cross-Asset Signal**: Incorporating non-equity data (Rates, FX, Commodities) reduces
  endogeneity issues common in pure price-volume equity factors.
- **Risk Mitigation**: High values in `InflationProxyScore` often precede volatility
  spikes in duration-sensitive sectors (Technology, Utilities).

Tools & Frameworks
------------------
- **Pandas**: Efficient $O(N)$ time-series rolling window operations and `groupby` transformations.
- **NumPy**: Vectorized numerical stability handling (clipping, NaN replacement).
- **FactorRegistry**: Decorator-based architecture for seamless pipeline integration.
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class OilUSDRatio(AlternativeFactor):
    """
    Oil-USD Ratio: Relative strength of Energy costs vs. Currency purchasing power.

    Mathematical Formulation:
        $$ Signal_t = \\frac{Ratio_t}{\\mu_{252}(Ratio)} - 1 $$
        Where $Ratio_t = \\frac{P_{Oil}}{P_{USD}}$.

    Interpretation:
        - **Positive**: Rising real energy costs (Inflationary Headwind).
        - **Negative**: Strengthening currency or falling energy costs (Deflationary Tailwind).
    """
    def __init__(self):
        """Initializes continuous cross-asset limits modeling purchasing power dynamically."""
        super().__init__(name='alt_oil_usd_ratio', description='Oil-USD Ratio Inflation Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts absolute statistical boundary representations strictly evaluating real energy burdens.
        
        Args:
            df (pd.DataFrame): Data matrix housing structural commodity and currency limits cleanly.
            
        Returns:
            pd.Series: Normalized sequences bounding localized deviations smoothly efficiently.
        """
        if not {'oil_close', 'usd_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Calculates real energy cost vectors adjusting for local currency purchasing power explicitly
        ratio = df['oil_close'] / df['usd_close'].replace(0, np.nan)
        
        if 'ticker' in df.columns:
            ratio_ma = ratio.groupby(df['ticker']).transform(lambda x: x.rolling(window=252, min_periods=63).mean())
            signal = (ratio / ratio_ma - 1) * 100
            return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        else:
            ratio_ma = ratio.rolling(window=252, min_periods=63).mean()
            signal = (ratio / ratio_ma - 1) * 100
            return signal.rolling(5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class YieldMomentum(AlternativeFactor):
    """
    Yield Momentum: Second derivative (velocity) of interest rate changes.

    Importance:
        Rapid changes in the risk-free rate (denominator in DCF models) cause
        violent re-ratings in long-duration growth equities.
    """
    def __init__(self):
        """Initializes parameters safely analyzing second-order derivative trajectories securely."""
        super().__init__(name='alt_yield_momentum', description='Yield Momentum (Growth Expectations)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes localized temporal dependencies strictly mapping treasury yield shifts explicitly.
        
        Args:
            df (pd.DataFrame): Evaluation dimensional limits exactly mapped strictly.
            
        Returns:
            pd.Series: Vector parameters extracting exact macroeconomic expectation forces smoothly.
        """
        if 'us_10y_close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        if 'ticker' in df.columns:
            momentum = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21) * 100
            return momentum.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        else:
            momentum = df['us_10y_close'].ffill().pct_change(21) * 100
            return momentum.rolling(5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class InflationProxyScore(AlternativeFactor):
    """
    Composite index blending realized cost-push inflation (Oil) and 
    market-implied expectations (Yields).

    Logic:
        - **Oil Component (60%)**: 21-day momentum, representing acute supply shocks.
        - **Yield Component (40%)**: Range position (Min-Max scaling) over 252 days, 
          representing structural rate resets.
    
    Scale: 0-100 (50 = Neutral)
    """
    def __init__(self):
        """Initializes composite weighting limits mathematically indexing inflation proxy variables safely."""
        super().__init__(name='alt_inflation_proxy', description='Inflation Proxy (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Formulates weighted evaluation boundaries standardizing multi-variable inflation matrices natively.
        
        Args:
            df (pd.DataFrame): Structural evaluating matrices mapping explicit boundaries efficiently.
            
        Returns:
            pd.Series: Computed cleanly explicitly mapping scaled indices precisely securely.
        """
        if not {'oil_close', 'us_10y_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        if 'ticker' in df.columns:
            oil_mom = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21).groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())
            y_min = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(252, min_periods=63).min())
            y_max = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(252, min_periods=63).max())
        else:
            oil_mom = df['oil_close'].ffill().pct_change(21).rolling(5, min_periods=1).mean()
            y_min = df['us_10y_close'].rolling(252, min_periods=63).min()
            y_max = df['us_10y_close'].rolling(252, min_periods=63).max()
            
        oil_score = (oil_mom * 100).clip(-50, 50) + 50 
        
        yield_score = ((df['us_10y_close'] - y_min) / (y_max - y_min + 1e-6)) * 100
        
        combined = (oil_score * 0.6 + yield_score * 0.4).clip(0, 100)
        return combined.fillna(np.nan)

@FactorRegistry.register()
class GrowthInflationMix(AlternativeFactor):
    """
    Statistical spread (Z-Score) between Nominal Growth expectations (Yields) 
    and Real Inflation inputs (Oil).

    Interpretation:
        - **Positive Z-Score (> 0)**: 'Goldilocks'. Yields rising faster than oil, 
          implying Real Rate expansion driven by economic growth.
        - **Negative Z-Score (< 0)**: 'Stagflation'. Oil rising faster than yields, 
          implying margin compression and stubborn inflation.
    """
    def __init__(self):
        """Initializes continuous structural spread normalization evaluating core correlations."""
        super().__init__(name='alt_growth_inflation_mix', description='Growth vs Inflation Balance')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates standardized statistical boundaries comparing independent rate-of-change maps structurally.
        
        Args:
            df (pd.DataFrame): Target evaluation sequences mapping fundamental market assumptions natively.
            
        Returns:
            pd.Series: Evaluated parameter representations precisely mapped cleanly securely correctly.
        """
        if not {'us_10y_close', 'oil_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        if 'ticker' in df.columns:
            growth_sig = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(21)
            infl_sig = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21)
            spread = growth_sig - infl_sig
            spread_mean = spread.groupby(df['ticker']).transform(lambda x: x.rolling(63, min_periods=21).mean())
            spread_std = spread.groupby(df['ticker']).transform(lambda x: x.rolling(63, min_periods=21).std())
        else:
            growth_sig = df['us_10y_close'].ffill().pct_change(21, fill_method=None)
            infl_sig = df['oil_close'].ffill().pct_change(21, fill_method=None)
            spread = growth_sig - infl_sig
            spread_mean = spread.rolling(63, min_periods=21).mean()
            spread_std = spread.rolling(63, min_periods=21).std()
        
        z_mix = (spread - spread_mean) / (spread_std + 1e-6)
        
        if 'ticker' in df.columns:
            return z_mix.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).clip(-3, 3).fillna(np.nan)
        return z_mix.rolling(5, min_periods=1).mean().clip(-3, 3).fillna(np.nan)