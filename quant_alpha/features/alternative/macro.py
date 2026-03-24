"""
Macro-Economic Regime Factors
=============================
Quantitative indicators derived from primary macro-economic primitives:
Commodities (Oil), Currencies (USD), and Fixed Income (Treasury Yields).

Purpose
-------
This module constructs alternative alpha factors that capture the broader
economic environment. Unlike idiosyncratic equity factors, these signals
measure systematic risks and regime shifts (e.g., Reflation, Deflation,
Stagflation) that condition the Equity Risk Premium (ERP).

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry
    
    # Compute specific macro factor
    df_macro = FactorRegistry.compute("alt_macro_score", market_data_df)

Importance
----------
- **Regime Conditioning**: Equity correlations often converge to 1.0 during macro
  shocks. These factors allow models to detect and adapt to such regimes.
- **Discount Rate sensitivity**: `YieldTrend` directly proxies changes in the
  Risk-Free Rate ($R_f$), the denominator in all DCF valuations.
- **Input Diversification**: Introduces non-equity covariance structures into
  the feature set, improving ensemble model robustness.

Tools & Frameworks
------------------
- **Pandas**: Used for panel data grouping (`groupby`) and rolling window statistics.
- **NumPy**: efficient handling of numerical operations (log returns, z-scores).
- **FactorRegistry**: Decorator-based architecture for seamless pipeline integration.
"""

import pandas as pd
import numpy as np
from ..base import AlternativeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class OilMomentum(AlternativeFactor):
    """
    Oil Price Momentum: A proxy for global demand and cost-push inflation.

    Logic:
        Calculates the 21-day (approx. 1 month) rate of change, smoothed over
        5 days to dampen high-frequency noise common in commodity markets.

    Mathematical Formulation:
        $$ Signal_t = \\mu_{5}\\left( \\frac{P_{t}}{P_{t-21}} - 1 \\right) $$

    Interpretation:
        - **Positive**: Strong demand (Growth) or Supply Shock (Inflation).
        - **Negative**: Weak demand (Recession) or Supply Glut (Deflation).
    """
    def __init__(self):
        """Initializes continuous cross-asset fundamental velocity measurements safely."""
        super().__init__(name='alt_oil_momentum', description='Oil Price Momentum (21D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates strict localized execution boundaries scaling sequential difference reliably.
        
        Args:
            df (pd.DataFrame): Systemic target evaluation matrix encapsulating commodity derivatives.
            
        Returns:
            pd.Series: Vectorized sequences structurally defining momentum perfectly efficiently.
        """
        if 'oil_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # Evaluates 21-day continuous geometric change bounded safely by 5-day variance smoothing
        if 'ticker' in df.columns:
            momentum = df.groupby('ticker')['oil_close'].ffill().groupby(df['ticker']).pct_change(21) * 100
            return momentum.groupby(df['ticker']).transform(lambda x: x.rolling(window=5, min_periods=1).mean()).fillna(0)
        else:
            momentum = df['oil_close'].ffill().pct_change(21) * 100
            return momentum.rolling(window=5, min_periods=1).mean().fillna(0)

@FactorRegistry.register()
class USDStrength(AlternativeFactor):
    """
    USD Index Strength: Measure of currency purchasing power and global liquidity.

    Logic:
        Computes the percentage deviation of the current price from a 20-day
        moving average. This acts as a short-term mean-reversion signal or
        trend strength indicator depending on the magnitude.

    Importance:
        - **Export Headwind**: Strong USD hurts earnings of US multinationals.
        - **Risk-Off Proxy**: USD often strengthens during global liquidity crunches.
    """
    def __init__(self):
        """Initializes the spatial limits binding global statistical currency arrays."""
        super().__init__(name='alt_usd_strength', description='USD Index Strength (Normalized)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates geometric bounds isolating fractional deviation from local macro centroids.
        
        Args:
            df (pd.DataFrame): The base multi-asset execution matrix.
            
        Returns:
            pd.Series: Continuous sequence cleanly mapping spatial divergence precisely.
        """
        if 'usd_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 'ticker' in df.columns:
            ma_20 = df.groupby('ticker')['usd_close'].transform(lambda x: x.rolling(window=20, min_periods=5).mean())
        else:
            ma_20 = df['usd_close'].rolling(window=20, min_periods=5).mean()
        
        strength = ((df['usd_close'] - ma_20) / (ma_20.replace(0, np.nan) + 1e-6)) * 100
        return strength.fillna(0)

@FactorRegistry.register()
class YieldTrend(AlternativeFactor):
    """
    10Y Treasury Yield Trend: Structural changes in the Risk-Free Rate.

    Logic:
        Calculates the 63-day (Quarterly) momentum of the 10-year Treasury yield.

    Interpretation:
        - **Rising**: Increasing discount rates (headwind for Long-Duration Assets like Tech).
          Often implies economic heating or inflation.
        - **Falling**: Decreasing discount rates (tailwind for Growth).
          Often implies flight-to-safety or economic cooling.
    """
    def __init__(self):
        """Initializes dynamic interest-rate momentum mapping structures."""
        super().__init__(name='alt_yield_trend', description='10Y Yield Momentum (63D)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Extracts structurally bounded normal limit trends assessing quarterly rate shifts.
        
        Args:
            df (pd.DataFrame): Systemic raw historical execution matrices tracking benchmark bonds.
            
        Returns:
            pd.Series: Extracted quarterly momentum vectors explicitly correctly.
        """
        if 'us_10y_close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 'ticker' in df.columns:
            yield_momentum = df.groupby('ticker')['us_10y_close'].ffill().groupby(df['ticker']).pct_change(63) * 100
        else:
            yield_momentum = df['us_10y_close'].ffill().pct_change(63) * 100
            
        return yield_momentum.fillna(0)

@FactorRegistry.register()
class MacroEconomicScore(AlternativeFactor):
    """
    Macro Health Score (0-100): Composite multi-asset regime indicator.

    Formulation:
        $$ Score = 50 + 15 \\times \\text{Avg}(Z_{Oil} - Z_{USD} + Z_{Yields}) $$

    Economic Rationale ("Reflation Trade"):
        - **Rising Oil ($+Z_{Oil}$)**: Indicative of industrial demand/growth.
        - **Falling USD ($-Z_{USD}$)**: Indicative of abundant global liquidity.
        - **Rising Yields ($+Z_{Yields}$)**: Indicative of "Risk-On" rotation from Bonds to Equities.
    """
    def __init__(self):
        """Initializes multidimensional aggregation scoring macro velocity states safely."""
        super().__init__(name='alt_macro_score', description='Macroeconomic Health Score (0-100)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes standardized structural index mappings bridging independent macro vectors uniformly.
        
        Args:
            df (pd.DataFrame): Cross-asset matrix strictly containing energy, rates, and FX bases.
            
        Returns:
            pd.Series: Continuous bounded limits tracking composite index structures explicitly.
        """
        required = ['oil_close', 'usd_close', 'us_10y_close']
        if not all(col in df.columns for col in required):
            return pd.Series(50, index=df.index)
        
        def z_score(series, window):
            return (series - series.rolling(window, min_periods=21).mean()) / (series.rolling(window, min_periods=21).std() + 1e-6)
        
        if 'ticker' in df.columns:
            usd_z = df.groupby('ticker')['usd_close'].transform(lambda x: z_score(x, 252))
            yield_z = df.groupby('ticker')['us_10y_close'].transform(lambda x: z_score(x.ffill().pct_change(63), 252))
            oil_z = df.groupby('ticker')['oil_close'].transform(lambda x: z_score(x.ffill().pct_change(21), 252))
        else:
            usd_z = z_score(df['usd_close'], 252)
            yield_z = z_score(df['us_10y_close'].ffill().pct_change(63), 252)
            oil_z = z_score(df['oil_close'].ffill().pct_change(21), 252)

        composite = (oil_z - usd_z + yield_z) / 3
        
        score = 50 + (composite * 15)
        
        return score.clip(lower=0, upper=100).fillna(50)