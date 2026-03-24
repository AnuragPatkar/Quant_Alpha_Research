"""
Macro-Adjusted Composite Factors
================================
Alpha factors conditioned on systematic macro-economic variables.

Purpose
-------
This module constructs composite signals by modulating idiosyncratic stock factors
(Momentum, Value, Growth, Quality) with systematic regime indicators (VIX, Oil, 
Rates, USD). The goal is to adapt factor exposure dynamically to the prevailing 
economic environment.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry

    # Compute momentum adjusted for volatility regimes
    df_adj_mom = FactorRegistry.compute("comp_macro_momentum", market_data_df)

Importance
----------
- **Regime Adaptation**: Traditional factors (e.g., pure Momentum) often suffer 
  drawdowns during regime shifts. Macro-adjustments act as a continuous 
  "regime switch," scaling exposure down during unfavorable conditions.
- **Alpha Preservation**: Filters out "Value Traps" or "Growth Traps" caused by 
  external macro headwinds (e.g., high oil prices compressing margins).
- **Multicollinearity Reduction**: Introduces orthogonality to standard factors, 
  improving the conditioning of the covariance matrix in portfolio optimization.

Tools & Frameworks
------------------
- **Pandas**: Used for grouped transforms and rolling window statistics ($O(N)$).
- **NumPy**: Vectorized conditional logic (`np.where`) and numerical clipping.
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MacroAdjustedMomentum(CompositeFactor):
    """
    Volatility-Scaled Momentum.
    
    Logic:
        Modulates the 21-day momentum signal by the inverse of the market volatility (VIX).
        
    Mathematical Formulation:
        $$ S_t = Mom_{21d} \times \frac{25}{\overline{VIX}_{5d}} $$
        
    Interpretation:
        Scales down positions during high-volatility regimes to maintain constant 
        risk contribution (Vol-Targeting).
    """
    def __init__(self):
        """Initializes the structural volatility-adjusted momentum metric."""
        super().__init__(name='comp_macro_momentum', description='Momentum modulated by VIX')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the volatility-scaled momentum signal continuously mapped per ticker.

        Args:
            df (pd.DataFrame): Systemic target evaluation matrix containing 'close' and 'vix_close'.

        Returns:
            pd.Series: Evaluated parameter bounds representing inverse-volatility adjusted momentum.
        """
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Isolates idiosyncratic 21-day continuous asset return velocity independently
        momentum = df.groupby('ticker')['close'].pct_change(21) * 100
        
        # Broadcasts and limits market-wide structural volatility to dampen risk scaling
        vix_smooth = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean()).clip(10, 40)
        adjustment = 25 / vix_smooth
        
        signal = momentum * adjustment
        # Attenuates high-frequency noise evaluating local moving averages smoothly
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())

@FactorRegistry.register()
class OilCorrectedValue(CompositeFactor):
    """
    Energy-Conditional Value Factor.
    
    Logic:
        Adjusts the Value signal (Inverse P/E) based on the crude oil price regime.
        High oil prices often act as a tax on consumption and input costs, turning 
        low P/E stocks into "Value Traps."
        
    Mechanism:
        - **Low Oil ($< \mu_{252}$)**: Full weight (1.0) to Value signal.
        - **High Oil ($> \mu_{252}$)**: Half weight (0.5), dampening exposure.
    """
    def __init__(self):
        """Initializes the commodity-conditional valuation boundary filter."""
        super().__init__(name='comp_oil_value', description='Value signals filtered by oil environment')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the value factor limits conditioned on prevailing spot energy prices.

        Args:
            df (pd.DataFrame): Data matrix housing fundamental valuation and oil proxies.

        Returns:
            pd.Series: Normalized cross-sectional value signal strictly dampened during high-energy cost regimes.
        """
        if not {'oil_close', 'pe_ratio'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Extracts energy regime states assessing spot prices against the structural 1-Year Median
        oil_median = df.groupby('ticker')['oil_close'].transform(lambda x: x.rolling(252).median())
        oil_regime = np.where(df['oil_close'] < oil_median, 1.0, 0.5)
        
        # Resolves inverse baseline earnings yields strictly clipped limiting mathematical outliers
        value_signal = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 100)
        
        signal = value_signal * oil_regime
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())

@FactorRegistry.register()
class RateEnvironmentScore(CompositeFactor):
    """
    Duration-Adjusted Quality Score.
    
    Logic:
        Penalizes Quality (ROE) exposure as interest rates rise.
        High ROE stocks often behave as "Long Duration" assets; their present value
        is highly sensitive to the discount rate (10Y Yield).
        
    Formula:
        $$ Signal = ROE \times \left( 1 - \frac{Yield_{10y}}{5\%} \right) $$
    """
    def __init__(self):
        """Initializes the interest-rate penalized quality score."""
        super().__init__(name='comp_rate_quality', description='Quality adjusted by interest rates')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates structural quality limits proportionally discounted by benchmark yields.

        Args:
            df (pd.DataFrame): Frame containing profitability metrics and 10Y treasury proxies.

        Returns:
            pd.Series: Continuous limits tracking duration-adjusted quality.
        """
        roe_col = 'qual_roe' if 'qual_roe' in df.columns else 'roe'
        if not {'us_10y_close', roe_col}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Applies strict linear decay penalty mapped identically against rising nominal yield bounds
        rate_weight = 1 - (df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(21).mean()).clip(0, 5) / 5)
        
        # Propagates the fundamental corporate efficiency variable continuously
        quality_signal = df.groupby('ticker')[roe_col].ffill().clip(-1, 1)
        
        signal = quality_signal * rate_weight
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())

@FactorRegistry.register()
class DollarAdjustedGrowth(CompositeFactor):
    """
    FX-Conditional Growth Factor.
    
    Logic:
        Filters Earnings Growth signals based on USD strength.
        A strong Dollar ($DXY > \mu_{252}$) is a headwind for US multinationals,
        making past earnings growth less predictive of future performance.
    """
    def __init__(self):
        """Initializes the dynamic currency-adjusted growth factor."""
        super().__init__(name='comp_dollar_growth', description='Growth adjusted for USD strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates the equity growth metrics dampened symmetrically by prevailing USD strength.

        Args:
            df (pd.DataFrame): Financial dataset mapping corporate growth matrices and USD indices.

        Returns:
            pd.Series: Evaluated parameter bounds representing currency-conditional growth scores.
        """
        growth_col = 'growth_earnings_growth' if 'growth_earnings_growth' in df.columns else 'earnings_growth'
        if not {'usd_close', growth_col}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Identifies structural FX state configurations measuring spot DXY against local medians
        usd_median = df.groupby('ticker')['usd_close'].transform(lambda x: x.rolling(252).median())
        usd_regime = np.where(df['usd_close'] < usd_median, 1.0, 0.7)
        
        growth_signal = df.groupby('ticker')[growth_col].ffill().clip(-1, 1)
        
        signal = growth_signal * usd_regime
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())

@FactorRegistry.register()
class RiskParityBlend(CompositeFactor):
    """
    Multi-Factor Risk Parity Score.
    
    Purpose:
        Combines Technical Momentum (Price-based) and Sentiment (VIX-based)
        using an equal-weight rank approach to diversify signal sources.
        
    Methodology:
        Average of Cross-Sectional Momentum Rank and Time-Series VIX Rank (Inverted).
    """
    def __init__(self):
        """Initializes the unified multi-factor risk parity momentum ranker."""
        super().__init__(name='comp_risk_parity', description='Momentum + Sentiment Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes relative strength ranking blended symmetrically with localized volatility bounds.

        Args:
            df (pd.DataFrame): Time-series matrices encapsulating continuous pricing geometries.

        Returns:
            pd.Series: Mapped normalized scalar limits representing blended cross-sectional opportunities.
        """
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Evaluates raw idiosyncratic momentum strictly sorting components sequentially per timestep
        mom = df.groupby('ticker')['close'].pct_change(21)
        mom_rank = mom.groupby(df['date']).rank(pct=True)
        
        # Generates inverted time-series percentiles modeling broad sentiment normalization
        vix_rank = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(252).rank(pct=True))
        sentiment_rank = 1 - vix_rank
        
        signal = (mom_rank + sentiment_rank) / 2
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())