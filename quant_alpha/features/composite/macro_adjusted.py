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
        super().__init__(name='comp_macro_momentum', description='Momentum modulated by VIX')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Input validation: Requires idiosyncratic price and systematic volatility
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # 1. Individual Stock Momentum (21-day) ($O(N)$)
        # Grouping by 'ticker' is essential to prevent cross-contamination between assets
        momentum = df.groupby('ticker')['close'].pct_change(21) * 100
        
        # 2. VIX Adjustment (Inverse Volatility Weighting)
        # Use transform to broadcast the market-wide VIX series to the shape of the panel
        vix_smooth = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(5).mean()).clip(10, 40)
        adjustment = 25 / vix_smooth
        
        return momentum * adjustment

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
        super().__init__(name='comp_oil_value', description='Value signals filtered by oil environment')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'oil_close', 'pe_ratio'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Define Oil Regime: Spot price vs 1-Year Median (252 days)
        oil_median = df.groupby('ticker')['oil_close'].transform(lambda x: x.rolling(252).median())
        oil_regime = np.where(df['oil_close'] < oil_median, 1.0, 0.5)
        
        # Base Value Signal: Earnings Yield (1 / PE)
        # Clipped to [0.01, 1.0] range (PE 1 to 100) to handle outliers/negative earnings
        value_signal = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 100)
        
        return value_signal * oil_regime

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
        super().__init__(name='comp_rate_quality', description='Quality adjusted by interest rates')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'us_10y_close', 'roe'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Rate Penalty: Linearly decays as yields approach 5%
        rate_weight = 1 - (df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(21).mean()).clip(0, 5) / 5)
        
        # Fundamental Signal: Return on Equity (ROE)
        # Forward-fill to handle quarterly reporting cadence
        quality_signal = df.groupby('ticker')['roe'].ffill().clip(-1, 1)
        
        return quality_signal * rate_weight

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
        super().__init__(name='comp_dollar_growth', description='Growth adjusted for USD strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'usd_close', 'earnings_growth'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # FX Regime: Spot DXY vs 1-Year Median
        usd_median = df.groupby('ticker')['usd_close'].transform(lambda x: x.rolling(252).median())
        usd_regime = np.where(df['usd_close'] < usd_median, 1.0, 0.7)
        
        # Fundamental Signal: Earnings Growth
        growth_signal = df.groupby('ticker')['earnings_growth'].ffill().clip(-1, 1)
        
        return growth_signal * usd_regime

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
        super().__init__(name='comp_risk_parity', description='Momentum + Sentiment Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Component 1: Cross-Sectional Momentum Rank
        # Ranks stocks against peers on specific date (Relative Strength)
        mom = df.groupby('ticker')['close'].pct_change(21)
        mom_rank = mom.groupby(df['date']).rank(pct=True)
        
        # Component 2: Time-Series VIX Rank (Inverted)
        # Ranks current VIX against its own 1-year history.
        # Low VIX relative to history = High Rank (1.0) -> Bullish
        vix_rank = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(252).rank(pct=True))
        sentiment_rank = 1 - vix_rank
        
        return (mom_rank + sentiment_rank) / 2