"""
Smart Composite Signals
=======================
Advanced alpha factors that capture non-linear interactions between asset classes.

Purpose
-------
This module implements "smart" composite signals that go beyond simple linear combinations.
It focuses on **interaction effects** between equity fundamentals and macro-economic
primitives (Volatility, Rates, Commodities, FX). These factors are designed to be
regime-aware, adjusting their signal strength based on the broader market context.

Usage
-----
Factors are automatically registered with the `FactorRegistry` upon import.

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry

    # Compute divergence between price momentum and implied volatility
    df_div = FactorRegistry.compute("comp_div_momentum_vix", market_data_df)

Importance
----------
- **Regime-Conditional Alpha**: Captures "flight to quality" dynamics during stress
  events (`QualityInDownturn`) and discount-rate sensitivity (`ValueYieldCombo`).
- **Cross-Asset Signal**: Uses information from the options market (VIX) and
  fixed income (Treasuries) to validate equity price trends, reducing false positives.
- **Divergence Detection**: Identifies unsustainable rallies where price rises
  but fear (VIX) increases ("climbing a wall of worry").

Tools & Frameworks
------------------
- **Pandas**: Used for complex grouping and rolling correlation calculations.
- **NumPy**: Vectorized conditional logic (`np.where`) and numerical clipping.
"""

import pandas as pd
import numpy as np
from ..base import CompositeFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class MomentumVIXDivergence(CompositeFactor):
    """
    Momentum-Volatility Divergence Indicator.

    Purpose:
        Detects market euphoria or capitulation by comparing the velocity of price
        changes against the velocity of implied volatility (VIX).

    Mathematical Formulation:
        $$ Signal_t = Mom_{Stock, 21d} - Mom_{VIX, 21d} $$

    Interpretation:
        - **High Positive**: "Clean Rally". Price rising, VIX falling (Confidence).
        - **Low/Negative**: "Wall of Worry". Price rising but VIX rising (Instability),
          or Price falling and VIX rising (Panic).
    """
    def __init__(self):
        super().__init__(name='comp_div_momentum_vix', description='Momentum-VIX Divergence Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # 1. Idiosyncratic Stock Momentum (21-day)
        mom = df.groupby('ticker')['close'].pct_change(21)
        
        # 2. Systematic VIX Momentum (21-day)
        # Represents the rate of change in market fear/hedging demand.
        vix_mom = df.groupby('ticker')['vix_close'].pct_change(21)
        
        # Divergence Logic:
        # - If Price $\uparrow$ and VIX $\downarrow$: Signal boosts (Healthy Trend).
        # - If Price $\uparrow$ and VIX $\uparrow$: Signal dampens (Divergence/Risk).
        signal = mom - vix_mom
        
        # Smoothing: 10-day rolling mean to reduce daily noise.
        # Grouping by ticker ensures time-series integrity ($O(N)$).
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)

@FactorRegistry.register()
class ValueYieldCombo(CompositeFactor):
    """
    Rate-Adjusted Value Factor.

    Logic:
        Combines a relative value signal (P/E Rank) with a macro-overlay based on
        the 10-Year Treasury Yield. Value stocks (short duration) tend to outperform
        Growth stocks (long duration) in rising rate environments, but high absolute
        yields can compress multiples globally.

    Formula:
        $$ Score = 0.5 \times (Rank_{XS}(Value) + (1 - \frac{Yield_{10Y}}{5\%})) $$
    """
    def __init__(self):
        super().__init__(name='comp_value_yield', description='Value-Yield Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'us_10y_close' not in df.columns or 'pe_ratio' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Macro Component: Inverted Yield Score
        # Lower rates (< 5%) imply higher equity multiple support.
        yield_smooth = df.groupby('ticker')['us_10y_close'].transform(lambda x: x.rolling(21, min_periods=5).mean()).clip(0.1, 5)
        rate_score = 1 - (yield_smooth / 5)
        
        # Equity Component: Cross-Sectional Value Rank
        # Inverse P/E (Earnings Yield). Clipped to handle outliers/negative earnings.
        inv_pe = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 200)
        
        # Rank stocks against peers on each specific date ($O(N \log N)$)
        value_rank = inv_pe.groupby(df['date']).rank(pct=True)
        
        return (value_rank + rate_score) / 2

@FactorRegistry.register()
class QualityInDownturn(CompositeFactor):
    """
    Crisis-Alpha Quality Factor.

    Logic:
        Identifies high-quality balance sheets (High ROE, Low Debt) and amplifies
        their signal strength during periods of high market volatility.

    Conditionality:
        $$ Signal = Quality \times \mathbb{I}(VIX > \mu_{VIX, 63d} ? 1.5 : 1.0) $$

    Importance:
        "Flight to Quality" is a persistent phenomenon during equity drawdowns.
    """
    def __init__(self):
        super().__init__(name='comp_quality_stress', description='Quality under VIX Stress')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Feature availability check with safe fallbacks
        roe = df.get('roe', pd.Series(0, index=df.index))
        debt = df.get('debt_to_equity', pd.Series(1, index=df.index))
        vix = df.get('vix_close', pd.Series(20, index=df.index))
        
        # Base Quality Score: Reward Efficiency (ROE), Penalize Leverage (Debt)
        # ROE clipped to [-50%, 50%] to remove accounting anomalies.
        quality = roe.clip(-0.5, 0.5) - (debt.clip(0, 5) * 0.1)
        
        # Stress Multiplier: Regime Detection
        # If current VIX is above its 1-quarter moving average, the market is stressed.
        vix_ma = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(63, min_periods=5).mean())
        stress_trigger = np.where(vix > vix_ma, 1.5, 1.0)
        
        return (quality * stress_trigger).fillna(0)

@FactorRegistry.register()
class EarningsMacroAlignment(CompositeFactor):
    """
    Earnings-Macro Sensitivity Score.

    Purpose:
        Measures the correlation between a stock's valuation (or growth) and 
        macro-economic interest rates.

    Mathematical Formulation:
        $$ \rho_{63d} = \text{Corr}(P/E_t, \Delta Yield_{t}) $$

    Interpretation:
        - **Positive Correlation**: Stock benefits from (or re-rates with) rising yields (e.g., Financials).
        - **Negative Correlation**: Stock hurt by rising yields (e.g., Long-duration Tech).
    """
    def __init__(self):
        super().__init__(name='comp_earnings_macro', description='Earnings-Macro Alignment Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Dynamic target selection: Prefer P/E for daily frequency, fallback to Growth
        target_col = 'pe_ratio' if 'pe_ratio' in df.columns else 'earnings_growth'
        
        if 'us_10y_close' not in df.columns or target_col not in df.columns:
            logger.warning(f"❌ {self.name}: Missing macro or earnings data")
            return pd.Series(0, index=df.index)
        
        # 1. Macro Signal: 21-day Change in 10Y Yields
        # Grouping ensures index alignment, though yields are systematic (same for all)
        macro_momentum = df.groupby('ticker')['us_10y_close'].pct_change(21)
        
        # 2. Rolling Correlation ($O(N \times Window)$)
        # Measures alignment between stock valuation/earnings and macro yields over 1 quarter (63d).
        # Note: `apply` here iterates per group; efficient enough for <5000 tickers.
        alignment = df.groupby('ticker', group_keys=False).apply(
            lambda x: x[target_col].rolling(63, min_periods=10).corr(macro_momentum.loc[x.index])
        )
        
        # Default to 0 (Neutral/No Relationship) if data insufficient
        return alignment.fillna(0)

@FactorRegistry.register()
class MultiAssetOpportunity(CompositeFactor):
    """
    Cross-Asset Consensus Oscillator.

    Purpose:
        Aggregates trends across Oil, Bonds (Yields), and Currency (USD) to
        identify broad macro-thematic opportunities.

    Logic:
        Calculates a consensus score based on how many macro assets are trading
        above their 21-day moving average.
    """
    def __init__(self):
        super().__init__(name='comp_multi_asset', description='Oil-Yield-USD Consensus')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        assets = ['oil_close', 'us_10y_close', 'usd_close']
        if not all(col in df.columns for col in assets):
            return pd.Series(50, index=df.index)
        
        # Calculate Binary Trend Direction for each macro asset
        # +1 if Price > 21-Day MA, else -1
        consensus = 0
        for col in assets:
            ma = df.groupby('ticker')[col].transform(lambda x: x.rolling(21).mean())
            consensus += np.where(df[col] > ma, 1, -1)
        
        # Normalization: Map discrete range [-3, +3] to continuous [0, 100]
        # -3 (All Bearish) -> 0
        #  0 (Mixed)       -> 50
        # +3 (All Bullish) -> 100
        opp_score = ((consensus / 3) + 1) * 50
        
        # Smooth the final score over 1 week (5 days)
        return pd.Series(opp_score, index=df.index).groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).fillna(50)