"""
Smart Composite Signals
=======================

Provides advanced composite alpha factors capturing non-linear cross-asset interactions.

Purpose
-------
This module implements "smart" composite signals that transcend simple linear 
combinations. It isolates interaction effects between equity fundamentals and 
macro-economic primitives (Volatility, Yields, Commodities, FX). These factors 
are structurally regime-aware, dynamically scaling signal strength based on 
changing global market contexts.

Role in Quantitative Workflow
-----------------------------
Serves as the high-level macro-overlay feature engineering engine. These 
composite vectors are strictly designed to feed non-linear machine learning 
ensembles (e.g., GBDTs), allowing tree nodes to learn regime-conditional 
splits (e.g., discounting Value signals during elevated VIX environments).

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

    Detects market euphoria or capitulation by mapping the velocity of price
    changes against the trajectory of implied volatility (VIX).

    Mathematical Formulation:
        $Signal_t = Mom_{Stock, 21d} - Mom_{VIX, 21d}$

    Interpretation:
        - **High Positive**: "Clean Rally". Price rising, VIX falling (Confidence).
        - **Low/Negative**: "Wall of Worry". Price rising but VIX rising (Instability),
          or Price falling and VIX rising (Panic).
    """
    def __init__(self):
        """Initializes the Momentum-VIX Divergence factor structure."""
        super().__init__(name='comp_div_momentum_vix', description='Momentum-VIX Divergence Signal')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the discrete divergence between asset price momentum and VIX momentum.

        Args:
            df (pd.DataFrame): Multi-asset market data panel containing strictly mapped 
                'close' and 'vix_close' price structures.

        Returns:
            pd.Series: Continuous divergence signal mapped per ticker, smoothed over 
                a 10-day rolling window.
        """
        if not {'close', 'vix_close'}.issubset(df.columns):
            return pd.Series(np.nan, index=df.index)
        
        # Extracts idiosyncratic 21-day continuous asset return velocity
        mom = df.groupby('ticker')['close'].pct_change(21)

        # Extracts systematic 21-day implied volatility momentum mapping market hedging demand
        vix_mom = df.groupby('ticker')['vix_close'].ffill().groupby(df['ticker']).pct_change(21, fill_method=None)
        
        # Calculates empirical divergence. Positive scores indicate healthy structural trends 
        # (Price UP, VIX DOWN), while negative scores flag localized systemic risk.
        signal = mom - vix_mom
        
        # Applies structural smoothing via a 10-day moving average to attenuate high-frequency noise
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)

@FactorRegistry.register()
class ValueYieldCombo(CompositeFactor):
    """
    Rate-Adjusted Value Factor.

    Combines a relative value signal (P/E Rank) with a macro-overlay anchored to
    the 10-Year Treasury Yield. Equities with short-duration cash flows (Value) 
    tend to outperform long-duration cash flows (Growth) in rising rate environments, 
    but high absolute yields compress equity multiples structurally.

    Mathematical Formulation:
        $Score = 0.5 \times (Rank_{XS}(Value) + (1 - \frac{Yield_{10Y}}{5\%}))$
    """
    def __init__(self):
        """Initializes the Rate-Adjusted Value factor structure."""
        super().__init__(name='comp_value_yield', description='Value-Yield Blend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the rate-adjusted value score mapped cross-sectionally.

        Args:
            df (pd.DataFrame): Multi-asset market data panel containing strictly mapped 
                'pe_ratio' fundamentals and a 10-Year Treasury yield proxy.

        Returns:
            pd.Series: Combined cross-sectional rank and normalized yield score, 
                smoothed over a 5-day window.
        """
        us_10y_col = next((c for c in ['us_10y_close', 'us_10y', 'treasury_yield', 'yield_proxy'] if c in df.columns), None)
        if us_10y_col is None or 'pe_ratio' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        # Constructs macro component: Inverted normalized continuous yield score.
        # Assumes benchmark rates < 5% structurally support higher baseline equity multiples.
        yield_smooth = df.groupby('ticker')[us_10y_col].transform(lambda x: x.rolling(21, min_periods=5).mean()).clip(0.1, 5)
        rate_score = 1 - (yield_smooth / 5)
        
        # Constructs equity component: Isolates standard Earnings Yield (1 / P/E).
        # Lower/Upper bounds strictly enforced to neutralize mathematical anomalies.
        inv_pe = 1 / df['pe_ratio'].replace(0, np.nan).clip(1, 200)
        
        # Executes continuous cross-sectional ranking (O(N log N)) independently per discrete temporal block
        value_rank = inv_pe.groupby(df['date']).rank(pct=True)
        
        # Blends macro support metrics and relative equity valuation into a unified probability vector
        signal = (value_rank + rate_score) / 2
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())

@FactorRegistry.register()
class QualityInDownturn(CompositeFactor):
    """
    Crisis-Alpha Quality Factor.

    Isolates highly efficient balance sheets (High ROE, Low Debt) and dynamically 
    amplifies their predictive weighting during periods of elevated systemic variance 
    to exploit institutional "Flight to Quality" regime shifts.

    Conditionality:
        $Signal = Quality \times \mathbb{I}(VIX > \mu_{VIX, 63d} ? 1.5 : 1.0)$
    """
    def __init__(self):
        """Initializes the Crisis-Alpha Quality factor structure."""
        super().__init__(name='comp_quality_stress', description='Quality under VIX Stress')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the structural quality signal dynamically conditioned on VIX expansion.

        Args:
            df (pd.DataFrame): Multi-asset market data panel containing fundamental 
                'roe', 'debt_to_equity', and macro 'vix_close' boundaries.

        Returns:
            pd.Series: Volatility-adjusted quality probability vector smoothed 
                over a 5-day observation window.
        """
        roe = df.get('roe', pd.Series(0, index=df.index))
        debt = df.get('debt_to_equity', pd.Series(1, index=df.index))
        
        if 'vix_close' not in df.columns:
            return (roe.clip(-0.5, 0.5) - (debt.clip(0, 5) * 0.1)).fillna(0)
            
        vix = df.get('vix_close', pd.Series(20, index=df.index))
        
        # Derives geometric baseline Quality Score: Rewards capital efficiency (ROE) 
        # while directly penalizing excessive leverage ratios.
        quality = roe.clip(-0.5, 0.5) - (debt.clip(0, 5) * 0.1)
        
        # Generates conditional regime state trigger: Evaluates the raw VIX trace 
        # against its empirical 1-quarter (63-day) structural moving average.
        vix_ma = df.groupby('ticker')['vix_close'].transform(lambda x: x.rolling(63, min_periods=5).mean())
        stress_trigger = np.where(vix > vix_ma, 1.5, 1.0)
        
        signal = quality * stress_trigger
        return signal.groupby(df['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)

@FactorRegistry.register()
class EarningsMacroAlignment(CompositeFactor):
    """
    Earnings-Macro Sensitivity Score.

    Measures the rolling Pearson correlation tracking individual equity valuation 
    elasticity responding to underlying macro-economic interest rate trajectories.

    Mathematical Formulation:
        $\rho_{63d} = \text{Corr}(P/E_t, \Delta Yield_{t})$

    Interpretation:
        - **Positive Correlation**: Stock strictly benefits from rising yields (e.g., Financials).
        - **Negative Correlation**: Stock hurt by rising yields (e.g., Long-duration Tech).
    """
    def __init__(self):
        """Initializes the Earnings-Macro Alignment factor structure."""
        super().__init__(name='comp_earnings_macro', description='Earnings-Macro Alignment Score')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the structural sensitivity correlation against macro yield constraints.

        Args:
            df (pd.DataFrame): Multi-asset market data panel containing foundational 
                equity ratios and Treasury benchmark derivatives.

        Returns:
            pd.Series: Normalized rolling correlation vector strictly bounded inside [-1, 1].
        """
        target_col = None
        for col in ['pe_ratio', 'growth_earnings_growth', 'earnings_growth']:
            if col in df.columns:
                target_col = col
                break
        
        us_10y_col = next((c for c in ['us_10y_close', 'us_10y', 'treasury_yield', 'yield_proxy'] if c in df.columns), None)
        
        if target_col is None or us_10y_col is None:
            missing = "us_10y_close" if us_10y_col is None else "pe_ratio/growth"
            logger.warning(f"❌ {self.name}: Missing {missing} data")
            return pd.Series(np.nan, index=df.index)
        
        # Isolates aggregate macro vector: Derives the trailing 21-day continuous yield momentum
        macro_momentum = df.groupby('ticker')[us_10y_col].ffill().groupby(df['ticker']).pct_change(21, fill_method=None)
        
        # Computes localized 1-quarter (63-day) discrete correlation matrices mapping 
        # individual valuation changes against the underlying macro vector. Execution O(N x Window).
        alignment = df.groupby('ticker', group_keys=False).apply(
            lambda x: x[target_col].rolling(63, min_periods=10).corr(macro_momentum.loc[x.index]),
            include_groups=False
        )
        
        return alignment.fillna(np.nan)

@FactorRegistry.register()
class MultiAssetOpportunity(CompositeFactor):
    """
    Cross-Asset Consensus Oscillator.

    Aggregates directional trends spanning global commodities (Oil), fixed income 
    (Treasury Yields), and FX parameters (USD) to identify structural shifts in 
    underlying macro-thematic liquidity flows.

    Calculates a binary consensus summation evaluating how many key macro components 
    are concurrently trading above their respective 21-day moving averages.
    """
    def __init__(self):
        """Initializes the Cross-Asset Consensus factor structure."""
        super().__init__(name='comp_multi_asset', description='Oil-Yield-USD Consensus')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the global macro trend consensus scaling array.

        Args:
            df (pd.DataFrame): Time-series multi-asset block populated with standard 
                macro proxies ('oil_close', Yields, 'usd_close').

        Returns:
            pd.Series: Continuous scalar mapped into standardized [0, 100] coordinates.
        """
        us_10y_col = next((c for c in ['us_10y_close', 'us_10y', 'treasury_yield', 'yield_proxy'] if c in df.columns), None)
        if us_10y_col is None:
            return pd.Series(np.nan, index=df.index)
            
        assets = ['oil_close', us_10y_col, 'usd_close']
        if not all(col in df.columns for col in assets):
            return pd.Series(np.nan, index=df.index)
        
        # Evaluates structural macro environments by cumulatively mapping components +1 
        # if currently traversing above their 21-day moving average bounds, and -1 if below.
        consensus = 0
        for col in assets:
            ma = df.groupby('ticker')[col].transform(lambda x: x.rolling(21).mean())
            consensus += np.where(df[col] > ma, 1, -1)
        
        # Interpolates the discrete summation boundaries [-3, +3] structurally 
        # onto an optimized probability continuum spanning [0, 100].
        opp_score = ((consensus / 3) + 1) * 50
        
        # Attenuates high-frequency volatility by applying a 1-week smoothing window
        return pd.Series(opp_score, index=df.index).groupby(df['ticker']).transform(lambda x: x.rolling(5).mean()).fillna(np.nan)