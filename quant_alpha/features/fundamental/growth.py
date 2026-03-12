"""
Fundamental Growth Factors
==========================
Quantitative metrics assessing historical and projected corporate growth rates.

Purpose
-------
This module constructs alpha factors that evaluate the velocity and sustainability
of a company's expansion. It includes:
1. **Historical Growth**: Realized expansion in Top-line (Revenue) and Bottom-line (Earnings).
2. **Forward Estimates**: Analyst consensus on future EPS trajectories.
3. **Valuation-Adjusted Growth**: Metrics like PEG Ratio (GARP strategies).
4. **Structural Growth**: Derived rates based on capital reinvestment and profitability (SGR).

Usage
-----
Factors are registered with the `FactorRegistry` and computed over standardized
fundamental data frames.

.. code-block:: python

    registry = FactorRegistry()
    growth_factor = registry.get('growth_fwd_eps_growth')
    signals = growth_factor.compute(fundamentals_df)

Importance
----------
- **Alpha Potential**: Growth acceleration is a key driver of momentum and
  long-term capital appreciation.
- **Valuation Context**: High P/E ratios are often justified by high growth rates;
  metrics like PEG help distinguish expensive stocks from true growth opportunities.
- **Sustainability**: The Sustainable Growth Rate (SGR) determines the maximum
  growth a firm can support without raising external equity, a critical solvency check.

Tools & Frameworks
------------------
- **Pandas**: Vectorized arithmetic on fundamental time-series.
- **NumPy**: Robust handling of division-by-zero using machine epsilon.
- **FactorRegistry**: Integration with the central feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor

@FactorRegistry.register()
class EarningsGrowth(SingleColumnFactor):
    """
    Historical Earnings Growth.
    
    The annualized rate of change in Net Income or EPS over the trailing period.
    """
    def __init__(self):
        super().__init__('growth_earnings_growth', 'earnings_growth', description='Historical Earnings Growth')

@FactorRegistry.register()
class RevenueGrowth(SingleColumnFactor):
    """
    Historical Revenue Growth.
    
    The annualized rate of change in Top-line Sales over the trailing period.
    """
    def __init__(self):
        super().__init__('growth_rev_growth', 'rev_growth', description='Historical Revenue Growth')

@FactorRegistry.register()
class ForwardEPSGrowth(FundamentalFactor):
    """
    Projected EPS Growth (Forward vs TTM).
    
    Measures the expected acceleration in earnings based on analyst consensus.
    
    Formula:
    $$ Growth_{fwd} = \frac{EPS_{consensus\_fwd} - EPS_{ttm}}{|EPS_{ttm}|} $$
    """
    def __init__(self):
        super().__init__(name='growth_fwd_eps_growth', description='Projected EPS Growth (Fwd vs TTM)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fwd_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        curr_col = FundamentalColumnValidator.find_column(df, 'eps')
        
        # Strategy 1: Fallback derivation if explicit Forward EPS is missing
        # $$ Forward EPS = \frac{Price}{Forward P/E} $$
        if not fwd_col:
            fwd_pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            if fwd_pe_col and price_col:
                # Implicitly calculate Forward EPS
                derived_fwd_eps = df[price_col] / (df[fwd_pe_col] + EPS)
                if curr_col:
                    # Calculate growth using derived estimate
                    return (derived_fwd_eps - df[curr_col]) / (df[curr_col].abs() + EPS)

        # Strategy 2: Standard calculation using explicit columns
        if fwd_col and curr_col:
            return (df[fwd_col] - df[curr_col]) / (df[curr_col].abs() + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing 'fwd_eps' or 'eps' columns")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class PEGRatio(FundamentalFactor):
    """
    Price/Earnings to Growth Ratio (PEG).
    
    A valuation metric that normalizes the P/E ratio by the expected growth rate,
    identifying "Growth at a Reasonable Price" (GARP).
    
    Formula:
    $$ PEG = \frac{\text{P/E Ratio}}{\text{Earnings Growth Rate}} $$
    """
    def __init__(self):
        super().__init__('growth_peg_ratio', description='PEG Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Direct Lookup (Preferred for vendor accuracy)
        direct_col = FundamentalColumnValidator.find_column(df, 'peg_ratio')
        if direct_col:
            return df[direct_col]
            
        # Strategy 2: Imputation via components
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        g_col = FundamentalColumnValidator.find_column(df, 'earnings_growth')
        
        # Sub-strategy: Derive P/E if missing
        pe_series = None
        if pe_col:
            pe_series = df[pe_col]
        else:
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            eps_col = FundamentalColumnValidator.find_column(df, 'eps')
            if price_col and eps_col:
                pe_series = df[price_col] / (df[eps_col] + EPS)
        
        if pe_series is not None and g_col:
            # Note: Ensure growth rate scaling matches vendor (e.g., 15.0 vs 0.15)
            return pe_series / (df[g_col] + EPS)
            
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class SustainableGrowthRate(FundamentalFactor):
    """
    Sustainable Growth Rate (SGR).
    
    The maximum growth rate a company can sustain without having to increase 
    financial leverage or issue new equity.
    
    Formula:
    $$ SGR = ROE \times (1 - \text{Payout Ratio}) $$
    
    Where:
    $$ \text{Payout Ratio} \approx \text{Dividend Yield} \times \text{P/E Ratio} $$
    """
    def __init__(self):
        super().__init__(name='growth_sgr', description='Sustainable Growth Rate (ROE * Retention)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        roe_col = FundamentalColumnValidator.find_column(df, 'roe')
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        div_col = FundamentalColumnValidator.find_column(df, 'dividend') # Represents Dividend Yield
        
        if roe_col and pe_col:
            roe = df[roe_col]
            pe = df[pe_col]
            
            # Data Integrity: Assume NaN dividend yield implies 0.0 (non-payer)
            div_yield = df[div_col].fillna(0.0) if div_col else 0.0
            
            # Derivation: Payout Ratio = DivYield * PE
            # Example: 2% Yield * 20 P/E = 0.40 (40% Payout)
            payout = div_yield * pe
            
            # Winsorization: Cap payout at 100% (1.0) to prevent negative retention rates
            payout = payout.clip(upper=1.0)
            
            retention = 1.0 - payout
            return roe * retention
            
        logger.warning(f"⚠️  {self.name}: Missing 'roe' or 'pe_ratio'")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ReinvestmentRate(FundamentalFactor):
    """
    Reinvestment Rate (Capex Intensity).
    
    The proportion of Operating Cash Flow plowed back into the business 
    as Capital Expenditures.
    
    Formula:
    $$ Rate = \frac{\text{Capex}}{\text{OCF}} \approx \frac{\text{OCF} - \text{FCF}}{\text{OCF}} $$
    """
    def __init__(self): 
        super().__init__('growth_reinvestment_rate', description='Reinvestment Rate (Capex/OCF)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Custom Logic: Derive Capex since it's often not a direct column
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        
        if ocf_col and fcf_col:
            # Derivation: Capex = Operating Cash Flow - Free Cash Flow
            capex = df[ocf_col] - df[fcf_col]
            
            return capex / (df[ocf_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing 'ocf' or 'fcf'")
        return pd.Series(np.nan, index=df.index)