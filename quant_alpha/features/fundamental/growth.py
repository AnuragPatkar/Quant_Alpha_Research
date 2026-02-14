"""
Growth Factors
Focus: Historical and Forward-looking growth metrics.

Factors:
1. Earnings Growth (Historical)
2. Revenue Growth (Historical)
3. Forward EPS Growth (Projected)
4. PEG Ratio (Valuation relative to Growth)
5. Sustainable Growth Rate (Derived from ROE & Retention)
6. Reinvestment Rate (Capex Intensity)
"""

import numpy as np
import pandas as pd
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor

@FactorRegistry.register()
class EarningsGrowth(SingleColumnFactor):
    def __init__(self):
        super().__init__('growth_earnings_growth', 'earnings_growth', description='Historical Earnings Growth')

@FactorRegistry.register()
class RevenueGrowth(SingleColumnFactor):
    def __init__(self):
        super().__init__('growth_rev_growth', 'rev_growth', description='Historical Revenue Growth')

@FactorRegistry.register()
class ForwardEPSGrowth(FundamentalFactor):
    def __init__(self):
        super().__init__(name='growth_fwd_eps_growth', description='Projected EPS Growth (Fwd vs TTM)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fwd_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        curr_col = FundamentalColumnValidator.find_column(df, 'eps')
        
        # Fallback: Derive Fwd EPS from Forward PE if missing
        if not fwd_col:
            fwd_pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            if fwd_pe_col and price_col:
                # Fwd EPS = Price / Fwd PE
                # We calculate this temporarily
                derived_fwd_eps = df[price_col] / (df[fwd_pe_col] + EPS)
                if curr_col:
                    return (derived_fwd_eps - df[curr_col]) / (df[curr_col].abs() + EPS)

        if fwd_col and curr_col:
            # (Forward - Current) / |Current|
            return (df[fwd_col] - df[curr_col]) / (df[curr_col].abs() + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing 'fwd_eps' or 'eps' columns")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class PEGRatio(FundamentalFactor):
    """
    Price/Earnings to Growth Ratio.
    Strategy:
    1. Try direct lookup (e.g., 'pegRatio')
    2. Calculate: P/E / Earnings Growth
    """
    def __init__(self):
        super().__init__('growth_peg_ratio', description='PEG Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 1. Try Direct Lookup
        direct_col = FundamentalColumnValidator.find_column(df, 'peg_ratio')
        if direct_col:
            return df[direct_col]
            
        # 2. Try Calculation
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        g_col = FundamentalColumnValidator.find_column(df, 'earnings_growth')
        
        # Derive PE if missing
        pe_series = None
        if pe_col:
            pe_series = df[pe_col]
        else:
            # Try Price / EPS
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            eps_col = FundamentalColumnValidator.find_column(df, 'eps')
            if price_col and eps_col:
                pe_series = df[price_col] / (df[eps_col] + EPS)
        
        if pe_series is not None and g_col:
            return pe_series / (df[g_col] + EPS)
            
        # logger.warning(f"⚠️  {self.name}: Missing PEG data (Direct or Calculated)")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class SustainableGrowthRate(FundamentalFactor):
    """
    SGR = ROE * Retention Ratio
    Retention Ratio = 1 - Payout Ratio
    Payout Ratio approx = Dividend Yield * P/E Ratio
    """
    def __init__(self):
        super().__init__(name='growth_sgr', description='Sustainable Growth Rate (ROE * Retention)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        roe_col = FundamentalColumnValidator.find_column(df, 'roe')
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        div_col = FundamentalColumnValidator.find_column(df, 'dividend') # Yield
        
        if roe_col and pe_col:
            roe = df[roe_col]
            pe = df[pe_col]
            
            # Handle Dividend (fill NaN with 0 for non-payers)
            div_yield = df[div_col].fillna(0.0) if div_col else 0.0
            
            # Payout Ratio = DivYield * PE (e.g., 0.02 * 20 = 0.40 or 40%)
            payout = div_yield * pe
            
            # Cap payout at 1.0 (100%) to avoid negative retention
            payout = payout.clip(upper=1.0)
            
            retention = 1.0 - payout
            return roe * retention
            
        logger.warning(f"⚠️  {self.name}: Missing 'roe' or 'pe_ratio'")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ReinvestmentRate(FundamentalFactor):
    """
    Measures % of OCF reinvested into the business (Capex).
    Formula: (OCF - FCF) / OCF
    """
    def __init__(self): 
        super().__init__('growth_reinvestment_rate', description='Reinvestment Rate (Capex/OCF)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # We need custom logic because numerator is (OCF - FCF)
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        
        if ocf_col and fcf_col:
            # Capex Proxy = OCF - FCF
            capex = df[ocf_col] - df[fcf_col]
            # Reinvestment = Capex / OCF
            return capex / (df[ocf_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing 'ocf' or 'fcf'")
        return pd.Series(np.nan, index=df.index)