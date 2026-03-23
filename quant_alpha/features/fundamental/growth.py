"""
Fundamental Growth Factors
===========================
Quantitative metrics assessing historical and projected corporate growth rates.

FIXES:
  BUG-032: SustainableGrowthRate.compute() — payout clip now uses
           clip(lower=0.0, upper=1.0) to prevent negative dividend yields
           (data errors) from inflating retention above 1.0 and making SGR
           meaningless. The formula is payout = div_yield × P/E, where
           div_yield must be in decimal form (e.g., 0.02 for 2%); the
           clip(lower=0.0) guards against any vendor providing a negative value.
"""

import numpy as np
import pandas as pd
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor


@FactorRegistry.register()
class EarningsGrowth(SingleColumnFactor):
    """Historical Earnings Growth (annualised rate of change in EPS)."""
    def __init__(self):
        super().__init__(
            'growth_earnings_growth', 'earnings_growth',
            description='Historical Earnings Growth'
        )


@FactorRegistry.register()
class RevenueGrowth(SingleColumnFactor):
    """Historical Revenue Growth (annualised rate of change in top-line sales)."""
    def __init__(self):
        super().__init__(
            'growth_rev_growth', 'rev_growth',
            description='Historical Revenue Growth'
        )


@FactorRegistry.register()
class ForwardEPSGrowth(FundamentalFactor):
    """
    Projected EPS Growth (Forward vs TTM).
    Formula: (EPS_consensus_fwd - EPS_ttm) / |EPS_ttm|
    """
    def __init__(self):
        super().__init__(
            name='growth_fwd_eps_growth',
            description='Projected EPS Growth (Fwd vs TTM)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fwd_col  = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        curr_col = FundamentalColumnValidator.find_column(df, 'eps')

        # Strategy 1: Fallback — derive Forward EPS from Price / Forward P/E
        if not fwd_col:
            fwd_pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
            price_col  = FundamentalColumnValidator.find_column(df, 'price')
            if fwd_pe_col and price_col:
                derived_fwd_eps = df[price_col] / (df[fwd_pe_col] + EPS)
                if curr_col:
                    return (derived_fwd_eps - df[curr_col]) / (df[curr_col].abs() + EPS)

        # Strategy 2: Standard calculation
        if fwd_col and curr_col:
            return (df[fwd_col] - df[curr_col]) / (df[curr_col].abs() + EPS)

        logger.warning(f"⚠️  {self.name}: Missing 'fwd_eps' or 'eps' columns")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class PEGRatio(FundamentalFactor):
    """
    Price/Earnings to Growth Ratio (PEG).
    Formula: P/E / Earnings Growth Rate
    """
    def __init__(self):
        super().__init__('growth_peg_ratio', description='PEG Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Direct Lookup (preferred for vendor accuracy)
        direct_col = FundamentalColumnValidator.find_column(df, 'peg_ratio')
        if direct_col:
            return df[direct_col]

        # Strategy 2: Derive from components
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        g_col  = FundamentalColumnValidator.find_column(df, 'earnings_growth')

        pe_series = None
        if pe_col:
            pe_series = df[pe_col]
        else:
            price_col = FundamentalColumnValidator.find_column(df, 'price')
            eps_col   = FundamentalColumnValidator.find_column(df, 'eps')
            if price_col and eps_col:
                pe_series = df[price_col] / (df[eps_col] + EPS)

        if pe_series is not None and g_col:
            return pe_series / (df[g_col] + EPS)

        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class SustainableGrowthRate(FundamentalFactor):
    """
    Sustainable Growth Rate (SGR) = ROE × Retention Ratio
    Where: Payout Ratio ≈ Dividend Yield × P/E

    FIX BUG-032: payout now clipped with lower=0.0 as well as upper=1.0.
    Without the lower bound, a negative dividend yield (data error) produces
    payout < 0, retention > 1.0, and SGR > ROE — a nonsensical result.
    Dividend yield in decimal form (0.02 = 2%); multiplied by P/E gives payout
    in fraction form. Both bounds must be applied.
    """
    def __init__(self):
        super().__init__(
            name='growth_sgr',
            description='Sustainable Growth Rate (ROE * Retention)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        roe_col = FundamentalColumnValidator.find_column(df, 'roe')
        pe_col  = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        div_col = FundamentalColumnValidator.find_column(df, 'dividend')

        if roe_col and pe_col:
            roe = df[roe_col]
            pe  = df[pe_col]

            # Dividend yield: NaN → 0.0 (non-payer assumption)
            div_yield = df[div_col].fillna(0.0) if div_col else pd.Series(0.0, index=df.index)

            # Payout Ratio = DivYield × P/E
            # Example: 0.02 yield × 20 P/E = 0.40 (40% payout)
            payout = div_yield * pe

            # FIX BUG-032: clip both lower AND upper to keep payout in [0, 1]
            # lower=0.0 prevents negative yields from pushing retention above 1.0
            payout = payout.clip(lower=0.0, upper=1.0)

            retention = 1.0 - payout
            return roe * retention

        logger.warning(f"⚠️  {self.name}: Missing 'roe' or 'pe_ratio'")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class ReinvestmentRate(FundamentalFactor):
    """
    Reinvestment Rate (CapEx Intensity) = (OCF - FCF) / OCF
    """
    def __init__(self):
        super().__init__(
            'growth_reinvestment_rate',
            description='Reinvestment Rate (Capex/OCF)'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')

        if ocf_col and fcf_col:
            capex = df[ocf_col] - df[fcf_col]
            return capex / (df[ocf_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing 'ocf' or 'fcf'")
        return pd.Series(np.nan, index=df.index)