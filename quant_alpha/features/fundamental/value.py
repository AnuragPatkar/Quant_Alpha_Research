"""
Fundamental Value Factors
=========================
Quantitative metrics assessing the relative valuation of assets.

FIXES:
  BUG-033: PriceToSalesRatio.compute() Strategy 2 was returning
           Market_Cap / Revenue (= P/S ratio, the RAW multiple) instead of
           Revenue / Market_Cap (= Sales Yield, the inverted signal).
           Strategy 1 correctly returns 1/PS, but Strategy 2 was backwards,
           meaning cheap stocks scored LOW when components were used. Fixed.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor


# ==================== PRICE YIELDS ====================

@FactorRegistry.register()
class EarningsYield(FundamentalFactor):
    """
    Earnings Yield (E/P).
    Formula: E/P = 1/PE ≈ EPS / Price
    Higher = cheaper (better value).
    """
    def __init__(self):
        super().__init__('val_earnings_yield', description='Earnings Yield (1/PE or EPS/Price)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert P/E Ratio
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if pe_col:
            pe = df[pe_col].copy().clip(lower=0.1, upper=200)
            return 1.0 / (pe + EPS)

        # Strategy 2: Derive from EPS and Price
        eps_col   = FundamentalColumnValidator.find_column(df, 'eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if eps_col and price_col:
            return df[eps_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/E or (EPS & Price)")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    """
    Forward Earnings Yield.
    Formula: Yield_fwd = 1 / Forward P/E
    """
    def __init__(self):
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)

        fwd_eps_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        price_col   = FundamentalColumnValidator.find_column(df, 'price')
        if fwd_eps_col and price_col:
            return df[fwd_eps_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing Forward P/E or Forward EPS")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class BookYield(FundamentalFactor):
    """
    Book Yield (B/P).
    Formula: B/P = 1/PB = Book Value / Price
    """
    def __init__(self):
        super().__init__(name='val_book_yield', description='Book Yield (1/PB)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        pb_col = FundamentalColumnValidator.find_column(df, 'pb_ratio')
        if pb_col:
            return 1.0 / (df[pb_col] + EPS)

        bv_col    = FundamentalColumnValidator.find_column(df, 'book_value')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if bv_col and price_col:
            return df[bv_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/B data")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class FCFYield(RatioFactor):
    """
    Free Cash Flow Yield.
    Formula: FCF / Market Cap
    """
    def __init__(self):
        super().__init__('val_fcf_yield', num_key='fcf', den_key='market_cap',
                         description='FCF Yield')


@FactorRegistry.register()
class OperatingCashFlowYield(RatioFactor):
    """
    Operating Cash Flow Yield.
    Formula: OCF / Market Cap
    """
    def __init__(self):
        super().__init__('val_ocf_yield', num_key='ocf', den_key='market_cap',
                         description='OCF Yield')


@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    """
    Dividend Yield.
    Missing data imputed as 0.0 (non-payer).
    """
    def __init__(self):
        super().__init__(name='val_dividend_yield', description='Dividend Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, 'dividend')
        if col:
            return df[col].fillna(0.0)
        return pd.Series(0.0, index=df.index)


# ==================== ADVANCED YIELDS ====================

@FactorRegistry.register()
class ShareholderYield(FundamentalFactor):
    """
    Shareholder Yield = (Dividends + Net Buybacks) / Market Cap
    """
    def __init__(self):
        super().__init__(name='val_shareholder_yield', description='Shareholder Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mkt_cap_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        if not mkt_cap_col:
            logger.warning(f"⚠️  {self.name}: Missing Market Cap")
            return pd.Series(np.nan, index=df.index)

        total_payout = pd.Series(0.0, index=df.index)

        buyback_col = FundamentalColumnValidator.find_column(df, 'buyback')
        if buyback_col:
            total_payout = total_payout + df[buyback_col].abs()

        div_paid_col = FundamentalColumnValidator.find_column(df, 'dividends_paid')
        if div_paid_col:
            total_payout = total_payout + df[div_paid_col].abs()

        return total_payout / (df[mkt_cap_col] + EPS)


@FactorRegistry.register()
class EnterpriseValueFCFYield(FundamentalFactor):
    """
    EV FCF Yield = FCF / (Market Cap + Total Debt)
    """
    def __init__(self):
        super().__init__(name='val_ev_fcf_yield', description='FCF / Enterprise Value')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')

        if fcf_col and mc_col:
            market_cap = df[mc_col]
            total_debt = pd.Series(0.0, index=df.index)

            td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
            de_col = FundamentalColumnValidator.find_column(df, 'debt_equity')
            pb_col = FundamentalColumnValidator.find_column(df, 'pb_ratio')

            if td_col:
                total_debt = df[td_col].fillna(0.0)
            elif de_col and pb_col:
                book_value = market_cap / (df[pb_col] + EPS)
                total_debt = df[de_col] * book_value

            ev = market_cap + total_debt
            return df[fcf_col] / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing columns for EV calculation")
        return pd.Series(np.nan, index=df.index)


# ==================== ADDITIONAL VALUATION METRICS ====================

@FactorRegistry.register()
class PriceToSalesRatio(FundamentalFactor):
    """
    Sales Yield (Inverted P/S Ratio).
    Higher is better (cheaper / undervalued).
    Formula: Sales Yield = 1/PS = Revenue / Market Cap

    FIX BUG-033: Strategy 2 previously returned Market_Cap / Revenue (= raw P/S),
    which was directionally backwards relative to Strategy 1 (1/PS). The signal
    was inverted when falling back to components. Fixed to Revenue / Market_Cap.
    """
    def __init__(self):
        super().__init__(name='val_ps_ratio', description='Sales Yield (1/PS)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert pre-computed P/S Ratio
        ps_col = FundamentalColumnValidator.find_column(df, 'ps_ratio')
        if ps_col:
            return 1.0 / (df[ps_col] + EPS)

        # Strategy 2: Derive from components
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and rev_col:
            # FIX BUG-033: Revenue / Market_Cap (Sales Yield), NOT Market_Cap / Revenue
            return df[rev_col] / (df[mc_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/S or (Market Cap & Revenue)")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class EVtoEBITDAValuation(FundamentalFactor):
    """
    EV/EBITDA (Inverted) — EBITDA Yield.
    Formula: EBITDA / EV
    Higher is better.
    """
    def __init__(self):
        super().__init__(name='val_ev_ebitda', description='EV/EBITDA Valuation')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert pre-computed EV/EBITDA
        ev_ebitda_col = FundamentalColumnValidator.find_column(df, 'ev_ebitda')
        if ev_ebitda_col:
            return 1.0 / (df[ev_ebitda_col] + EPS)

        # Strategy 2: Derive from raw components
        mc_col     = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col     = FundamentalColumnValidator.find_column(df, 'total_debt')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda_margin')
        rev_col    = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and ebitda_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            ebitda = df[rev_col] * df[ebitda_col]
            return ebitda / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing data for EV/EBITDA calculation")
        return pd.Series(np.nan, index=df.index)


@FactorRegistry.register()
class EVtoSalesValuation(FundamentalFactor):
    """
    EV/Sales (Inverted) — Enterprise Sales Yield.
    Formula: Revenue / EV
    Higher is better.
    """
    def __init__(self):
        super().__init__(name='val_ev_sales', description='EV/Sales Valuation')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col  = FundamentalColumnValidator.find_column(df, 'total_debt')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if mc_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            return df[rev_col] / (ev + EPS)

        logger.warning(f"⚠️  {self.name}: Missing Market Cap or Revenue")
        return pd.Series(np.nan, index=df.index)