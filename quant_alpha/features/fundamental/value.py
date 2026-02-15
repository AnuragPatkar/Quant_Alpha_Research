import numpy as np
import pandas as pd
from typing import Optional, List
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator, SingleColumnFactor, RatioFactor
    
# ==================== PRICE YIELDS (Clean & Robust) ====================

@FactorRegistry.register()
class EarningsYield(FundamentalFactor):
    def __init__(self):
        super().__init__('val_earnings_yield', description='Earnings Yield (1/PE or EPS/Price)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 1. Try 1 / PE (Most direct)
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)
        
        # 2. Try EPS / Price
        eps_col = FundamentalColumnValidator.find_column(df, 'eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if eps_col and price_col:
            return df[eps_col] / (df[price_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing P/E or (EPS & Price)")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    # Kept as FundamentalFactor because it's a simple inversion 1/PE, not A/B
    def __init__(self):
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: 1 / Forward P/E
        pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)
            
        # Strategy 2: Forward EPS / Price
        fwd_eps_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if fwd_eps_col and price_col:
            return df[fwd_eps_col] / (df[price_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing Forward P/E or Forward EPS")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class BookYield(FundamentalFactor):
    # Hybrid: Try 1/PB first, then BV/Price
    def __init__(self):
        super().__init__(name='val_book_yield', description='Book Yield (1/PB)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        pb_col = FundamentalColumnValidator.find_column(df, 'price_to_book')
        if pb_col:
            return 1.0 / (df[pb_col] + EPS)
            
        bv_col = FundamentalColumnValidator.find_column(df, 'book_value')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if bv_col and price_col:
             return df[bv_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/B data")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class FCFYield(RatioFactor):
    def __init__(self):
        super().__init__('val_fcf_yield', num_key='fcf', den_key='market_cap', description='FCF Yield')

@FactorRegistry.register()
class OperatingCashFlowYield(RatioFactor):
    def __init__(self):
        super().__init__('val_ocf_yield', num_key='ocf', den_key='market_cap', description='OCF Yield')

@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    def __init__(self):
        super().__init__('val_div_yield', description='Dividend Yield')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, 'dividend')
        if col:
            return df[col].fillna(0.0)
        # Default to 0.0 if missing (Safe assumption for Yields)
        return pd.Series(0.0, index=df.index)

# ==================== ADVANCED YIELDS ====================

@FactorRegistry.register()
class ShareholderYield(FundamentalFactor):
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
            total_payout += df[buyback_col].abs()
        
        div_paid_col = FundamentalColumnValidator.find_column(df, 'dividends_paid')
        if div_paid_col:
            total_payout += df[div_paid_col].abs()
        
        return total_payout / (df[mkt_cap_col] + EPS)

@FactorRegistry.register()
class EnterpriseValueFCFYield(FundamentalFactor):
    """
    EV Yield = FCF / Enterprise Value
    EV = Market Cap + Total Debt
    Total Debt (Derived) = (Debt/Equity) * Book Value
    """
    def __init__(self):
        super().__init__(name='val_ev_fcf_yield', description='FCF / Enterprise Value')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        
        if fcf_col and mc_col:
            market_cap = df[mc_col]
            total_debt = pd.Series(0.0, index=df.index)
            
            # Try to find Total Debt directly or derive it
            td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
            de_col = FundamentalColumnValidator.find_column(df, 'debt_equity')
            pb_col = FundamentalColumnValidator.find_column(df, 'price_to_book')
            
            if td_col:
                total_debt = df[td_col].fillna(0.0)
            elif de_col and pb_col:
                # Derived Total Debt = D/E * (Market Cap / PB)
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
    Price-to-Sales Ratio - Less manipulation prone than P/E
    Formula: Market Cap / Total Revenue
    """
    def __init__(self):
        super().__init__(name='val_ps_ratio', description='Price-to-Sales Ratio')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 1. Try Direct Lookup
        ps_col = FundamentalColumnValidator.find_column(df, 'ps_ratio')
        if ps_col:
            # Invert because lower P/S is better (like P/E)
            return 1.0 / (df[ps_col] + EPS)
        
        # 2. Calculate: Market Cap / Revenue
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')
        
        if mc_col and rev_col:
            return df[mc_col] / (df[rev_col] + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing P/S or (Market Cap & Revenue)")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class EVtoEBITDAValuation(FundamentalFactor):
    """
    EV/EBITDA - Most robust valuation metric
    Formula: Enterprise Value / EBITDA
    Lower is Better = More Attractive
    """
    def __init__(self):
        super().__init__(name='val_ev_ebitda', description='EV/EBITDA Valuation')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # 1. Try Direct Lookup
        ev_ebitda_col = FundamentalColumnValidator.find_column(df, 'ev_ebitda')
        if ev_ebitda_col:
            # Invert: Lower EV/EBITDA is better = Higher score
            return 1.0 / (df[ev_ebitda_col] + EPS)
        
        # 2. Calculate: EV / (Revenue * EBITDA Margin)
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda_margin')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')
        
        if mc_col and ebitda_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            
            # EBITDA = Revenue * EBITDA Margin
            ebitda = df[rev_col] * df[ebitda_col]
            
            # Invert the ratio: Higher score = Better value
            return ebitda / (ev + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing data for EV/EBITDA calculation")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class EVtoSalesValuation(FundamentalFactor):
    """
    EV/Sales - Hard to manipulate, robust valuation
    Formula: Enterprise Value / Total Revenue
    Lower is Better = More Attractive
    """
    def __init__(self):
        super().__init__(name='val_ev_sales', description='EV/Sales Valuation')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')
        
        if mc_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            
            # Invert: Higher EV/Sales is worse, so invert for scoring
            return df[rev_col] / (ev + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing Market Cap or Revenue")
        return pd.Series(np.nan, index=df.index)