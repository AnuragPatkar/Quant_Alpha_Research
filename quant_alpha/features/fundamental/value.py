import numpy as np
import pandas as pd
from typing import Optional, List
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from .utils import FundamentalColumnValidator
    
# ==================== PRICE YIELDS (Clean & Robust) ====================

@FactorRegistry.register()
class EarningsYield(FundamentalFactor):
    def __init__(self):
        super().__init__(name='val_earnings_yield', description='Trailing Earnings Yield (1/PE)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Step 1: Try Direct Ratio (1/PE)
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)
        
        # Step 2: Try Components (EPS / Price)
        eps_col = FundamentalColumnValidator.find_column(df, 'eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        
        if eps_col and price_col:
            return df[eps_col] / (df[price_col] + EPS)
        
        # Step 3: Soft Fail (Log Warning + Return NaN)
        logger.warning(f"⚠️  {self.name}: Missing P/E or EPS data")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    def __init__(self):
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing Forward P/E")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class BookYield(FundamentalFactor):
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
class FCFYield(FundamentalFactor):
    def __init__(self):
        super().__init__(name='val_fcf_yield', description='FCF Yield')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check pre-calculated first
        if 'freeCashFlowYield' in df.columns:
            return df['freeCashFlowYield']

        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        mkt_cap_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        
        if fcf_col and mkt_cap_col:
            return df[fcf_col] / (df[mkt_cap_col] + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing FCF or Market Cap")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class OperatingCashFlowYield(FundamentalFactor):
    def __init__(self):
        super().__init__(name='val_ocf_yield', description='OCF Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        mkt_cap_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        
        if ocf_col and mkt_cap_col:
            return df[ocf_col] / (df[mkt_cap_col] + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing OCF or Market Cap")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    def __init__(self):
        super().__init__(name='val_div_yield', description='Dividend Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        div_col = FundamentalColumnValidator.find_column(df, 'dividend')
        if div_col:
            return df[div_col]
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
