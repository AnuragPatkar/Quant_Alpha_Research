"""
Value Factors (Production Grade - Yield Edition)
Focus: ALL factors are converted to YIELDS (Inverse Ratios).
Advantage: Higher Score ALWAYS equals Better Value (Cheaper).

Total Factors: 13

--- MAPPING GUIDE (Factor Name -> Implemented Class) ---
1.  P/E Ratio          -> EarningsYield (1 / PE)
2.  Forward P/E        -> ForwardEarningsYield (1 / Forward PE)
3.  P/B Ratio          -> BookYield (1 / PB)
4.  P/S Ratio          -> SalesYield (1 / PS)
5.  FCF Yield          -> FCFYield
6.  Cash Flow to Price -> OperatingCashFlowYield
7.  Dividend Yield     -> DividendYield
8.  EV/EBITDA          -> EbitdaEVYield (1 / EV/EBITDA)
9.  EV/Sales           -> RevenueEVYield (1 / EV/Sales)
10. Acquirers Multiple -> AcquirersYield (1 / Acquirers Multiple)
11. Shareholder Yield  -> ShareholderYield
12. PEG Ratio          -> PEGYield (1 / PEG)
13. Tangible Book      -> TangibleBookYield
-----------------------------------------------------------
"""
import numpy as np
import pandas as pd
from ..registry import FactorRegistry
from ..base import FundamentalFactor

EPS = 1e-9

# ==================== 1. PRICE YIELDS (Equity Value) ====================
@FactorRegistry.register
class EarningsYield(FundamentalFactor):
    """
    [Mapping: P/E Ratio]
    Earnings Yield = Trailing EPS / Price.
    This is the inverse of the P/E Ratio.
    Higher Yield = Cheaper Stock.
    """

    def __init__(self):
        super().__init__(name='val_earnings_yield', description='Trailing Earnings Yield (1/PE)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'trailingEps' in df.columns and 'currentPrice' in df.columns:
            return df['trailingEps'] / (df['currentPrice'] + EPS)
        elif 'trailingPE' in df.columns:
            return 1.0 / (df['trailingPE'] + EPS)
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    """
    [Mapping: Forward P/E]
    Forward Earnings Yield = Forward EPS / Price.
    Inverse of Forward P/E.
    """
    def __init__(self):
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield (1/Forward PE)')

    def compute(self, df:pd.DataFrame) -> pd.Series:
        if 'forwardEps' in df.columns and 'currentPrice' in df.columns:
            return df['forwardEps'] / (df['currentPrice'] + EPS)
        elif 'forwardPE' in df.columns:
            return 1.0 / (df['forwardPE'] + EPS)
        return pd.Series(np.nan, index=df.index)
    
@FactorRegistry.register()
class BookYield(FundamentalFactor):
    """
    [Mapping: P/B Ratio & Book-to-Market]
    Book Yield = Book Value / Price.
    Inverse of P/B Ratio. Also known as Book-to-Market.
    """
    def __init__(self):
        super().__init__(name='val_book_yield', description='Book Yield (1/PB)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'bookValue' in df.columns and 'currentPrice' in df.columns:
            return df['bookValue'] / (df['currentPrice'] + EPS)
        elif 'priceToBook' in df.columns:
            return 1.0 / (df['priceToBook'] + EPS)
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class SalesYield(FundamentalFactor):
    """
    [Mapping: P/S Ratio]
    Sales Yield = Revenue / Price.
    Inverse of P/S Ratio.
    """
    def __init__(self):
        super().__init__(name='val_sales_yield', description='Sales Yield (1/PS)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'revenuePerShare' in df.columns and 'currentPrice' in df.columns:
            return df['revenuePerShare'] / (df['currentPrice'] + EPS)
        elif 'priceToSalesTrailing12Months' in df.columns:
            return 1.0 / (df['priceToSalesTrailing12Months'] + EPS)
        return pd.Series(np.nan, index=df.index)
    
@FactorRegistry.register()
class FCFYield(FundamentalFactor):
    """
    [Mapping: FCF Yield]
    FCF Yield = Free Cash Flow / Price.
    
    """
    def __init__(self):
        super().__init__(name='val_fcf_yield', description='FCF Yield (1/FCF)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'freeCashflow' in df.columns and 'marketCap' in df.columns:
            return df['freeCashflow'] / (df['marketCap'] + EPS)
        elif 'freeCashFlowYield' in df.columns:
            return df['freeCashFlowYield']
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class OperatingCashFlowYield(FundamentalFactor):
    """
    [Mapping: Cash Flow to Price]
    OCF Yield = Operating Cash Flow / Market Cap.
    """
    def __init__(self):
        super().__init__(name='val_ocf_yield', description='Operating Cash Flow Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'operatingCashflow' in df.columns and 'marketCap' in df.columns:
            return df['operatingCashflow'] / (df['marketCap'] + EPS)
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    """
    [Mapping: Dividend Yield]
    Annual Dividend / Price.
    """
    def __init__(self):
        super().__init__(name='val_div_yield', description='Dividend Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'dividendYield' in df.columns:
            return df['dividendYield']
        elif 'trailingAnnualDividendYield' in df.columns:
            return df['trailingAnnualDividendYield']
        return pd.Series(0.0, index=df.index)

# ==================== 2. ENTERPRISE YIELDS (Debt Included) ====================
@FactorRegistry.register()
class EbitdaEVYield(FundamentalFactor):
    """
    [Mapping: EV/EBITDA]
    EBITDA Yield = EBITDA / Enterprise Value.
    Inverse of EV/EBITDA.
    """
    def __init__(self):
        super().__init__(name='val_ebitda_ev_yield', description='EBITDA / EV Yield (1 / EV_EBITDA)')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = 'enterpriseToEbitda' if 'enterpriseToEbitda' in df.columns else 'enterpriseValueOverEBITDA'
        if col in df.columns:
            return 1.0 / (df[col] + EPS)
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class RevenueEVYield(FundamentalFactor):
    """
    [Mapping: EV/Sales]
    Revenue Yield = Revenue / Enterprise Value.
    Inverse of EV/Sales.
    """
    def __init__(self):
        super().__init__(name='val_revenue_ev_yield', description='Revenue / EV Yield (1 / EV_Sales)')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = 'enterpriseToRevenue' if 'enterpriseToRevenue' in df.columns else 'enterpriseValueOverRevenue'
        if col in df.columns:
            return 1.0 / (df[col] + EPS)
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class AcquirersYield(FundamentalFactor):
    """
    [Mapping: Acquirers Multiple]
    Acquirer's Yield = EBIT / Enterprise Value.
    Inverse of EV/EBIT. Deep Value metric.
    """
    def __init__(self):
        super().__init__(name='val_acquirers_yield', description='EBIT / EV Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'enterpriseValue' in df.columns and 'EBIT' in df.columns:
            return df['EBIT'] / (df['enterpriseValue'] + EPS)
        return pd.Series(np.nan, index=df.index)
    
# ==================== 3. ADVANCED YIELDS ====================
@FactorRegistry.register()
class ShareholderYield(FundamentalFactor):
    """
    [Mapping: Shareholder Yield]
    Yield = (Dividends + Net Buybacks) / Market Cap.
    """
    def __init__(self):
        super().__init__(name='val_shareholder_yield', description='Dividends + Buybacks Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        buyback_col = 'RepurchaseOfCapitalStock'
        div_col = 'CashDividendsPaid'
        mkt_cap_col = 'marketCap'
        
        if mkt_cap_col in df.columns:
            total_payout = pd.Series(0.0, index=df.index)
            if buyback_col in df.columns:
                total_payout += df[buyback_col].abs()
            if div_col in df.columns:
                total_payout += df[div_col].abs()
            return total_payout / (df[mkt_cap_col] + EPS)
        return pd.Series(np.nan, index=df.index)
    
@FactorRegistry.register()
class PEGYield(FundamentalFactor):
    """
    [Mapping: PEG Ratio]
    PEG Yield = 1 / PEG Ratio.
    """

    def __init__(self):
        super().__init__(name='val_peg_yield', description='PEG Yield (1/PEG)')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = 'trailingPegRatio' if 'trailingPegRatio' in df.columns else 'pegRatio'
        if col in df.columns:
            return 1.0 / (df[col] + EPS)
        return pd.Series(np.nan, index=df.index)
    
@FactorRegistry.register()
class TangibleBookYield(FundamentalFactor):
    """
    [Mapping: Tangible Book]
    Tangible Book Yield = Tangible Book Value / Market Cap.
    """
    def __init__(self):
        super().__init__(name='val_tbv_yield', description='Tangible Book Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'TangibleBookValue' in df.columns and 'marketCap' in df.columns:
            return df['TangibleBookValue'] / (df['marketCap'] + EPS)
        return pd.Series(np.nan, index=df.index)