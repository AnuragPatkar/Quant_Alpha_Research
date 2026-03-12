"""
Fundamental Value Factors
=========================
Quantitative metrics assessing the relative valuation of assets ("Cheap" vs "Expensive").

Purpose
-------
This module constructs alpha factors based on the Value investing philosophy (e.g., Fama-French HML).
It standardizes valuation ratios by inverting traditional "Price-to-X" multiples (like P/E) into
"Yields" (like E/P). This transformation ensures that:
1.  **Linearity**: Yields are linear and additive, whereas multiples have singularities as the denominator approaches zero.
2.  **Directionality**: Higher scores consistently represent "cheaper" (better) valuation, aligning with long-only alpha ranking systems.
1.  **Linearity**: Yields are linear and additive, preventing singularities when the denominator (Earnings) approaches zero.
2.  **Directionality**: Higher scores consistently represent "cheaper" (better) valuation, aligning with standard alpha ranking systems (Long Top Decile).

Usage
-----
Factors are registered with the `FactorRegistry` and computed over standardized fundamental data.

.. code-block:: python

    registry = FactorRegistry()
    value_factor = registry.get('val_earnings_yield')
    signals = value_factor.compute(fundamentals_df)

Importance
----------
- **Alpha Generation**: Value is one of the most robust and persistent style premia in finance.
- **Risk Assessment**: High multiples (low yields) often indicate overvaluation or excessive growth expectations.
- **Robustness**: Inverted ratios handle negative earnings (negative yield) more gracefully than negative P/E ratios in sorting algorithms.
- **Statistical Robustness**: Inverted ratios handle negative earnings (negative yield) more gracefully than negative P/E ratios in sorting algorithms.

Tools & Frameworks
------------------
- **Pandas**: Vectorized DataFrame operations for high-throughput ratio calculation ($O(n)$).
- **NumPy**: Handling of outliers and numerical stability constants ($EPS$).
- **Pandas**: Vectorized DataFrame operations for high-throughput ratio calculation ($O(N)$).
- **NumPy**: Handling of outliers and numerical stability constants ($\epsilon$).
- **FactorRegistry**: Dynamic discovery and instantiation of factor classes.
"""

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
    """
    Earnings Yield (E/P).
    
    Inverse of the Price-to-Earnings ratio. Represents the percentage of earnings
    per dollar invested.
    
    Formula:
    $$ E/P = \frac{1}{\text{P/E}} \approx \frac{\text{EPS}}{\text{Price}} $$
    """
    def __init__(self):
        super().__init__('val_earnings_yield', description='Earnings Yield (1/PE or EPS/Price)')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert P/E Ratio (Vendor Provided)
        # Strategy 1: Invert Vendor-Precomputed P/E Ratio
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        if pe_col:
            pe = df[pe_col].copy()
            # Winsorization: Clip P/E to [0.1, 200] to avoid division-by-zero or 
            # generating massive yields from near-zero denominators (e.g., P/E=0.01).
            # Outlier Mitigation: Clip P/E to [0.1, 200] to prevent DivisionByZero 
            # and suppress massive yields from near-zero denominators (e.g., P/E=0.01).
            pe = pe.clip(lower=0.1, upper=200)
            return 1.0 / (pe + EPS)
        
        # Strategy 2: Derive from EPS and Price
        eps_col = FundamentalColumnValidator.find_column(df, 'eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if eps_col and price_col:
            return df[eps_col] / (df[price_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing P/E or (EPS & Price)")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ForwardEarningsYield(FundamentalFactor):
    """
    Forward Earnings Yield.
    
    Valuation based on analyst consensus estimates for the next fiscal period.
    Valuation based on analyst consensus estimates for the next fiscal period (NTM).
    
    Formula:
    $$ Yield_{fwd} = \frac{1}{\text{Forward P/E}} $$
    """
    def __init__(self):
        super().__init__(name='val_forward_earnings_yield', description='Forward Earnings Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert Forward P/E
        pe_col = FundamentalColumnValidator.find_column(df, 'forward_pe')
        if pe_col:
            return 1.0 / (df[pe_col] + EPS)
            
        # Strategy 2: Derive from Forward EPS and Price
        fwd_eps_col = FundamentalColumnValidator.find_column(df, 'fwd_eps')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if fwd_eps_col and price_col:
            return df[fwd_eps_col] / (df[price_col] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing Forward P/E or Forward EPS")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class BookYield(FundamentalFactor):
    """
    Book Yield (B/P).
    
    Inverse of the Price-to-Book ratio. High book yield implies the stock is
    Inverse of the Price-to-Book ratio. High Book Yield implies the stock is
    trading closer to (or below) the value of its net assets.
    
    Formula:
    $$ B/P = \frac{1}{\text{P/B}} = \frac{\text{Book Value}}{\text{Price}} $$
    """
    def __init__(self):
        super().__init__(name='val_book_yield', description='Book Yield (1/PB)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert P/B Ratio
        pb_col = FundamentalColumnValidator.find_column(df, 'price_to_book')
        if pb_col:
            return 1.0 / (df[pb_col] + EPS)
            
        # Strategy 2: Derive from Book Value and Price
        bv_col = FundamentalColumnValidator.find_column(df, 'book_value')
        price_col = FundamentalColumnValidator.find_column(df, 'price')
        if bv_col and price_col:
             return df[bv_col] / (df[price_col] + EPS)

        logger.warning(f"⚠️  {self.name}: Missing P/B data")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class FCFYield(RatioFactor):
    """
    Free Cash Flow Yield.
    
    A measure of solvency and value.
    A robust valuation metric representing the cash available to shareholders after
    capital expenditures. Unlike earnings, FCF is harder to manipulate via accounting
    accruals.
    
    Formula:
    $$ Yield_{FCF} = \frac{\text{Free Cash Flow}}{\text{Market Cap}} $$
    """
    def __init__(self):
        super().__init__('val_fcf_yield', num_key='fcf', den_key='market_cap', description='FCF Yield')

@FactorRegistry.register()
class OperatingCashFlowYield(RatioFactor):
    """
    Operating Cash Flow Yield.
    
    Focuses on core business cash generation relative to valuation.
    Measures the cash generated from core business operations relative to market valuation.
    High OCF Yield indicates the company generates significant cash per dollar of equity,
    often a sign of high quality and undervaluation.
    
    Formula:
    $$ Yield_{OCF} = \frac{\text{Operating Cash Flow}}{\text{Market Cap}} $$
    """
    def __init__(self):
        super().__init__('val_ocf_yield', num_key='ocf', den_key='market_cap', description='OCF Yield')

@FactorRegistry.register()
class DividendYield(FundamentalFactor):
    """
    Dividend Yield.
    
    Annualized dividend payments relative to price. Missing data is imputed as 0.0
    (assuming non-payer).
    Quantifies the cash return on investment from dividends alone.
    Missing data is imputed as 0.0 (assuming non-payer status), preserving the signal's
    validity for the entire universe.
    
    Formula:
    $$ Yield_{div} = \frac{\text{Annualized Dividends}}{\text{Price}} $$
    """
    def __init__(self):
        super().__init__('val_div_yield', description='Dividend Yield')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, 'dividend')
        if col:
            # Data Integrity: Treat missing dividend data as 0% yield (Non-payer)
            # Data Integrity: Treat missing dividend yield as 0.0 (Non-payer)
            # rather than dropping the row.
            return df[col].fillna(0.0)
        return pd.Series(0.0, index=df.index)

# ==================== ADVANCED YIELDS ====================

@FactorRegistry.register()
class ShareholderYield(FundamentalFactor):
    """
    Shareholder Yield.
    
    Total capital returned to shareholders via dividends and share buybacks.
    Often considered a more holistic metric than simple Dividend Yield.
    
    Formula:
    $$ Yield_{SH} = \frac{\text{Dividends} + \text{Net Buybacks}}{\text{Market Cap}} $$
    """
    def __init__(self):
        super().__init__(name='val_shareholder_yield', description='Shareholder Yield')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        mkt_cap_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        
        if not mkt_cap_col:
            logger.warning(f"⚠️  {self.name}: Missing Market Cap")
            return pd.Series(np.nan, index=df.index)
        
        total_payout = pd.Series(0.0, index=df.index)
        
        # Add Buybacks (if available)
        buyback_col = FundamentalColumnValidator.find_column(df, 'buyback')
        if buyback_col:
            total_payout += df[buyback_col].abs()
        
        # Add Cash Dividends (if available)
        div_paid_col = FundamentalColumnValidator.find_column(df, 'dividends_paid')
        if div_paid_col:
            total_payout += df[div_paid_col].abs()
        
        return total_payout / (df[mkt_cap_col] + EPS)

@FactorRegistry.register()
class EnterpriseValueFCFYield(FundamentalFactor):
    """
    Enterprise Value FCF Yield.
    
    Measures the cash flow yield available to all capital providers (Debt + Equity).
    
    Capital structure neutral.

    Formula:
    $$ Yield = \frac{\text{Free Cash Flow}}{\text{Enterprise Value}} $$
    
    Where:
    $$ EV \approx \text{Market Cap} + \text{Total Debt} $$
    """
    def __init__(self):
        super().__init__(name='val_ev_fcf_yield', description='FCF / Enterprise Value')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        fcf_col = FundamentalColumnValidator.find_column(df, 'fcf')
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        
        if fcf_col and mc_col:
            market_cap = df[mc_col]
            total_debt = pd.Series(0.0, index=df.index)
            
            # Strategy: Locate Total Debt or derive it from ratios
            # Strategy: Locate Total Debt or derive it from fundamental ratios
            td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
            de_col = FundamentalColumnValidator.find_column(df, 'debt_equity')
            pb_col = FundamentalColumnValidator.find_column(df, 'price_to_book')
            
            if td_col:
                total_debt = df[td_col].fillna(0.0)
            elif de_col and pb_col:
                # Derivation: $$ Debt = \frac{Debt}{Equity} \times \frac{Market Cap}{P/B} $$
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
    Price-to-Sales Ratio (Inverted).
    
    Also known as "Sales Yield". Harder to manipulate than earnings-based metrics.
    Inverted so that **Higher is Better** (Cheaper).
    Inverted so that **Higher is Better** (Cheaper/Undervalued).
    
    Formula:
    $$ \text{Sales Yield} = \frac{1}{\text{P/S}} = \frac{\text{Total Revenue}}{\text{Market Cap}} $$
    """
    def __init__(self):
        super().__init__(name='val_ps_ratio', description='Price-to-Sales Ratio')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert P/S Ratio
        ps_col = FundamentalColumnValidator.find_column(df, 'ps_ratio')
        if ps_col:
            return 1.0 / (df[ps_col] + EPS)
        
        # Strategy 2: Derive from components
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')
        
        if mc_col and rev_col:
            # Inverted: Revenue / Market Cap
            return df[mc_col] / (df[rev_col] + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing P/S or (Market Cap & Revenue)")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class EVtoEBITDAValuation(FundamentalFactor):
    """
    EV/EBITDA (Inverted).
    
    EBITDA Yield. A capital structure-neutral valuation metric.
    EBITDA Yield. A capital structure-neutral operating metric.
    Inverted so that **Higher is Better**.
    
    Formula:
    $$ \text{EBITDA Yield} = \frac{\text{EBITDA}}{\text{Enterprise Value}} $$
    """
    def __init__(self):
        super().__init__(name='val_ev_ebitda', description='EV/EBITDA Valuation')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Strategy 1: Invert Pre-computed EV/EBITDA
        # Strategy 1: Invert Vendor-Precomputed EV/EBITDA
        ev_ebitda_col = FundamentalColumnValidator.find_column(df, 'ev_ebitda')
        if ev_ebitda_col:
            return 1.0 / (df[ev_ebitda_col] + EPS)
        
        # Strategy 2: Derive components
        # Strategy 2: Derive from raw components
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        td_col = FundamentalColumnValidator.find_column(df, 'total_debt')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda_margin')
        rev_col = FundamentalColumnValidator.find_column(df, 'total_revenue')
        
        if mc_col and ebitda_col and rev_col:
            ev = df[mc_col].copy()
            if td_col:
                ev = ev + df[td_col].fillna(0.0)
            
            # Derivation: EBITDA = Revenue * Margin
            ebitda = df[rev_col] * df[ebitda_col]
            
            # Inverted: EBITDA / EV
            return ebitda / (ev + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing data for EV/EBITDA calculation")
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class EVtoSalesValuation(FundamentalFactor):
    """
    EV/Sales (Inverted).
    
    Enterprise Sales Yield. Useful for valuing loss-making companies.
    Inverted so that **Higher is Better**.
    
    Formula:
    $$ \text{Yield} = \frac{\text{Total Revenue}}{\text{Enterprise Value}} $$
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
            
            # Inverted: Revenue / EV
            return df[rev_col] / (ev + EPS)
        
        logger.warning(f"⚠️  {self.name}: Missing Market Cap or Revenue")
        return pd.Series(np.nan, index=df.index)