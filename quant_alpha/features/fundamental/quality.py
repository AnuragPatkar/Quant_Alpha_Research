"""
Fundamental Quality Factors
===========================
Quantitative metrics assessing the profitability, safety, and operational efficiency of a firm.

Purpose
-------
This module constructs alpha factors based on the "Quality Minus Junk" (QMJ)
investment philosophy. High-quality firms typically exhibit:
1. **Profitability**: High margins and returns on capital (ROE/ROA).
2. **Safety**: Low financial leverage and credit risk.
3. **Stability**: Low beta and earnings volatility.
4. **Earnings Quality**: Cash-driven rather than accrual-driven earnings.

Usage
-----
Factors are registered with the `FactorRegistry` and computed over standardized
fundamental data frames.

.. code-block:: python

    registry = FactorRegistry()
    quality_factor = registry.get('qual_roe')
    signals = quality_factor.compute(fundamentals_df)

Importance
----------
- **Alpha Preservation**: Quality factors act as a defensive ballast during
  market downturns, often outperforming in "flight-to-quality" regimes.
- **Forensic Accounting**: The Accruals Ratio helps detect earnings manipulation
  or deteriorating earnings quality before it impacts the top line.

Tools & Frameworks
------------------
- **Pandas**: Vectorized arithmetic for ratio derivation.
- **NumPy**: Robust handling of division-by-zero using machine epsilon.
- **FactorRegistry**: Dynamic discovery and instantiation of factor classes.
"""

from ..registry import FactorRegistry
from .utils import SingleColumnFactor, RatioFactor, FundamentalColumnValidator
from ..base import FundamentalFactor, EPS
import pandas as pd
import numpy as np
from config.logging_config import logger

# ==================== 1. PROFITABILITY ====================

@FactorRegistry.register()
class ROE(SingleColumnFactor):
    """
    Return on Equity (ROE).
    
    Measures the profitability of a business in relation to the equity.
    $$ ROE = \frac{\text{Net Income}}{\text{Shareholders' Equity}} $$
    """
    def __init__(self): super().__init__('qual_roe', 'roe', description='Return on Equity')

@FactorRegistry.register()
class ROA(SingleColumnFactor):
    """
    Return on Assets (ROA).
    
    Indicates how efficiently a company uses its assets to generate earnings.
    $$ ROA = \frac{\text{Net Income}}{\text{Total Assets}} $$
    """
    def __init__(self): super().__init__('qual_roa', 'roa', description='Return on Assets')

@FactorRegistry.register()
class GrossMargin(SingleColumnFactor):
    """
    Gross Profit Margin.
    
    $$ \text{Margin} = \frac{\text{Revenue} - \text{COGS}}{\text{Revenue}} $$
    """
    def __init__(self): super().__init__('qual_gross_margin', 'gross_margin', description='Gross Margin')
    
@FactorRegistry.register()
class OperatingMargin(SingleColumnFactor):
    """
    Operating Margin.
    
    $$ \text{Margin} = \frac{\text{Operating Income}}{\text{Revenue}} $$
    """
    def __init__(self): super().__init__('qual_op_margin', 'op_margin', description='Operating Margin')

@FactorRegistry.register()
class EBITDAMargin(SingleColumnFactor):
    """
    EBITDA Margin.
    
    A measure of a company's operating profitability as a percentage of its revenue.
    """
    def __init__(self): super().__init__('qual_ebitda_margin', 'ebitda_margin', description='EBITDA Margin')

@FactorRegistry.register()
class ProfitMargin(SingleColumnFactor):
    """
    Net Profit Margin.
    
    $$ \text{Margin} = \frac{\text{Net Income}}{\text{Revenue}} $$
    """
    def __init__(self): super().__init__('qual_profit_margin', 'profit_margin', description='Net Profit Margin')
   
# ==================== 2. SAFETY & LIQUIDITY ====================

@FactorRegistry.register()
class LowLeverage(SingleColumnFactor):
    """
    Low Financial Leverage (Safety Factor).
    
    Inverted Debt-to-Equity ratio. Higher score indicates a safer (less levered) balance sheet.
    """
    def __init__(self): 
        super().__init__('qual_low_leverage', 'debt_equity', invert=True, description='Inverted Debt/Equity')

@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    """
    Current Ratio.
    
    Standard measure of short-term liquidity.
    """
    def __init__(self): super().__init__('qual_current_ratio', 'current_ratio', description='Current Ratio')

@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    """
    Quick Ratio.
    
    Strict liquidity measure excluding inventory.
    """
    def __init__(self): super().__init__('qual_quick_ratio', 'quick_ratio', description='Quick Ratio')

# ==================== 3. STABILITY & GROWTH ====================

@FactorRegistry.register()
class LowBeta(SingleColumnFactor):
    """
    Low Beta (Low Volatility Factor).
    
    Inverted Beta. Capitalizes on the "Low Volatility Anomaly" where safer stocks 
    often offer superior risk-adjusted returns.
    """
    def __init__(self): 
        super().__init__('qual_low_beta', 'beta', invert=True, description='Inverted Beta')

# ==================== 4. EFFICIENCY (Derived) ====================

@FactorRegistry.register()
class FCFConversion(RatioFactor):
    """
    FCF Conversion Ratio.
    
    Measures how efficiently Operating Cash Flow is converted into Free Cash Flow.
    $$ \text{Ratio} = \frac{\text{FCF}}{\text{OCF}} $$
    """
    def __init__(self): 
        super().__init__('qual_fcf_conversion', num_key='fcf', den_key='ocf', description='FCF / OCF')

@FactorRegistry.register()
class AccrualsRatio(FundamentalFactor):
    """
    Sloan Accruals Ratio (Inverted).
    
    Quantifies the degree to which earnings are driven by non-cash accounting adjustments.
    Based on Richard Sloan's anomaly: high accruals predict lower future returns.
    
    Formula:
    $$ \text{Accruals} = \frac{\text{Net Income} - \text{OCF}}{\text{Total Assets}} $$
    
    Note: We invert the score so that **Higher is Better** (Lower Accruals).
    """
    def __init__(self):
        super().__init__(name='qual_accruals', description='Inverted Accruals Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for component derivation
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        roa_col = FundamentalColumnValidator.find_column(df, 'roa')
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        
        if ocf_col and roa_col and mc_col and pe_col:
            # 1. Derivation: Net Income = Market Cap / PE Ratio
            # (Implicitly handling per-share vs total scaling via ratios)
            net_income = df[mc_col] / (df[pe_col] + EPS)
            
            # 2. Derivation: Total Assets = Net Income / ROA
            total_assets = net_income / (df[roa_col] + EPS)
            
            # 3. Calculation: Accruals = (Net Income - OCF) / Total Assets
            accruals_ratio = (net_income - df[ocf_col]) / (total_assets + EPS)
            
            # 4. Normalization: Invert sign. Low Accruals -> Positive Score.
            return -1.0 * accruals_ratio
            
        logger.warning(f"⚠️  {self.name}: Missing columns for Accruals")
        return pd.Series(np.nan, index=df.index)




    