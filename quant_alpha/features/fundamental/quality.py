"""
Quality Factors (Production Grade - Verified Data Edition)
Focus: Only factors supported by 'fundamentals.parquet'.

Active Factors (10):
1. ROE (Profitability)
2. ROA (Profitability)
3. Gross Margin (Profitability)
4. Operating Margin (Profitability)
5. Low Leverage (Safety)
6. Current Ratio (Liquidity)
7. Quick Ratio (Liquidity)
8. Low Beta (Stability)
9. FCF Conversion (Efficiency)      
10. Accruals Ratio (Earnings Quality)
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
    def __init__(self): super().__init__('qual_roe', 'roe', description='Return on Equity')

@FactorRegistry.register()
class ROA(SingleColumnFactor):
    def __init__(self): super().__init__('qual_roa', 'roa', description='Return on Assets')

@FactorRegistry.register()
class GrossMargin(SingleColumnFactor):
    def __init__(self): super().__init__('qual_gross_margin', 'gross_margin', description='Gross Margin')
    
@FactorRegistry.register()
class OperatingMargin(SingleColumnFactor):
    def __init__(self): super().__init__('qual_op_margin', 'op_margin', description='Operating Margin')

@FactorRegistry.register()
class EBITDAMargin(SingleColumnFactor):
    def __init__(self): super().__init__('qual_ebitda_margin', 'ebitda_margin', description='EBITDA Margin')

@FactorRegistry.register()
class ProfitMargin(SingleColumnFactor):
    def __init__(self): super().__init__('qual_profit_margin', 'profit_margin', description='Net Profit Margin')
   
# ==================== 2. SAFETY & LIQUIDITY ====================

@FactorRegistry.register()
class LowLeverage(SingleColumnFactor):
    def __init__(self): 
        super().__init__('qual_low_leverage', 'debt_equity', invert=True, description='Inverted Debt/Equity')

@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    def __init__(self): super().__init__('qual_current_ratio', 'current_ratio', description='Current Ratio')

@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    def __init__(self): super().__init__('qual_quick_ratio', 'quick_ratio', description='Quick Ratio')

# ==================== 3. STABILITY & GROWTH ====================

@FactorRegistry.register()
class LowBeta(SingleColumnFactor):
    def __init__(self): 
        super().__init__('qual_low_beta', 'beta', invert=True, description='Inverted Beta')

# ==================== 4. EFFICIENCY (Derived) ====================

@FactorRegistry.register()
class FCFConversion(RatioFactor):
    def __init__(self): 
        super().__init__('qual_fcf_conversion', num_key='fcf', den_key='ocf', description='FCF / OCF')

@FactorRegistry.register()
class AccrualsRatio(FundamentalFactor):
    """
    Sloan Accruals Ratio = (Net Income - OCF) / Total Assets
    Lower is Better (High Accruals = Low Quality Earnings).
    We invert it so Higher Score = Better Quality.
    """
    def __init__(self):
        super().__init__(name='qual_accruals', description='Inverted Accruals Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        roa_col = FundamentalColumnValidator.find_column(df, 'roa')
        mc_col = FundamentalColumnValidator.find_column(df, 'market_cap')
        pe_col = FundamentalColumnValidator.find_column(df, 'pe_ratio')
        
        if ocf_col and roa_col and mc_col and pe_col:
            # 1. Derive Net Income = Market Cap / PE
            net_income = df[mc_col] / (df[pe_col] + EPS)
            
            # 2. Derive Total Assets = Net Income / ROA
            total_assets = net_income / (df[roa_col] + EPS)
            
            # 3. Accruals = (Net Income - OCF) / Assets
            accruals_ratio = (net_income - df[ocf_col]) / (total_assets + EPS)
            
            # 4. Invert because Lower Accruals is Better
            return -1.0 * accruals_ratio
            
        logger.warning(f"⚠️  {self.name}: Missing columns for Accruals")
        return pd.Series(np.nan, index=df.index)




    