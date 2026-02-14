"""
Quality Factors (Production Grade - Verified Data Edition)
Focus: Only factors supported by 'fundamentals.parquet'.

Active Factors (11):
1. ROE (Profitability)
2. ROA (Profitability)
3. Gross Margin (Profitability)
4. Operating Margin (Profitability)
5. Low Leverage (Safety)
6. Current Ratio (Liquidity)
7. Quick Ratio (Liquidity)
8. Low Beta (Stability)
9. Earnings Growth (Growth Quality)  
10. Revenue Growth (Growth Quality)  
11. FCF Conversion (Efficiency)      
"""

import numpy as np
import pandas as pd
from typing import Optional
from config.logging_config import logger
from ..registry import FactorRegistry
from ..base import FundamentalFactor, EPS
from config.mappings import COLUMN_MAPPINGS  

class QualityColumnValidator:
    """
    Helper to find quality-related columns using centralized mappings.
    """
    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        # 1. Direct Check
        if key in df.columns: return key
        # 2. Mapping Check
        if key in COLUMN_MAPPINGS:
            for variant in COLUMN_MAPPINGS[key]:
                if variant in df.columns: return variant
        # 3. Case-Insensitive
        col_map_lower = {c.lower(): c for c in df.columns}
        if key.lower() in col_map_lower: return col_map_lower[key.lower()]
        return None

# ==================== SMART BASE CLASSES (The Engine) ====================

class SingleColumnFactor(FundamentalFactor):
    """
    Parent class for factors that just retrieve 1 column.
    Handles lookup, optional inversion (for Risk metrics), and logging.
    """
    def __init__(self, name: str, col_key: str, invert: bool = False, description: str = ""):
        super().__init__(name=name, description=description)
        self.col_key = col_key
        self.invert = invert

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = QualityColumnValidator.find_column(df, self.col_key)
        if col:
            val = df[col]
            return -1.0 * val if self.invert else val
        
        logger.warning(f"⚠️  {self.name}: Missing '{self.col_key}' column")
        return pd.Series(np.nan, index=df.index)

class RatioFactor(FundamentalFactor):
    """
    Parent class for factors calculated as A / B.
    """
    def __init__(self, name: str, num_key: str, den_key: str, description: str = ""):
        super().__init__(name=name, description=description)
        self.num_key = num_key
        self.den_key = den_key

    def compute(self, df: pd.DataFrame) -> pd.Series:
        num = QualityColumnValidator.find_column(df, self.num_key)
        den = QualityColumnValidator.find_column(df, self.den_key)
        
        if num and den:
            # Handle division by zero safely using EPS (consistent with value.py)
            return df[num] / (df[den] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing {self.num_key} or {self.den_key}")
        return pd.Series(np.nan, index=df.index)

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

@FactorRegistry.register()
class EarningsGrowth(SingleColumnFactor):
    def __init__(self): super().__init__('qual_earnings_growth', 'earnings_growth', description='Earnings Growth')

@FactorRegistry.register()
class RevenueGrowth(SingleColumnFactor):
    def __init__(self): super().__init__('qual_rev_growth', 'rev_growth', description='Revenue Growth')
    
# ==================== 4. EFFICIENCY (Derived) ====================

@FactorRegistry.register()
class FCFConversion(RatioFactor):
    def __init__(self): 
        super().__init__('qual_fcf_conversion', num_key='fcf', den_key='ocf', description='FCF / OCF')




    