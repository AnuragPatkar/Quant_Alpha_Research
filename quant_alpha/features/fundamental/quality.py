"""
Fundamental Quality Factors
===========================
Quantitative metrics assessing profitability, safety, and operational efficiency.

FIXES:
  BUG-047: Removed duplicate CurrentRatio and QuickRatio class definitions.
           Both were previously defined here (with 'qual_' prefix) AND in
           financial_health.py (with 'health_' prefix). Both registered with
           FactorRegistry at import time, doubling these computations and
           consuming 2× memory for identical calculations.
           The 'health_current_ratio' and 'health_quick_ratio' factors in
           financial_health.py are the canonical versions. LowLeverage and
           LowBeta remain here because they are inverted (different signal
           direction) and not duplicated in financial_health.py.
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
    Return on Equity.
    Formula: Net Income / Shareholders' Equity
    """
    def __init__(self):
        super().__init__('qual_roe', 'roe', description='Return on Equity')


@FactorRegistry.register()
class ROA(SingleColumnFactor):
    """
    Return on Assets.
    Formula: Net Income / Total Assets
    """
    def __init__(self):
        super().__init__('qual_roa', 'roa', description='Return on Assets')


@FactorRegistry.register()
class GrossMargin(SingleColumnFactor):
    """
    Gross Profit Margin.
    Formula: (Revenue - COGS) / Revenue
    """
    def __init__(self):
        super().__init__('qual_gross_margin', 'gross_margin', description='Gross Margin')


@FactorRegistry.register()
class OperatingMargin(SingleColumnFactor):
    """
    Operating Margin.
    Formula: Operating Income / Revenue
    """
    def __init__(self):
        super().__init__('qual_op_margin', 'op_margin', description='Operating Margin')


@FactorRegistry.register()
class EBITDAMargin(SingleColumnFactor):
    """EBITDA Margin = EBITDA / Revenue."""
    def __init__(self):
        super().__init__('qual_ebitda_margin', 'ebitda_margin', description='EBITDA Margin')


@FactorRegistry.register()
class ProfitMargin(SingleColumnFactor):
    """
    Net Profit Margin.
    Formula: Net Income / Revenue
    """
    def __init__(self):
        super().__init__('qual_profit_margin', 'profit_margin', description='Net Profit Margin')


# ==================== 2. SAFETY & LEVERAGE ====================

@FactorRegistry.register()
class LowLeverage(SingleColumnFactor):
    """
    Low Financial Leverage (Safety Factor).
    Inverted D/E ratio: higher score = safer (less levered) balance sheet.
    """
    def __init__(self):
        super().__init__(
            'qual_low_leverage', 'debt_equity',
            invert=True, description='Inverted Debt/Equity'
        )


# FIX BUG-047: CurrentRatio and QuickRatio REMOVED from this file.
# Use health_current_ratio and health_quick_ratio from financial_health.py.
# They are identical computations with different factor names; having both
# doubles memory and computation for no additional information.


# ==================== 3. STABILITY ====================

@FactorRegistry.register()
class LowBeta(SingleColumnFactor):
    """
    Low Beta (Low Volatility Factor).
    Inverted Beta. Capitalises on the Low Volatility Anomaly.
    """
    def __init__(self):
        super().__init__(
            'qual_low_beta', 'beta',
            invert=True, description='Inverted Beta'
        )


# ==================== 4. EFFICIENCY (Derived) ====================

@FactorRegistry.register()
class FCFConversion(RatioFactor):
    """
    FCF Conversion Ratio = FCF / OCF
    Measures how efficiently Operating Cash Flow converts into Free Cash Flow.
    """
    def __init__(self):
        super().__init__(
            'qual_fcf_conversion', num_key='fcf', den_key='ocf',
            description='FCF / OCF'
        )


@FactorRegistry.register()
class AccrualsRatio(FundamentalFactor):
    """
    Sloan Accruals Ratio (Inverted).
    Formula: -(Net Income - OCF) / Total Assets
    Higher score = lower accruals = higher earnings quality.
    """
    def __init__(self):
        super().__init__(name='qual_accruals', description='Inverted Accruals Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        roa_col = FundamentalColumnValidator.find_column(df, 'roa')
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        pe_col  = FundamentalColumnValidator.find_column(df, 'pe_ratio')

        if ocf_col and roa_col and mc_col and pe_col:
            # Derivation: Net Income = Market Cap / PE Ratio
            net_income  = df[mc_col] / (df[pe_col] + EPS)
            # Derivation: Total Assets = Net Income / ROA
            total_assets = net_income / (df[roa_col] + EPS)
            # Accruals = (Net Income - OCF) / Total Assets
            accruals_ratio = (net_income - df[ocf_col]) / (total_assets + EPS)
            # Invert: Low Accruals → Positive Score
            return -1.0 * accruals_ratio

        logger.warning(f"⚠️  {self.name}: Missing columns for Accruals")
        return pd.Series(np.nan, index=df.index)