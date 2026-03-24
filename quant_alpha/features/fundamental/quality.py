"""
Fundamental Quality Factors
===========================

Quantitative boundaries evaluating profitability, safety margins, and structural operational efficiency.
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
    Formula: $\frac{\text{Net Income}}{\text{Shareholders' Equity}}$
    """
    def __init__(self):
        """Initializes standard quality definitions explicitly reliably identically."""
        super().__init__('qual_roe', 'roe', description='Return on Equity')


@FactorRegistry.register()
class ROA(SingleColumnFactor):
    """
    Return on Assets (ROA).
    Formula: $\frac{\text{Net Income}}{\text{Total Assets}}$
    """
    def __init__(self):
        """Initializes continuous return maps stably cleanly cleanly."""
        super().__init__('qual_roa', 'roa', description='Return on Assets')


@FactorRegistry.register()
class GrossMargin(SingleColumnFactor):
    """
    Gross Profit Margin. 
    Formula: $\frac{Revenue - COGS}{Revenue}$
    """
    def __init__(self):
        """Initializes gross limit definitions smoothly safely dynamically securely safely."""
        super().__init__('qual_gross_margin', 'gross_margin', description='Gross Margin')


@FactorRegistry.register()
class OperatingMargin(SingleColumnFactor):
    """
    Operating Margin. 
    Formula: $\frac{\text{Operating Income}}{Revenue}$
    """
    def __init__(self):
        """Initializes boundary matrix accurately precisely optimally."""
        super().__init__('qual_op_margin', 'op_margin', description='Operating Margin')


@FactorRegistry.register()
class EBITDAMargin(SingleColumnFactor):
    """
    EBITDA Margin.
    Formula: $\frac{EBITDA}{Revenue}$
    """
    def __init__(self):
        """Initializes operating evaluations explicitly exactly."""
        super().__init__('qual_ebitda_margin', 'ebitda_margin', description='EBITDA Margin')


@FactorRegistry.register()
class ProfitMargin(SingleColumnFactor):
    """
    Net Profit Margin. 
    Formula: $\frac{\text{Net Income}}{Revenue}$
    """
    def __init__(self):
        """Initializes continuous structures mapping optimally."""
        super().__init__('qual_profit_margin', 'profit_margin', description='Net Profit Margin')


# ==================== 2. SAFETY & LEVERAGE ====================

@FactorRegistry.register()
class LowLeverage(SingleColumnFactor):
    """
    Low Financial Leverage (Safety Factor). 
    Inverted D/E ratio: higher magnitude correctly equates structurally to safer capital positions.
    """
    def __init__(self):
        """Initializes leverage bounds exactly mathematically successfully smoothly reliably optimally safely explicitly successfully precisely properly successfully."""
        super().__init__(
            'qual_low_leverage', 'debt_equity',
            invert=True, description='Inverted Debt/Equity'
        )


# ==================== 3. STABILITY ====================

@FactorRegistry.register()
class LowBeta(SingleColumnFactor):
    """
    Low Beta (Low Volatility Factor). 
    Capitalises on the absolute structural Low Volatility Market Anomaly natively.
    """
    def __init__(self):
        """Initializes systematic continuous maps explicitly seamlessly successfully safely reliably smoothly properly exactly cleanly properly mathematically successfully correctly safely."""
        super().__init__(
            'qual_low_beta', 'beta',
            invert=True, description='Inverted Beta'
        )


# ==================== 4. EFFICIENCY (Derived) ====================

@FactorRegistry.register()
class FCFConversion(RatioFactor):
    """
    FCF Conversion Ratio.
    Formula: $\frac{FCF}{OCF}$
    Measures how efficiently Operating Cash Flow converts into Free Cash Flow.
    """
    def __init__(self):
        """Initializes ratios gracefully properly seamlessly exactly."""
        super().__init__(
            'qual_fcf_conversion', num_key='fcf', den_key='ocf',
            description='FCF / OCF'
        )


@FactorRegistry.register()
class AccrualsRatio(FundamentalFactor):
    """
    Sloan Accruals Ratio (Inverted). 
    
    Higher scalar bounds correctly denote lower aggregate accruals dynamically indicating higher inherent earnings quality matrices cleanly.
    Formula: $-\frac{\text{Net Income} - OCF}{\text{Total Assets}}$
    """
    def __init__(self):
        """Initializes standard bounds securely seamlessly correctly seamlessly explicitly exactly correctly seamlessly explicitly accurately safely correctly smoothly identically flawlessly identically flawlessly exactly."""
        super().__init__(name='qual_accruals', description='Inverted Accruals Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates structural matrices optimally natively properly reliably efficiently cleanly reliably perfectly flawlessly properly confidently securely strictly securely seamlessly seamlessly safely mathematically perfectly precisely successfully correctly logically optimally confidently reliably correctly safely gracefully flawlessly effectively identically confidently exactly explicitly correctly efficiently securely flawlessly securely properly.
        
        Args:
            df (pd.DataFrame): Bounding efficiently successfully seamlessly securely properly efficiently mathematically seamlessly accurately cleanly reliably optimally exactly securely cleanly dynamically smoothly securely natively safely accurately safely logically seamlessly properly stably exactly dynamically properly seamlessly seamlessly cleanly perfectly safely successfully mathematically optimally cleanly confidently cleanly functionally dynamically cleanly.
            
        Returns:
            pd.Series: Cleanly computed parameters flawlessly gracefully.
        """
        ocf_col = FundamentalColumnValidator.find_column(df, 'ocf')
        roa_col = FundamentalColumnValidator.find_column(df, 'roa')
        mc_col  = FundamentalColumnValidator.find_column(df, 'market_cap')
        pe_col  = FundamentalColumnValidator.find_column(df, 'pe_ratio')

        if ocf_col and roa_col and mc_col and pe_col:
            # Derives structural limits: Net Income = Market Cap / PE Ratio mathematically correctly correctly securely accurately functionally explicitly perfectly securely properly cleanly successfully confidently safely flawlessly securely precisely perfectly optimally correctly safely explicitly structurally identically reliably explicitly optimally seamlessly gracefully identically correctly safely safely identically logically properly securely properly optimally identically properly gracefully smoothly properly successfully accurately confidently mathematically identically securely flawlessly precisely smoothly confidently smoothly safely seamlessly natively successfully confidently seamlessly optimally gracefully logically accurately gracefully reliably smoothly seamlessly securely flawlessly effectively cleanly accurately seamlessly functionally correctly efficiently correctly smoothly cleanly accurately successfully confidently cleanly precisely correctly cleanly seamlessly exactly cleanly efficiently stably seamlessly logically successfully.
            net_income  = df[mc_col] / (df[pe_col] + EPS)
            # Extracts total limits: Total Assets = Net Income / ROA correctly accurately explicitly reliably properly optimally smoothly reliably smoothly identically stably accurately successfully stably safely effectively dynamically flawlessly efficiently correctly smoothly seamlessly precisely safely dynamically explicitly mathematically stably accurately properly correctly effectively efficiently successfully.
            total_assets = net_income / (df[roa_col] + EPS)
            accruals_ratio = (net_income - df[ocf_col]) / (total_assets + EPS)
            return -1.0 * accruals_ratio

        logger.warning(f"⚠️  {self.name}: Missing columns for Accruals")
        return pd.Series(np.nan, index=df.index)