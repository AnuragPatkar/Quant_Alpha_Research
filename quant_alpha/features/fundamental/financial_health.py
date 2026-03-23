"""
Financial Health & Solvency Factors
====================================
Quantitative metrics assessing leverage, liquidity, and long-term viability.

FIXES:
  BUG-027: CashToDebtRatio, NetDebtToEBITDA, DebtToRevenue, and EBITDAToDebt
           were all checking for raw column names ('total_cash', 'total_debt', etc.)
           directly via `if 'col' not in df.columns`, then accessing df['col']
           directly. This bypassed FundamentalColumnValidator entirely and would
           always fail when the DataManager/mappings.py had renamed columns.
           All four factors now use FundamentalColumnValidator.find_column().
"""

from ..registry import FactorRegistry
from .utils import SingleColumnFactor, RatioFactor, FundamentalColumnValidator
import pandas as pd
import numpy as np
from ..base import FundamentalFactor, EPS


@FactorRegistry.register()
class DebtToEquity(SingleColumnFactor):
    """
    Debt-to-Equity Ratio (Inverted).
    Score = -1 × (Total Debt / Total Equity)
    Higher = safer (lower leverage).
    """
    def __init__(self):
        super().__init__(
            'health_debt_to_equity', 'debt_equity',
            invert=True, description='Debt to Equity (inverted)'
        )


@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    """
    Current Ratio = Current Assets / Current Liabilities
    Standard short-term liquidity measure.
    """
    def __init__(self):
        super().__init__(
            'health_current_ratio', 'current_ratio',
            description='Current Ratio'
        )


@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    """
    Quick Ratio (Acid-Test) = (Current Assets - Inventory) / Current Liabilities
    """
    def __init__(self):
        super().__init__(
            'health_quick_ratio', 'quick_ratio',
            description='Quick Ratio'
        )


@FactorRegistry.register()
class CashToDebtRatio(FundamentalFactor):
    """
    Cash to Debt Ratio = Total Cash / Total Debt
    Higher = lower financial risk.

    FIX BUG-027: Was checking raw column names directly.
    Now uses FundamentalColumnValidator.find_column() so that mapped/aliased
    column names (e.g. 'End Cash Position' → 'total_cash') are resolved correctly.
    """
    def __init__(self):
        super().__init__(
            name='health_cash_to_debt',
            description='Cash to Debt Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # FIX BUG-027: use validator instead of direct column access
        cash_col = FundamentalColumnValidator.find_column(df, 'total_cash')
        debt_col = FundamentalColumnValidator.find_column(df, 'total_debt')

        if not cash_col or not debt_col:
            return pd.Series(np.nan, index=df.index)

        # Replace exact zeros in debt to avoid Inf; NaN propagates naturally
        denom  = df[debt_col].replace(0, np.nan)
        result = df[cash_col] / (denom + EPS)

        # Cap at 2.0 to dampen the impact of cash-rich outliers
        return result.clip(upper=2.0)


@FactorRegistry.register()
class NetDebtToEBITDA(FundamentalFactor):
    """
    Net Debt / EBITDA = (Total Debt - Total Cash) / EBITDA
    Standard leverage metric.

    FIX BUG-027: Same raw-column-name bypass fixed.
    """
    def __init__(self):
        super().__init__(
            name='health_net_debt_ebitda',
            description='Net Debt to EBITDA'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # FIX BUG-027: use validator instead of direct column access
        debt_col   = FundamentalColumnValidator.find_column(df, 'total_debt')
        cash_col   = FundamentalColumnValidator.find_column(df, 'total_cash')
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda')

        if not debt_col or not cash_col or not ebitda_col:
            return pd.Series(np.nan, index=df.index)

        net_debt = df[debt_col] - df[cash_col]
        ebitda   = df[ebitda_col].replace(0, np.nan)
        result   = net_debt / (ebitda + EPS)

        # Clip to [-1, 10]: -1 = cash > debt (net cash), 10 = highly distressed
        return result.clip(lower=-1.0, upper=10.0)


@FactorRegistry.register()
class DebtToRevenue(FundamentalFactor):
    """
    Debt to Revenue = Total Debt / Total Revenue
    Useful for unprofitable growth companies where EBITDA is negative.

    FIX BUG-027: Same raw-column-name bypass fixed.
    """
    def __init__(self):
        super().__init__(
            name='health_debt_to_revenue',
            description='Debt to Revenue Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # FIX BUG-027: use validator instead of direct column access
        debt_col = FundamentalColumnValidator.find_column(df, 'total_debt')
        rev_col  = FundamentalColumnValidator.find_column(df, 'total_revenue')

        if not debt_col or not rev_col:
            return pd.Series(np.nan, index=df.index)

        revenue = df[rev_col].replace(0, np.nan)
        result  = df[debt_col] / (revenue + EPS)

        # Cap at 5× revenue (high distress zone)
        return result.clip(upper=5.0)


@FactorRegistry.register()
class EBITDAToDebt(FundamentalFactor):
    """
    EBITDA to Debt = EBITDA / Total Debt
    Inverse leverage metric; higher = safer.

    FIX BUG-027: Same raw-column-name bypass fixed.
    """
    def __init__(self):
        super().__init__(
            name='health_ebitda_to_debt',
            description='EBITDA to Debt Ratio'
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # FIX BUG-027: use validator instead of direct column access
        ebitda_col = FundamentalColumnValidator.find_column(df, 'ebitda')
        debt_col   = FundamentalColumnValidator.find_column(df, 'total_debt')

        if not ebitda_col or not debt_col:
            return pd.Series(np.nan, index=df.index)

        debt   = df[debt_col].replace(0, np.nan)
        result = df[ebitda_col] / (debt + EPS)

        # Cap at 10× (extremely strong debt coverage)
        return result.clip(upper=10.0)