"""
Financial Health Factors
Leverage, liquidity, and solvency metrics.

Active factors:
- Debt to Equity (Inverted): measures financial leverage safety
- Current Ratio: measures short-term liquidity
- Quick Ratio: measures strict liquidity (excluding inventory)
- Cash to Debt Ratio: measures immediate liquidity vs debt obligations
- Net Debt to EBITDA: measures financial leverage relative to profitability
- Debt to Revenue: measures debt as percentage of revenue
- EBITDA to Debt: measures annual profitability available to pay debt
"""

from ..registry import FactorRegistry
from .utils import SingleColumnFactor, RatioFactor
import pandas as pd
import numpy as np
from ..base import FundamentalFactor

@FactorRegistry.register()
class DebtToEquity(SingleColumnFactor):
    def __init__(self):
        super().__init__('health_debt_to_equity', 'debt_equity', invert=True, description='Debt to Equity (inverted)')

@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    def __init__(self):
        super().__init__('health_current_ratio', 'current_ratio', description='Current Ratio')

@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    def __init__(self):
        super().__init__('health_quick_ratio', 'quick_ratio', description='Quick Ratio')


@FactorRegistry.register()
class CashToDebtRatio(RatioFactor):
    """
    Cash to Debt Ratio = Total Cash / Total Debt
    Measures ability to immediately pay off debt with available cash.
    Higher is better (indicator of low financial risk).
    
    Formula: total_cash / total_debt
    """
    def __init__(self):
        super().__init__('health_cash_to_debt', num_key='total_cash', den_key='total_debt',
                        description='Cash to Debt Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'total_cash' not in df.columns or 'total_debt' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['total_cash'] / df['total_debt']
        # Cap at 2.0 for extreme values
        result = result.clip(upper=2.0)
        return result


@FactorRegistry.register()
class NetDebtToEBITDA(RatioFactor):
    """
    Net Debt to EBITDA = (Total Debt - Total Cash) / EBITDA
    Measures financial leverage relative to cash generation ability.
    Industry standard metric - lower is better.
    
    Interpretation:
    - < 2.0: Very healthy leverage
    - 2.0-3.0: Moderate leverage
    - 3.0-5.0: Elevated leverage
    - > 5.0: High leverage/risk
    
    Formula: (total_debt - total_cash) / ebitda
    """
    def __init__(self):
        super().__init__('health_net_debt_ebitda', num_key='net_debt', den_key='ebitda',
                        description='Net Debt to EBITDA')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'total_debt' not in df.columns or 'total_cash' not in df.columns or 'ebitda' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        net_debt = df['total_debt'] - df['total_cash']
        result = net_debt / df['ebitda']
        # Handle negative values (company has more cash than debt)
        result = result.clip(lower=-1.0, upper=10.0)
        return result


@FactorRegistry.register()
class DebtToRevenue(RatioFactor):
    """
    Debt to Revenue Ratio = Total Debt / Total Revenue
    Measures debt as a percentage of annual revenue.
    Lower is better - shows how many years of revenue needed to pay debt.
    
    Interpretation:
    - < 1.0: Low debt relative to revenue
    - 1.0-2.0: Moderate debt
    - 2.0-3.0: High debt
    - > 3.0: Very high debt
    
    Formula: total_debt / total_revenue
    """
    def __init__(self):
        super().__init__('health_debt_to_revenue', num_key='total_debt', den_key='total_revenue',
                        description='Debt to Revenue Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'total_debt' not in df.columns or 'total_revenue' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['total_debt'] / df['total_revenue']
        return result.clip(upper=5.0)  # Cap extreme values


@FactorRegistry.register()
class EBITDAToDebt(RatioFactor):
    """
    EBITDA to Debt Ratio = EBITDA / Total Debt
    Measures annual profit generation ability relative to total debt.
    Higher is better - inverse of Net Debt to EBITDA.
    
    Shows how many years of full EBITDA needed to pay off total debt.
    
    Formula: ebitda / total_debt
    """
    def __init__(self):
        super().__init__('health_ebitda_to_debt', num_key='ebitda', den_key='total_debt',
                        description='EBITDA to Debt Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'ebitda' not in df.columns or 'total_debt' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['ebitda'] / df['total_debt']
        return result.clip(upper=10.0)  # Cap extreme values
