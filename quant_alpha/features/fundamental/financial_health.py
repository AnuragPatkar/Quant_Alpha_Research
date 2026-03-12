"""
Financial Health & Solvency Factors
===================================
Quantitative metrics assessing a firm's leverage, liquidity, and long-term viability.

Purpose
-------
This module constructs factors that evaluate the balance sheet strength of a company.
It focuses on two critical dimensions of risk:
1. **Solvency**: The ability to meet long-term debt obligations (e.g., Debt/Equity, Net Debt/EBITDA).
2. **Liquidity**: The ability to meet short-term liabilities (e.g., Current Ratio, Quick Ratio).

Usage
-----
These factors are registered with the `FactorRegistry` and are typically used as
"Quality" signals or negative screens (exclusionary filters) in portfolio construction.

.. code-block:: python

    registry = FactorRegistry()
    health_factor = registry.get('health_net_debt_ebitda')
    signals = health_factor.compute(fundamentals_df)

Importance
----------
- **Risk Mitigation**: High leverage is a primary predictor of bankruptcy and
  equity dilution events.
- **Quality Factor**: "Quality-Minus-Junk" (QMJ) strategies rely heavily on
  solvency metrics to identify robust firms.
- **Regime Sensitivity**: These factors effectively stratify performance during
  credit contractions and rising interest rate environments.

Tools & Frameworks
------------------
- **Pandas**: Vectorized DataFrame operations for ratio calculations.
- **NumPy**: Efficient handling of `NaN` propagation and numerical clipping.
- **FactorRegistry**: Decorator-based registration system.
"""

from ..registry import FactorRegistry
from .utils import SingleColumnFactor, RatioFactor
import pandas as pd
import numpy as np
from ..base import FundamentalFactor

@FactorRegistry.register()
class DebtToEquity(SingleColumnFactor):
    """
    Debt-to-Equity Ratio (Inverted).
    
    Measures financial leverage. We invert the sign ($ \times -1 $) so that
    higher scores represent safer (lower) leverage, aligning with the
    "Higher is Better" convention of alpha factors.
    
    Formula:
    $$ Score = -1 \times \frac{\text{Total Debt}}{\text{Total Equity}} $$
    """
    def __init__(self):
        super().__init__('health_debt_to_equity', 'debt_equity', invert=True, description='Debt to Equity (inverted)')

@FactorRegistry.register()
class CurrentRatio(SingleColumnFactor):
    """
    Current Ratio.
    
    A broad measure of short-term liquidity.
    $$ \text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}} $$
    """
    def __init__(self):
        super().__init__('health_current_ratio', 'current_ratio', description='Current Ratio')

@FactorRegistry.register()
class QuickRatio(SingleColumnFactor):
    """
    Quick Ratio (Acid-Test).
    
    A stringent measure of liquidity that excludes inventory.
    $$ \text{Quick Ratio} = \frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}} $$
    """
    def __init__(self):
        super().__init__('health_quick_ratio', 'quick_ratio', description='Quick Ratio')


@FactorRegistry.register()
class CashToDebtRatio(RatioFactor):
    """
    Cash to Debt Ratio = Total Cash / Total Debt
    
    Measures the ability to immediately service debt with available cash.
    Higher values indicate lower financial risk.
    
    Formula:
    $$ \text{Ratio} = \frac{\text{Total Cash}}{\text{Total Debt}} $$
    """
    def __init__(self):
        super().__init__('health_cash_to_debt', num_key='total_cash', den_key='total_debt',
                        description='Cash to Debt Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure required columns exist
        if 'total_cash' not in df.columns or 'total_debt' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['total_cash'] / df['total_debt']
        
        # Winsorization: Cap at 2.0 to dampen the impact of cash-rich outliers
        # (e.g., biotech/tech firms) on the cross-sectional distribution.
        result = result.clip(upper=2.0)
        return result


@FactorRegistry.register()
class NetDebtToEBITDA(RatioFactor):
    """
    Net Debt to EBITDA.
    
    Standard industry metric for leverage relative to earnings power.
    Note: This factor is usually "Lower is Better".
    
    Formula:
    $$ \text{Ratio} = \frac{\text{Total Debt} - \text{Total Cash}}{\text{EBITDA}} $$
    
    Interpretation:
    - $< 2.0$: Conservative
    - $> 5.0$: High Risk
    """
    def __init__(self):
        super().__init__('health_net_debt_ebitda', num_key='net_debt', den_key='ebitda',
                        description='Net Debt to EBITDA')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure required columns exist
        if 'total_debt' not in df.columns or 'total_cash' not in df.columns or 'ebitda' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        net_debt = df['total_debt'] - df['total_cash']
        result = net_debt / df['ebitda']
        
        # Outlier Management:
        # Lower bound -1.0: Handles "Negative Net Debt" (Cash > Debt).
        # Upper bound 10.0: Caps distressed firms to prevent skew.
        result = result.clip(lower=-1.0, upper=10.0)
        return result


@FactorRegistry.register()
class DebtToRevenue(RatioFactor):
    """
    Debt to Revenue Ratio.
    
    Measures the debt load relative to top-line sales. Useful for valuing
    unprofitable growth companies where EBITDA is negative.
    
    Formula:
    $$ \text{Ratio} = \frac{\text{Total Debt}}{\text{Total Revenue}} $$
    """
    def __init__(self):
        super().__init__('health_debt_to_revenue', num_key='total_debt', den_key='total_revenue',
                        description='Debt to Revenue Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure required columns exist
        if 'total_debt' not in df.columns or 'total_revenue' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['total_debt'] / df['total_revenue']
        
        # Winsorization: Cap at 5.0x revenue (High distress zone).
        return result.clip(upper=5.0)


@FactorRegistry.register()
class EBITDAToDebt(RatioFactor):
    """
    EBITDA to Debt Ratio.
    
    Inverse leverage metric measuring the years of EBITDA required to
    repay gross debt. Higher is better (safer).
    
    Formula:
    $$ \text{Ratio} = \frac{\text{EBITDA}}{\text{Total Debt}} $$
    """
    def __init__(self):
        super().__init__('health_ebitda_to_debt', num_key='ebitda', den_key='total_debt',
                        description='EBITDA to Debt Ratio')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure required columns exist
        if 'ebitda' not in df.columns or 'total_debt' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        result = df['ebitda'] / df['total_debt']
        
        # Winsorization: Cap at 10.0 (Extremely strong coverage).
        return result.clip(upper=10.0)
