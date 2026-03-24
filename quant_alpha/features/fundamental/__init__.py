"""
Fundamental Feature Engineering Subsystem
=========================================

Provides structural extraction algorithms mapping trailing and forward-looking 
accounting data into cross-sectional predictive alpha metrics.

Purpose
-------
Aggregates valuation, quality, growth, and financial health boundaries, bridging 
unstructured SEC filings and corporate releases into standardized, quantitatively 
stationary machine-learning signals.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Dimensional mapping, scalar conversions, and dynamic 
  data type enforcement matching historical execution states.
"""

# Value Factors
from .value import (
    EarningsYield,
    ForwardEarningsYield,
    BookYield,
    FCFYield,
    OperatingCashFlowYield,
    DividendYield,
    ShareholderYield,
    EnterpriseValueFCFYield,
    PriceToSalesRatio,
    EVtoEBITDAValuation,
    EVtoSalesValuation,
)

# Quality Factors
from .quality import (
    ROE,
    ROA,
    GrossMargin,
    OperatingMargin,
    EBITDAMargin,
    ProfitMargin,
    LowLeverage,
    LowBeta,
    FCFConversion,
    AccrualsRatio,
)

# Growth Factors
from .growth import (
    EarningsGrowth,
    RevenueGrowth,
    ForwardEPSGrowth,
    PEGRatio,
    SustainableGrowthRate,
    ReinvestmentRate,
)

# Financial Health Factors
from .financial_health import (
    DebtToEquity,
    CurrentRatio,
    QuickRatio,
    CashToDebtRatio,
    NetDebtToEBITDA,
    DebtToRevenue,
    EBITDAToDebt,
)

# Utilities
from .utils import (
    FundamentalColumnValidator,
    SingleColumnFactor,
    RatioFactor,
)

__all__ = [
    # Value
    'EarningsYield',
    'ForwardEarningsYield',
    'BookYield',
    'FCFYield',
    'OperatingCashFlowYield',
    'DividendYield',
    'ShareholderYield',
    'EnterpriseValueFCFYield',
    'PriceToSalesRatio',
    'EVtoEBITDAValuation',
    'EVtoSalesValuation',
    # Quality
    'ROE',
    'ROA',
    'GrossMargin',
    'OperatingMargin',
    'EBITDAMargin',
    'ProfitMargin',
    'LowLeverage',
    'LowBeta',
    'FCFConversion',
    'AccrualsRatio',
    # Growth
    'EarningsGrowth',
    'RevenueGrowth',
    'ForwardEPSGrowth',
    'PEGRatio',
    'SustainableGrowthRate',
    'ReinvestmentRate',
    # Financial Health
    'DebtToEquity',
    'CurrentRatio',
    'QuickRatio',
    'CashToDebtRatio',
    'NetDebtToEBITDA',
    'DebtToRevenue',
    'EBITDAToDebt',
    # Utilities
    'FundamentalColumnValidator',
    'SingleColumnFactor',
    'RatioFactor',
]
