"""
Fundamental Factors Package
Value, Quality, Growth, and Financial Health metrics
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
    EVtoEBIDAValuation,
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
    'EVtoEBIDAValuation',
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
