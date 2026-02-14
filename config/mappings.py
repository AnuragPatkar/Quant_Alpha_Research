"""
Centralized Column Mappings for Data Normalization
Used by ColumnValidator to map diverse data sources to standard internal names.
"""

COLUMN_MAPPINGS = {
    # Earnings
    'pe_ratio':      ['pe_ratio', 'trailingPE', 'priceEarningsRatio', 'trailingPeRatio'],
    'forward_pe':    ['forward_pe', 'forwardPE', 'forwardPeRatio'],
    'eps':           ['eps', 'trailingEps', 'dilutedEPS', 'epsTrailingTwelveMonths'],
    'fwd_eps':       ['forwardEps', 'forward_eps', 'epsForward', 'forwardEPS'],
    
    # Book Value
    'price_to_book': ['pb_ratio', 'priceToBook', 'price_to_book'],
    'book_value':    ['bookValue', 'book_value'],
    
    # Cash Flow
    'fcf':           ['fcf', 'free_cashflow', 'freeCashflow'],
    'ocf':           ['op_cashflow', 'operating_cashflow', 'ocf', 'totalCashFromOperatingActivities', 'operatingCashflow'],
    'cash_equivalents': ['cashAndCashEquivalents', 'cash_and_cash_equivalents', 'cash', 'cash_equivalents'],
    
    # Dividends & Payouts
    'dividend':      ['dividend_yield', 'dividendYield', 'trailingAnnualDividendYield', 'dividendRate'],
    'buyback':       ['buybacks', 'RepurchaseOfCapitalStock', 'repurchase_of_capital_stock'],
    'dividends_paid':['CashDividendsPaid', 'cash_dividends_paid', 'dividends_paid'],
    
    # Misc
    'market_cap':    ['market_cap', 'marketCap', 'mkt_cap'],
    'price':         ['currentPrice', 'close', 'adj_close'],

    # ==================== QUALITY & SAFETY FACTORS ====================
    # Returns (Profitability)
    'roe':           ['roe', 'returnOnEquity'],
    'roa':           ['roa', 'returnOnAssets'],
    
    # Margins
    'gross_margin':  ['gross_margin', 'grossMargins'],
    'op_margin':     ['op_margin', 'operatingMargins'],
    'ebitda_margin': ['ebitda_margin', 'ebitdaMargins'],
    'profit_margin': ['profit_margin', 'profitMargins', 'netMargins'],
    
    # Safety / Leverage
    'debt_equity':   ['debt_to_equity', 'debtToEquity'],
    'total_debt':    ['totalDebt', 'shortLongTermDebtTotal', 'total_debt'],
    'total_cash':    ['totalCash', 'cash_and_cash_equivalents', 'total_cash', 'cashAndCashEquivalents'],
    'interest_expense': ['interestExpense', 'interest_expense'],
    'ebitda':        ['ebitda', 'earningsBeforeInterestTaxesDepreciationAmortization'],
    'current_ratio': ['current_ratio', 'currentRatio'],
    'quick_ratio':   ['quick_ratio', 'quickRatio'],
    
    # Stability / Risk
    'beta':          ['beta', 'beta_5y'],
    
    # Growth (Bonus)
    'earnings_growth': ['earnings_growth', 'earningsGrowth', 'quarterlyEarningsGrowthYOY', 'eps_growth', 'earningsQuarterlyGrowth'],
    'rev_growth':      ['rev_growth', 'revenueGrowth', 'revenue_growth', 'quarterlyRevenueGrowthYOY'],
    'peg_ratio':       ['peg_ratio', 'pegRatio', 'trailingPegRatio', 'peg', 'priceToEarningsGrowth'],
    
    # Additional Valuation Metrics
    'ps_ratio':        ['ps_ratio', 'priceToSalesTrailing12Months', 'price_to_sales'],
    'ev_ebitda':       ['ev_ebitda', 'enterpriseToEbitda', 'enterprise_to_ebitda'],
    'total_revenue':   ['total_revenue', 'totalRevenue', 'revenue'],
}