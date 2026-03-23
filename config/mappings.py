"""
Centralized Column Mappings for Data Normalization
Used by ColumnValidator to map diverse data sources to standard internal names.
Matches 'fundamentals.parquet', 'earnings.parquet', and 'alternative.parquet'.
"""

COLUMN_MAPPINGS = {
    # ==================== 1. EARNINGS EVENTS (From EarningsLoader) ====================
    # Raw columns from your CSV: 'eps_estimate', 'eps_actual', 'surprise_pct'
    'eps_estimate':  ['eps_estimate', 'consensus_eps', 'estimatedEPS'],
    'eps_actual':    ['eps_actual', 'reported_eps', 'reportedEPS'],
    'surprise_pct':  ['surprise_pct', 'earningsSurprise', 'surprise'],

    # ==================== 2. ALTERNATIVE DATA (From AlternativeLoader) ====================
    # Raw columns produced by your loader: 'oil_close', 'vix_close', 'us_10y_close' etc.
    'oil_close':     ['oil_close', 'crude_oil', 'oil'],
    'oil_vol':       ['oil_volume'],
    'vix_close':     ['vix_close', 'VIX', 'volatility_index'],
    'usd_close':     ['usd_close', 'DX-Y.NYB', 'usd_index'],
    'us_10y_close':  ['us_10y_close', 'us10y_close', 'US_10Y', 'treasury_yield', 'us_10y'],
    'sp500_close':   ['sp500_close', 'sp500', 'gspc', '^GSPC'],

    # ==================== 3. FUNDAMENTAL DATA (From FundamentalsLoader) ====================
    # Valuation
    'pe_ratio':      ['pe_ratio', 'trailingPE', 'priceEarningsRatio'],
    'forward_pe':    ['forward_pe', 'forwardPE'],
    'eps':           ['eps', 'trailingEps', 'dilutedEPS'],
    'fwd_eps':       ['forwardEps', 'forward_eps', 'forwardEPS'],
    'peg_ratio':     ['peg_ratio', 'pegRatio', 'priceToEarningsGrowth'],
    'ps_ratio':      ['ps_ratio', 'priceToSalesTrailing12Months'],
    'pb_ratio':      ['pb_ratio', 'priceToBook', 'price_to_book'],
    'book_value':    ['book_value', 'bookValue'],
    'ev_ebitda':     ['ev_ebitda', 'enterpriseToEbitda', 'val_ev_ebitda'],

    # Cash Flow
    'fcf':           ['fcf', 'free_cashflow', 'freeCashflow', 'Free Cash Flow'],
    'ocf':           ['ocf', 'op_cashflow', 'operating_cashflow', 'totalCashFromOperatingActivities', 'Operating Cash Flow'],
    'total_cash':    ['total_cash', 'totalCash', 'cashAndCashEquivalents', 'Cash And Cash Equivalents', 'End Cash Position'],
    'total_debt':    ['total_debt', 'totalDebt', 'shortLongTermDebtTotal', 'Total Debt', 'Long Term Debt'],

    # Returns & Margins
    'roe':           ['roe', 'returnOnEquity'],
    'roa':           ['roa', 'returnOnAssets'],
    'gross_margin':  ['gross_margin', 'grossMargins'],
    'op_margin':     ['op_margin', 'operatingMargins'],
    'ebitda_margin': ['ebitda_margin', 'ebitdaMargins'],
    'profit_margin': ['profit_margin', 'profitMargins', 'netMargins'],

    # Health
    'debt_equity':   ['debt_to_equity', 'debtToEquity'],
    'current_ratio': ['current_ratio', 'currentRatio'],
    'quick_ratio':   ['quick_ratio', 'quickRatio'],
    
    # Growth
    'earnings_growth': ['earnings_growth', 'earningsGrowth', 'quarterlyEarningsGrowthYOY', 'growth_earnings_growth'],
    'rev_growth':      ['rev_growth', 'revenueGrowth', 'revenue_growth', 'growth_rev_growth'],

    # Dividends
    'dividend':      ['dividend', 'dividend_yield', 'dividendYield', 'trailingAnnualDividendYield'],
    'buyback':       ['buyback', 'buybacks', 'RepurchaseOfCapitalStock', 'Repurchase Of Capital Stock'],
    'dividends_paid':['dividends_paid', 'CashDividendsPaid', 'cash_dividends_paid', 'Cash Dividends Paid'],

    # ==================== 4. MISC / COMMON ====================
    'market_cap':    ['market_cap', 'marketCap', 'mkt_cap'],
    'price':         ['price', 'currentPrice'],
    'beta':          ['beta', 'beta_5y'],
    'total_revenue': ['total_revenue', 'totalRevenue', 'revenue', 'Total Revenue'],
    'ebitda':        ['ebitda', 'earningsBeforeInterestTaxesDepreciationAmortization']
}