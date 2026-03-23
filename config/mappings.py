"""
Canonical Data Schema Mapping Registry
======================================
Centralized dictionary defining the canonical nomenclature for the Quant Alpha platform.

Purpose
-------
This module serves as the authoritative mapping registry for the ingestion (ETL) layer.
It orchestrates the translation of disparate, vendor-specific column names (e.g., from
YFinance, SEC Edgar, or proprietary CSVs) into the platform's standardized internal schema.
This normalization is critical for ensuring deterministic feature engineering and strict
cross-sectional alignment downstream.

Role in Quantitative Workflow
-----------------------------
Consumed exclusively by the `ColumnValidator` and respective data loaders during the
acquisition phase to harmonize fundamental, alternative, and earnings datasets into
monolithic, aligned parquet structures.
"""

COLUMN_MAPPINGS = {
    # 1. EARNINGS EVENTS
    # Resolves vendor-specific nomenclature for earnings announcements into canonical schemas.
    'eps_estimate':  ['eps_estimate', 'consensus_eps', 'estimatedEPS'],
    'eps_actual':    ['eps_actual', 'reported_eps', 'reportedEPS'],
    'surprise_pct':  ['surprise_pct', 'earningsSurprise', 'surprise'],

    # 2. ALTERNATIVE DATA
    # Standardizes macroeconomic indicators, commodity prices, and systemic risk metrics.
    'oil_close':     ['oil_close', 'crude_oil', 'oil'],
    'oil_vol':       ['oil_volume'],
    'vix_close':     ['vix_close', 'VIX', 'volatility_index'],
    'usd_close':     ['usd_close', 'DX-Y.NYB', 'usd_index'],
    'us_10y_close':  ['us_10y_close', 'us10y_close', 'US_10Y', 'treasury_yield', 'us_10y'],
    'sp500_close':   ['sp500_close', 'sp500', 'gspc', '^GSPC'],

    # 3. FUNDAMENTAL DATA
    # Normalizes balance sheet, income statement, and cash flow attributes.
    
    # Valuation Metrics
    'pe_ratio':      ['pe_ratio', 'trailingPE', 'priceEarningsRatio'],
    'forward_pe':    ['forward_pe', 'forwardPE'],
    'eps':           ['eps', 'trailingEps', 'dilutedEPS'],
    'fwd_eps':       ['forwardEps', 'forward_eps', 'forwardEPS'],
    'peg_ratio':     ['peg_ratio', 'pegRatio', 'priceToEarningsGrowth'],
    'ps_ratio':      ['ps_ratio', 'priceToSalesTrailing12Months'],
    'pb_ratio':      ['pb_ratio', 'priceToBook', 'price_to_book'],
    'book_value':    ['book_value', 'bookValue'],
    'ev_ebitda':     ['ev_ebitda', 'enterpriseToEbitda', 'val_ev_ebitda'],

    # Cash Flow Dynamics
    'fcf':           ['fcf', 'free_cashflow', 'freeCashflow', 'Free Cash Flow'],
    'ocf':           ['ocf', 'op_cashflow', 'operating_cashflow', 'totalCashFromOperatingActivities', 'Operating Cash Flow'],
    'total_cash':    ['total_cash', 'totalCash', 'cashAndCashEquivalents', 'Cash And Cash Equivalents', 'End Cash Position'],
    'total_debt':    ['total_debt', 'totalDebt', 'shortLongTermDebtTotal', 'Total Debt', 'Long Term Debt'],

    # Returns & Margin Profiles
    'roe':           ['roe', 'returnOnEquity'],
    'roa':           ['roa', 'returnOnAssets'],
    'gross_margin':  ['gross_margin', 'grossMargins'],
    'op_margin':     ['op_margin', 'operatingMargins'],
    'ebitda_margin': ['ebitda_margin', 'ebitdaMargins'],
    'profit_margin': ['profit_margin', 'profitMargins', 'netMargins'],

    # Financial Health & Solvency
    'debt_equity':   ['debt_to_equity', 'debtToEquity'],
    'current_ratio': ['current_ratio', 'currentRatio'],
    'quick_ratio':   ['quick_ratio', 'quickRatio'],
    
    # Growth Trajectories
    'earnings_growth': ['earnings_growth', 'earningsGrowth', 'quarterlyEarningsGrowthYOY', 'growth_earnings_growth'],
    'rev_growth':      ['rev_growth', 'revenueGrowth', 'revenue_growth', 'growth_rev_growth'],

    # Shareholder Yield (Dividends & Buybacks)
    'dividend':      ['dividend', 'dividend_yield', 'dividendYield', 'trailingAnnualDividendYield'],
    'buyback':       ['buyback', 'buybacks', 'RepurchaseOfCapitalStock', 'Repurchase Of Capital Stock'],
    'dividends_paid':['dividends_paid', 'CashDividendsPaid', 'cash_dividends_paid', 'Cash Dividends Paid'],

    # 4. MISCELLANEOUS COMMON METRICS
    'market_cap':    ['market_cap', 'marketCap', 'mkt_cap'],
    'price':         ['price', 'currentPrice'],
    'beta':          ['beta', 'beta_5y'],
    'total_revenue': ['total_revenue', 'totalRevenue', 'revenue', 'Total Revenue'],
    'ebitda':        ['ebitda', 'earningsBeforeInterestTaxesDepreciationAmortization']
}