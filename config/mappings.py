"""
Centralized Column Mappings for Data Normalization
Used by ColumnValidator to map diverse data sources to standard internal names.
"""

COLUMN_MAPPINGS = {
    # Earnings
    'pe_ratio':      ['pe_ratio', 'trailingPE', 'priceEarningsRatio'],
    'forward_pe':    ['forward_pe', 'forwardPE'],
    'eps':           ['eps', 'trailingEps'],
    'fwd_eps':       ['forwardEps', 'forward_eps'],
    
    # Book Value
    'price_to_book': ['pb_ratio', 'priceToBook', 'price_to_book'],
    'book_value':    ['bookValue', 'book_value'],
    
    # Cash Flow
    'fcf':           ['fcf', 'free_cashflow', 'freeCashflow'],
    'ocf':           ['op_cashflow', 'operating_cashflow', 'ocf'],
    
    # Dividends & Payouts
    'dividend':      ['dividend_yield', 'dividendYield'],
    'buyback':       ['buybacks', 'RepurchaseOfCapitalStock', 'repurchase_of_capital_stock'],
    'dividends_paid':['CashDividendsPaid', 'cash_dividends_paid', 'dividends_paid'],
    
    # Misc
    'market_cap':    ['market_cap', 'marketCap', 'mkt_cap'],
    'price':         ['currentPrice', 'close', 'adj_close']
}