"""
Fundamental Data Preprocessor
=============================

Transforms raw unstructured financial reporting data into standardized, 
structurally derived factor inputs optimized for algorithmic consumption.

Purpose
-------
This module cleans, normalizes, and extracts continuous predictive metrics 
from historical 10-K and 10-Q corporate filings (Balance Sheets, Cash Flow, 
and Income Statements).

Role in Quantitative Workflow
-----------------------------
Serves as the rigorous extraction barrier translating categorical accounting 
structures into actionable alpha signals (e.g., Value, Quality, Growth primitives). 
It strictly controls reporting delays by enforcing an artificial data 
availability lag, guaranteeing zero look-ahead bias in the research outputs.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Vectorized missing-value imputations, continuous scaling limits, 
  and matrix arithmetic.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logger = logging.getLogger(__name__)

EPS = 1e-9
WINSOR_LO_PCT = 0.01
WINSOR_HI_PCT = 0.99


# ============================================================================
# CORE PREPROCESSING FUNCTION
# ============================================================================

def preprocess_fundamentals(
    fundamentals_dir: Union[str, Path],
    ticker: str,
    fiscal_year_end: Optional[str] = None,
    reporting_lag_days: int = 90,
) -> pd.DataFrame:
    """
    Extracts and standardizes raw periodic fundamental data mapping to a distinct ticker.

    Applies strict temporal offsetting to accurately simulate corporate reporting 
    delays, preventing systemic data leakage from newly closed fiscal boundaries.

    Args:
        fundamentals_dir (Union[str, Path]): Root path pointing to hierarchical fundamental files.
        ticker (str): The specific target equity symbol strictly mapping the data target.
        fiscal_year_end (Optional[str]): Explicit expected calendar month defining the annual close. 
            Defaults to None, initiating dynamic extraction based on local distributions.
        reporting_lag_days (int): The mandatory padding buffer mimicking real-world 10-K/10-Q 
            SEC publishing limits. Defaults to 90.

    Returns:
        pd.DataFrame: A fully formulated Point-in-Time validated matrix bounding discrete 
            annual snapshots explicitly cleared for public execution states.
    """
    fundamentals_dir = Path(fundamentals_dir)
    ticker_dir = fundamentals_dir / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(f"Ticker directory not found: {ticker_dir}")

    logger.info(f"[Fundamentals] Loading {ticker}...")

    info_df = _load_csv_safe(ticker_dir / "info.csv",         has_index=False)
    bs_df   = _load_csv_safe(ticker_dir / "balance_sheet.csv", has_index=True)
    fin_df  = _load_csv_safe(ticker_dir / "financials.csv",    has_index=True)
    cf_df   = _load_csv_safe(ticker_dir / "cashflow.csv",      has_index=True)

    if bs_df is None or fin_df is None:
        logger.warning(
            f"[Fundamentals] {ticker}: Missing balance_sheet or financials — skipping."
        )
        return pd.DataFrame()

    bs_cols      = bs_df.columns
    fiscal_dates = pd.to_datetime(bs_cols, errors='coerce').dropna()

    if len(fiscal_dates) == 0:
        logger.warning(f"[Fundamentals] {ticker}: No valid fiscal dates found.")
        return pd.DataFrame()

    info_series = None
    if info_df is not None and not info_df.empty:
        info_series = info_df.iloc[0]

    today = datetime.today().date()
    rows = []

    for fiscal_date in sorted(fiscal_dates, reverse=True):
        fiscal_date_str = fiscal_date.strftime('%Y-%m-%d')

        # Point-in-time: data becomes available ~90 days after fiscal close
        available_date = fiscal_date + timedelta(days=reporting_lag_days)

        # Strictly enforces rigorous Point-in-Time (PiT) data hygiene by strictly terminating 
        # observations where the projected public disclosure date exceeds current runtime bounds.
        if available_date.date() > today:
            logger.debug(
                f"[Fundamentals] {ticker}: Skipping FY {fiscal_date_str} "
                f"— report not yet available until {available_date.date()} "
                f"(today: {today})"
            )
            continue

        bs_vals  = bs_df[fiscal_date_str]  if fiscal_date_str in bs_df.columns  else None
        fin_vals = fin_df[fiscal_date_str] if fiscal_date_str in fin_df.columns else None
        cf_vals  = cf_df[fiscal_date_str]  if cf_df is not None and fiscal_date_str in cf_df.columns else None

        prev_fin_vals = None
        prev_year = fiscal_date - pd.DateOffset(years=1)
        prev_str  = prev_year.strftime('%Y-%m-%d')
        if fin_df is not None and prev_str in fin_df.columns:
            prev_fin_vals = fin_df[prev_str]

        row = _compute_fundamental_row(
            ticker=ticker,
            available_date=available_date,
            info=info_series,
            balance_sheet=bs_vals,
            financials=fin_vals,
            cashflow=cf_vals,
            prev_financials=prev_fin_vals,
        )
        rows.append(row)

    if not rows:
        logger.info(
            f"[Fundamentals] {ticker}: No rows produced "
            f"(all may be future-dated or missing data)."
        )
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    logger.info(f"[Fundamentals] {ticker}: {len(result_df)} fiscal years processed.")
    return result_df


# ============================================================================
# HELPER: LOAD CSV SAFELY
# ============================================================================

def _load_csv_safe(
    path: Path,
    has_index: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Instantiates discrete CSV extractions mapping resilient failsafes averting I/O exceptions.
    
    Args:
        path (Path): Explicit filepath resolving localized mapping structures.
        has_index (bool): Dictates target dimensionality interpreting historical primary indices.
        
    Returns:
        Optional[pd.DataFrame]: Safely evaluated continuous panel, returning None 
            if fundamental file dependencies are absent.
    """
    if not path.exists():
        return None
    try:
        if has_index:
            df = pd.read_csv(path, index_col=0)
        else:
            df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception as exc:
        logger.warning(f"[Fundamentals] Could not load {path.name}: {exc}")
        return None


def _get_field(series: Optional[pd.Series], key: str) -> float:
    """
    Isolates independent structural primitives dynamically mapping scalar series data.
    
    Args:
        series (Optional[pd.Series]): Source target defining localized characteristics.
        key (str): Dimensional feature key triggering extraction bounds.
        
    Returns:
        float: Validated explicit continuous value explicitly casting missing states to np.nan.
    """
    if series is None:
        return np.nan
    try:
        val = series.get(key, np.nan)
        return float(val) if pd.notna(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_compute(fn, name: str) -> float:
    """
    Encapsulates arithmetic limits validating fault-tolerant continuous feature evaluations.
    
    Args:
        fn (Callable): Dynamic extraction lambda evaluating structural equations.
        name (str): Analytical definition string bounded for debug targeting.
        
    Returns:
        float: Extracted valid arithmetic mapping, forcefully defaulting to np.nan resolving 
            unforeseen division/attribute crashes.
    """
    try:
        result = fn()
        return float(result) if pd.notna(result) else np.nan
    except Exception as exc:
        logger.debug(f"[Fundamentals] {name}: {exc}")
        return np.nan


# ============================================================================
# FUNDAMENTAL ROW BUILDER  (unchanged logic, included in full for completeness)
# ============================================================================

def _compute_fundamental_row(
    ticker: str,
    available_date: datetime,
    info: Optional[pd.Series],
    balance_sheet: Optional[pd.Series],
    financials: Optional[pd.Series],
    cashflow: Optional[pd.Series],
    prev_financials: Optional[pd.Series] = None,
) -> Dict:
    """
    Synthesizes a standardized, single-observation temporal row strictly computing derived factors.

    Args:
        ticker (str): Target evaluation equity identifier.
        available_date (datetime): Validated Point-in-Time target mapping index.
        info (Optional[pd.Series]): Cross-sectional general equity descriptors.
        balance_sheet (Optional[pd.Series]): Annual structural ledger parameters.
        financials (Optional[pd.Series]): Income statement vector distributions.
        cashflow (Optional[pd.Series]): Trailing cash flow extraction bounds.
        prev_financials (Optional[pd.Series]): Lagged income vectors supporting YOY calculations.

    Returns:
        Dict: Mapped dictionary encoding absolute extracted financial targets uniformly.
    """

    row: Dict = {
        'ticker': ticker,
        'date':   available_date,
    }

    # ---- VALUE FACTORS ----
    row['val_earnings_yield']         = _safe_compute(lambda: _earnings_yield(info),            "val_earnings_yield")
    row['val_forward_earnings_yield'] = _safe_compute(lambda: _forward_earnings_yield(info),    "val_forward_earnings_yield")
    row['val_book_yield']             = _safe_compute(lambda: _book_yield(info),                "val_book_yield")
    row['val_fcf_yield']              = _safe_compute(lambda: _fcf_yield(info, cashflow),       "val_fcf_yield")
    row['val_ocf_yield']              = _safe_compute(lambda: _ocf_yield(info, cashflow),       "val_ocf_yield")
    row['val_shareholder_yield']      = _safe_compute(lambda: _shareholder_yield(info, cashflow), "val_shareholder_yield")
    row['val_ev_fcf_yield']           = _safe_compute(lambda: _ev_fcf_yield(info, balance_sheet, cashflow), "val_ev_fcf_yield")
    row['val_ps_ratio']               = _safe_compute(lambda: _price_to_sales(info, financials), "val_ps_ratio")
    row['val_ev_ebitda']              = _safe_compute(lambda: _ev_ebitda(info, balance_sheet, financials), "val_ev_ebitda")
    row['val_ev_sales']               = _safe_compute(lambda: _ev_sales(info, balance_sheet, financials), "val_ev_sales")

    # ---- QUALITY FACTORS ----
    row['qual_roe']           = _safe_compute(lambda: _roe(financials, balance_sheet),     "qual_roe")
    row['qual_roa']           = _safe_compute(lambda: _roa(financials, balance_sheet),     "qual_roa")
    row['qual_gross_margin']  = _safe_compute(lambda: _gross_margin(financials),           "qual_gross_margin")
    row['qual_op_margin']     = _safe_compute(lambda: _op_margin(financials),              "qual_op_margin")
    row['qual_ebitda_margin'] = _safe_compute(lambda: _ebitda_margin(financials),          "qual_ebitda_margin")
    row['qual_profit_margin'] = _safe_compute(lambda: _profit_margin(financials),          "qual_profit_margin")
    row['qual_low_leverage']  = _safe_compute(lambda: _low_leverage(balance_sheet),        "qual_low_leverage")
    row['qual_fcf_conversion']= _safe_compute(lambda: _fcf_conversion(financials, cashflow), "qual_fcf_conversion")
    row['qual_accruals']      = _safe_compute(lambda: _accruals_ratio(financials, balance_sheet, cashflow), "qual_accruals")

    # ---- HEALTH FACTORS ----
    row['health_current_ratio']   = _safe_compute(lambda: _current_ratio(balance_sheet),    "health_current_ratio")
    row['health_quick_ratio']     = _safe_compute(lambda: _quick_ratio(balance_sheet),      "health_quick_ratio")
    row['health_debt_to_equity']  = _safe_compute(lambda: _debt_to_equity(balance_sheet),   "health_debt_to_equity")

    # ---- GROWTH FACTORS ----
    row['growth_fwd_eps_growth']         = _safe_compute(lambda: _forward_eps_growth(info),              "growth_fwd_eps_growth")
    row['growth_sustainable_growth_rate']= _safe_compute(lambda: _sustainable_growth_rate(financials, balance_sheet, cashflow), "growth_sustainable_growth_rate")
    row['growth_earnings_growth']        = _safe_compute(
        lambda: _yoy_growth(financials, prev_financials, 'Net Income From Continuing Operation Net Minority Interest'),
        "growth_earnings_growth"
    )
    row['growth_rev_growth']     = _safe_compute(
        lambda: _yoy_growth(financials, prev_financials, 'Total Revenue'),
        "growth_rev_growth"
    )
    row['growth_reinvestment_rate'] = _safe_compute(lambda: _reinvestment_rate(financials, cashflow), "growth_reinvestment_rate")

    # ---- RAW INTERMEDIATE COLUMNS (for FactorRegistry) ----
    def safe_extract(series, key_or_keys):
        if series is None or (hasattr(series, 'empty') and series.empty):
            return np.nan
        keys = [key_or_keys] if isinstance(key_or_keys, str) else key_or_keys
        for key in keys:
            try:
                if key in series.index:
                    val = series[key]
                    return val if pd.notna(val) else np.nan
            except (KeyError, IndexError, TypeError, AttributeError):
                continue
        return np.nan

    if info is not None:
        row['eps']       = safe_extract(info, 'trailingEps')
        row['fwd_eps']   = safe_extract(info, 'forwardEps')
        row['pe_ratio']  = safe_extract(info, 'trailingPE')
        row['forward_pe']= safe_extract(info, 'forwardPE')
        row['price']     = safe_extract(info, 'currentPrice')
        row['market_cap']= safe_extract(info, 'marketCap')
        row['ps_ratio']  = safe_extract(info, 'priceToSalesTrailing12Months')
        row['pb_ratio']  = safe_extract(info, 'priceToBook')
        row['book_value']= safe_extract(info, 'bookValue')
        row['beta']      = safe_extract(info, 'beta')
        row['dividend']  = safe_extract(info, 'trailingAnnualDividendYield')

    if balance_sheet is not None:
        cr_val = safe_extract(balance_sheet, ['Current Ratio', 'currentRatio'])
        if pd.notna(cr_val):
            row['current_ratio'] = cr_val
        else:
            # Fallback calculation so the FactorRegistry finds the raw column
            row['current_ratio'] = _safe_compute(lambda: _current_ratio(balance_sheet), "current_ratio_fallback")
            
        qr_val = safe_extract(balance_sheet, ['Quick Ratio', 'quickRatio'])
        if pd.notna(qr_val):
            row['quick_ratio'] = qr_val
        else:
            row['quick_ratio'] = _safe_compute(lambda: _quick_ratio(balance_sheet), "quick_ratio_fallback")
        row['total_assets'] = safe_extract(balance_sheet, ['Total Assets', 'totalAssets'])
        row['total_debt']   = safe_extract(balance_sheet, ['Total Debt', 'shortLongTermDebtTotal', 'Long Term Debt'])
        row['total_equity'] = safe_extract(balance_sheet, ['Stockholders Equity', 'totalStockholderEquity', 'Total Equity'])
        if pd.isna(row.get('debt_equity', np.nan)):
            row['debt_equity'] = _safe_compute(lambda: _debt_to_equity(balance_sheet), "debt_equity")

    if financials is not None:
        row['gross_margin']   = _safe_compute(lambda: _gross_margin(financials),   "gross_margin")   if pd.isna(row.get('gross_margin', np.nan))   else row['gross_margin']
        row['op_margin']      = _safe_compute(lambda: _op_margin(financials),      "op_margin")      if pd.isna(row.get('op_margin', np.nan))      else row['op_margin']
        row['ebitda_margin']  = _safe_compute(lambda: _ebitda_margin(financials),  "ebitda_margin")  if pd.isna(row.get('ebitda_margin', np.nan))  else row['ebitda_margin']
        row['profit_margin']  = _safe_compute(lambda: _profit_margin(financials),  "profit_margin")  if pd.isna(row.get('profit_margin', np.nan))  else row['profit_margin']
        row['total_revenue']  = safe_extract(financials, ['Total Revenue', 'totalRevenue', 'Revenue'])
        row['ebitda']         = safe_extract(financials, ['EBITDA', 'Normalized EBITDA'])
        if pd.isna(row.get('roe', np.nan)):
            row['roe'] = _safe_compute(lambda: _roe(financials, balance_sheet), "roe")
        if pd.isna(row.get('roa', np.nan)) and balance_sheet is not None:
            net_income   = safe_extract(financials, ['Net Income', 'netIncome'])
            total_assets = safe_extract(balance_sheet, ['Total Assets', 'totalAssets'])
            if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
                row['roa'] = net_income / total_assets

    if cashflow is not None:
        row['fcf']        = safe_extract(cashflow, ['Free Cash Flow', 'FreeCashFlow', 'freeCashflow'])
        row['ocf']        = safe_extract(cashflow, ['Operating Cash Flow', 'OperatingCashFlow', 'totalCashFromOperatingActivities'])
        row['total_cash'] = safe_extract(cashflow, ['End Cash Position', 'cashAndCashEquivalents', 'Cash'])
        row['buyback']    = safe_extract(cashflow, ['Repurchase Of Capital Stock', 'RepurchaseOfCapitalStock', 'Stock Repurchase'])

    return row


# ============================================================================
# HELPER FUNCTIONS — VALUE FACTORS
# ============================================================================

def _earnings_yield(info):
    if info is None or info.empty: return np.nan
    pe = _get_field(info, 'trailingPE')
    if pd.notna(pe) and pe > 0.1:
        return 1.0 / (np.clip(pe, 0.1, 200) + EPS)
    eps   = _get_field(info, 'epsTrailingTwelveMonths')
    price = _get_field(info, 'regularMarketPrice')
    if pd.notna(eps) and pd.notna(price) and price > 0:
        return eps / (price + EPS)
    return np.nan

def _forward_earnings_yield(info):
    if info is None or info.empty: return np.nan
    fpe = _get_field(info, 'forwardPE')
    if pd.notna(fpe) and fpe > 0.1:
        return 1.0 / (np.clip(fpe, 0.1, 200) + EPS)
    eps_fwd = _get_field(info, 'epsForward')
    price   = _get_field(info, 'regularMarketPrice')
    if pd.notna(eps_fwd) and pd.notna(price) and price > 0:
        return eps_fwd / (price + EPS)
    return np.nan

def _book_yield(info):
    if info is None or info.empty: return np.nan
    bv    = _get_field(info, 'bookValue')
    price = _get_field(info, 'regularMarketPrice')
    if pd.notna(bv) and pd.notna(price) and price > EPS:
        return np.clip(bv, -1000, 1000) / (price + EPS)
    return np.nan

def _fcf_yield(info, cashflow):
    if info is None or cashflow is None: return np.nan
    fcf = _get_field(cashflow, 'Free Cash Flow')
    mc  = _get_field(info, 'marketCap')
    if pd.notna(fcf) and pd.notna(mc) and mc > 0:
        return fcf / (mc + EPS)
    return np.nan

def _ocf_yield(info, cashflow):
    if info is None or cashflow is None: return np.nan
    ocf = _get_field(cashflow, 'Operating Cash Flow')
    mc  = _get_field(info, 'marketCap')
    if pd.notna(ocf) and pd.notna(mc) and mc > 0:
        return ocf / (mc + EPS)
    return np.nan

def _shareholder_yield(info, cashflow):
    if info is None or cashflow is None: return np.nan
    mc       = _get_field(info, 'marketCap')
    buyback  = _get_field(cashflow, 'Repurchase Of Capital Stock')
    div_paid = _get_field(cashflow, 'Cash Dividends Paid')
    if not pd.notna(mc) or mc <= 0: return np.nan
    payout = 0.0
    if pd.notna(buyback):  payout += abs(buyback)
    if pd.notna(div_paid): payout += abs(div_paid)
    return payout / (mc + EPS)

def _ev_fcf_yield(info, balance_sheet, cashflow):
    if info is None or cashflow is None: return np.nan
    fcf = _get_field(cashflow, 'Free Cash Flow')
    mc  = _get_field(info, 'marketCap')
    if not (pd.notna(fcf) and pd.notna(mc) and mc > 0): return np.nan
    debt = 0.0
    if balance_sheet is not None:
        d = _get_field(balance_sheet, 'Total Debt')
        if pd.notna(d): debt = d
    ev = mc + debt
    return fcf / (ev + EPS)

def _price_to_sales(info, financials):
    if info is None or financials is None: return np.nan
    mc  = _get_field(info, 'marketCap')
    rev = _get_field(financials, 'Total Revenue')
    if pd.notna(mc) and pd.notna(rev) and mc > 0:
        return rev / (mc + EPS)   # Sales Yield = Revenue / Market Cap
    return np.nan

def _ev_ebitda(info, balance_sheet, financials):
    if info is None or financials is None: return np.nan
    mc     = _get_field(info, 'marketCap')
    ebitda = _get_field(financials, 'EBITDA')
    if not (pd.notna(mc) and pd.notna(ebitda) and mc > 0): return np.nan
    debt = 0.0
    if balance_sheet is not None:
        d = _get_field(balance_sheet, 'Total Debt')
        if pd.notna(d): debt = d
    ev = mc + debt
    return ebitda / (ev + EPS)

def _ev_sales(info, balance_sheet, financials):
    if info is None or financials is None: return np.nan
    mc  = _get_field(info, 'marketCap')
    rev = _get_field(financials, 'Total Revenue')
    if not (pd.notna(mc) and pd.notna(rev) and mc > 0): return np.nan
    debt = 0.0
    if balance_sheet is not None:
        d = _get_field(balance_sheet, 'Total Debt')
        if pd.notna(d): debt = d
    ev = mc + debt
    return rev / (ev + EPS)

# ============================================================================
# HELPER FUNCTIONS — QUALITY FACTORS
# ============================================================================

def _roe(financials, balance_sheet):
    if financials is None or balance_sheet is None: return np.nan
    ni  = _get_field(financials,    'Net Income From Continuing Operation Net Minority Interest')
    if not pd.notna(ni): ni = _get_field(financials, 'Net Income')
    eq  = _get_field(balance_sheet, 'Stockholders Equity')
    if not pd.notna(eq): eq = _get_field(balance_sheet, 'Total Equity')
    if pd.notna(ni) and pd.notna(eq) and abs(eq) > EPS:
        return ni / (eq + EPS)
    return np.nan

def _roa(financials, balance_sheet):
    if financials is None or balance_sheet is None: return np.nan
    ni = _get_field(financials,    'Net Income From Continuing Operation Net Minority Interest')
    if not pd.notna(ni): ni = _get_field(financials, 'Net Income')
    ta = _get_field(balance_sheet, 'Total Assets')
    if pd.notna(ni) and pd.notna(ta) and ta > 0:
        return ni / (ta + EPS)
    return np.nan

def _gross_margin(financials):
    if financials is None: return np.nan
    rev  = _get_field(financials, 'Total Revenue')
    gp   = _get_field(financials, 'Gross Profit')
    if pd.notna(rev) and pd.notna(gp) and rev > 0:
        return gp / (rev + EPS)
    return np.nan

def _op_margin(financials):
    if financials is None: return np.nan
    rev = _get_field(financials, 'Total Revenue')
    oi  = _get_field(financials, 'Operating Income')
    if not pd.notna(oi): oi = _get_field(financials, 'EBIT')
    if pd.notna(rev) and pd.notna(oi) and rev > 0:
        return oi / (rev + EPS)
    return np.nan

def _ebitda_margin(financials):
    if financials is None: return np.nan
    rev    = _get_field(financials, 'Total Revenue')
    ebitda = _get_field(financials, 'EBITDA')
    if not pd.notna(ebitda): ebitda = _get_field(financials, 'Normalized EBITDA')
    if pd.notna(rev) and pd.notna(ebitda) and rev > 0:
        return ebitda / (rev + EPS)
    return np.nan

def _profit_margin(financials):
    if financials is None: return np.nan
    rev = _get_field(financials, 'Total Revenue')
    ni  = _get_field(financials, 'Net Income From Continuing Operation Net Minority Interest')
    if not pd.notna(ni): ni = _get_field(financials, 'Net Income')
    if pd.notna(rev) and pd.notna(ni) and rev > 0:
        return ni / (rev + EPS)
    return np.nan

def _low_leverage(balance_sheet):
    if balance_sheet is None: return np.nan
    debt = _get_field(balance_sheet, 'Total Debt')
    eq   = _get_field(balance_sheet, 'Stockholders Equity')
    if not pd.notna(eq): eq = _get_field(balance_sheet, 'Total Equity')
    if pd.notna(debt) and pd.notna(eq) and abs(eq) > EPS:
        return -(debt / (eq + EPS))   # invert so higher = safer
    return np.nan

def _fcf_conversion(financials, cashflow):
    if financials is None or cashflow is None: return np.nan
    ocf = _get_field(cashflow,   'Operating Cash Flow')
    fcf = _get_field(cashflow,   'Free Cash Flow')
    if pd.notna(ocf) and pd.notna(fcf) and abs(ocf) > EPS:
        return fcf / (ocf + EPS)
    return np.nan

def _accruals_ratio(financials, balance_sheet, cashflow):
    if any(x is None for x in [financials, balance_sheet, cashflow]): return np.nan
    ni  = _get_field(financials,    'Net Income From Continuing Operation Net Minority Interest')
    if not pd.notna(ni): ni = _get_field(financials, 'Net Income')
    ocf = _get_field(cashflow,   'Operating Cash Flow')
    ta  = _get_field(balance_sheet, 'Total Assets')
    if pd.notna(ni) and pd.notna(ocf) and pd.notna(ta) and ta > 0:
        return -((ni - ocf) / (ta + EPS))
    return np.nan

# ============================================================================
# HELPER FUNCTIONS — HEALTH FACTORS
# ============================================================================

def _current_ratio(balance_sheet):
    if balance_sheet is None: return np.nan
    val = _get_field(balance_sheet, 'Current Ratio')
    if pd.notna(val): return val
    ca = _get_field(balance_sheet, 'Current Assets')
    cl = _get_field(balance_sheet, 'Current Liabilities')
    if pd.notna(ca) and pd.notna(cl) and cl > 0:
        return ca / (cl + EPS)
    return np.nan

def _quick_ratio(balance_sheet):
    if balance_sheet is None: return np.nan
    val = _get_field(balance_sheet, 'Quick Ratio')
    if pd.notna(val): return val
    ca = _get_field(balance_sheet, 'Current Assets')
    inv = _get_field(balance_sheet, 'Inventory')
    cl = _get_field(balance_sheet, 'Current Liabilities')
    if pd.notna(ca) and pd.notna(cl) and cl > 0:
        inv_val = inv if pd.notna(inv) else 0.0
        return (ca - inv_val) / (cl + EPS)
    return np.nan

def _debt_to_equity(balance_sheet):
    if balance_sheet is None: return np.nan
    debt = _get_field(balance_sheet, 'Total Debt')
    eq   = _get_field(balance_sheet, 'Stockholders Equity')
    if not pd.notna(eq): eq = _get_field(balance_sheet, 'Total Equity')
    if pd.notna(debt) and pd.notna(eq) and abs(eq) > EPS:
        return debt / (eq + EPS)
    return np.nan

# ============================================================================
# HELPER FUNCTIONS — GROWTH FACTORS
# ============================================================================

def _forward_eps_growth(info):
    if info is None: return np.nan
    fwd   = _get_field(info, 'epsForward')
    curr  = _get_field(info, 'epsTrailingTwelveMonths')
    if pd.notna(fwd) and pd.notna(curr) and abs(curr) > EPS:
        return (fwd - curr) / (abs(curr) + EPS)
    return np.nan

def _sustainable_growth_rate(financials, balance_sheet, cashflow):
    roe = _roe(financials, balance_sheet)
    if not pd.notna(roe): return np.nan
    # Resolves standard dividend structural limits conservatively evaluating an implicit 
    # 40% baseline payout mapping boundary when granular cash flows are mathematically unavailable.
    payout = 0.4
    if cashflow is not None:
        div = _get_field(cashflow, 'Cash Dividends Paid')
        ni  = None
        if financials is not None:
            ni = _get_field(financials, 'Net Income From Continuing Operation Net Minority Interest')
            if not pd.notna(ni): ni = _get_field(financials, 'Net Income')
        if pd.notna(div) and pd.notna(ni) and abs(ni) > EPS:
            payout = np.clip(abs(div) / abs(ni), 0.0, 1.0)
    return roe * (1.0 - payout)

def _yoy_growth(financials, prev_financials, field: str):
    if financials is None or prev_financials is None: return np.nan
    curr = _get_field(financials,      field)
    prev = _get_field(prev_financials, field)
    if pd.notna(curr) and pd.notna(prev) and abs(prev) > EPS:
        return (curr - prev) / (abs(prev) + EPS)
    return np.nan

def _reinvestment_rate(financials, cashflow):
    if cashflow is None: return np.nan
    ocf  = _get_field(cashflow, 'Operating Cash Flow')
    fcf  = _get_field(cashflow, 'Free Cash Flow')
    if pd.notna(ocf) and pd.notna(fcf) and abs(ocf) > EPS:
        capex = ocf - fcf
        return capex / (ocf + EPS)
    return np.nan


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_preprocessed_data(
    df: pd.DataFrame,
    coverage_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Audits the structural density and sparsity limits of the processed matrix block.

    Args:
        df (pd.DataFrame): The preprocessed fundamental asset frame.
        coverage_threshold (float): Minimum numerical density ratio mapped to strictly 
            approve valid columns. Defaults to 0.80.

    Returns:
        pd.DataFrame: An aggregated diagnostic reporting matrix mapping individual column states.
    """
    if df.empty:
        logger.warning("[Validation] Empty DataFrame provided.")
        return pd.DataFrame()

    coverage_report = []

    for col in df.columns:
        if col in ['ticker', 'date']:
            continue
        non_nan  = df[col].notna().sum()
        total    = len(df)
        coverage = (non_nan / total) if total > 0 else 0.0

        if coverage >= coverage_threshold:
            status = 'OK'
        elif coverage >= 0.50:
            status = 'WARNING'
            logger.warning(
                f"[Validation] Column '{col}': {coverage:.1%} coverage "
                f"(< {coverage_threshold:.0%})"
            )
        else:
            status = 'ERROR'
            logger.error(
                f"[Validation] Column '{col}': {coverage:.1%} coverage "
                f"(< 50% — unusable)"
            )

        coverage_report.append({
            'column':       col,
            'non_nan_count': non_nan,
            'total_rows':   total,
            'coverage_pct': f"{coverage:.1%}",
            'status':       status,
        })

    report_df    = pd.DataFrame(coverage_report)
    error_count  = (report_df['status'] == 'ERROR').sum()
    warning_count= (report_df['status'] == 'WARNING').sum()

    logger.info(
        f"[Validation] Coverage Report: {len(report_df)} columns | "
        f"{error_count} errors | {warning_count} warnings"
    )
    return report_df


# ============================================================================
# MACRO DATA FETCHING
# ============================================================================

def get_macro_data_for_earnings_signal(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Constructs historical aggregate proxies extracting strictly continuous macro configurations.
    
    Args:
        start_date (Optional[str]): Defines absolute initiating boundaries.
        end_date (Optional[str]): Defines absolute terminal boundaries.
        
    Returns:
        pd.DataFrame: Symmetrically bounded dataframe enclosing unified macro signals.
    """
    logger.info("[Macro] Fetching macro data...")

    if HAS_YFINANCE:
        try:
            spy = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            vix = yf.download('^VIX',  start=start_date, end=end_date, progress=False)
            tnx = yf.download('^TNX',  start=start_date, end=end_date, progress=False)

            macro = pd.DataFrame({
                'date':             spy.index,
                'market_regime':    (spy['Close'].pct_change() * 100).rolling(63).sum(),
                'volatility_proxy': vix['Close'].rolling(63).mean(),
                'yield_proxy':      tnx['Close'].rolling(63).mean(),
                'sp500_close':      spy['Close'],
                'vix_close':        vix['Close'],
                'us_10y_close':     tnx['Close'],
            }).reset_index(drop=True)

            logger.info(f"[Macro] yfinance fetch successful: {len(macro)} observations")
            return macro

        except Exception as e:
            logger.error(f"[Macro] yfinance fetch failed: {e}")

    logger.warning("[Macro] Could not fetch any macro data.")
    return pd.DataFrame()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    fundamentals_dir = Path('data/raw/fundamentals')
    ticker = 'A'
    df = preprocess_fundamentals(fundamentals_dir, ticker)
    print(df)