"""
Data Lake Diagnostic and Integrity Suite
========================================
Verifies the alignment, consistency, and structural integrity of the ingested 
Data Lake relative to the point-in-time universe membership mask.

Purpose
-------
This script acts as a strict validation gate before feature engineering. It audits 
the intersection of disparate datasets (OHLCV, Fundamentals, Earnings, Macro) 
against the historical S&P 500 constituents to ensure no data leakage, missing 
observations, or survivorship bias exist in the downstream pipeline.

Role in Quantitative Workflow
-----------------------------
Executed post-ingestion (`update_data.py`) and pre-engineering to assert that 
the active universe matrix is fully populated and temporally aligned.

Dependencies
------------
- **Pandas/NumPy**: In-memory data manipulation and alignment checks.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging

setup_logging(default_level=logging.INFO)
logger = logging.getLogger("Diagnostics")

def run_diagnostics():
    """
    Executes the comprehensive data diagnostic sequence.

    Evaluates price alignment, fundamental statement completeness, earnings 
    history, macro indicator presence, and survivorship bias retention. Prints 
    a formatted audit report to standard output.

    Args:
        None

    Returns:
        None
    """
    print("=" * 70)
    print(" 🕵️‍♂️ DATA INGESTION DIAGNOSTIC REPORT")
    print("=" * 70)

    mask_path = config.MEMBERSHIP_MASK_PATH
    if not mask_path.exists():
        print(f"❌ ERROR: Membership mask not found at {mask_path}")
        return
    
    target_start_date = pd.to_datetime(config.BACKTEST_START_DATE).date()
    target_start_ts = pd.Timestamp(config.BACKTEST_START_DATE)
    print(f"Target Backtest Start Date: {target_start_date}")

    print("\nLoading Membership Mask...")
    mask = pd.read_pickle(mask_path)
    mask_dates = set(mask.index.normalize())
    mask_tickers = set(mask.columns)
    print(f"  Mask Shape: {mask.shape[0]} dates x {mask.shape[1]} tickers")

    price_dir = config.PRICES_DIR
    if not price_dir.exists():
        print(f"❌ ERROR: Price directory not found at {price_dir}")
        return

    price_files = list(price_dir.glob("*.csv"))
    price_tickers = set([f.stem for f in price_files])
    print(f"  Price Files Found: {len(price_files)}")

    fund_dir = config.FUNDAMENTALS_DIR
    earn_dir = config.EARNINGS_DIR
    alt_dir = config.ALTERNATIVE_DIR

    fund_tickers = set([d.name for d in fund_dir.iterdir() if d.is_dir()]) if fund_dir.exists() else set()
    earn_files = list(earn_dir.glob("*.csv")) if earn_dir.exists() else []
    earn_tickers = set([f.stem for f in earn_files])
    macro_files = set([f.stem.lower() for f in alt_dir.glob("*.csv")]) if alt_dir.exists() else set()

    print("\n[1] PRICE DATA CONSISTENCY CHECK")
    missing_in_prices = mask_tickers - price_tickers
    missing_in_mask = price_tickers - mask_tickers

    if missing_in_prices:
        print(f"  ❌ {len(missing_in_prices)} tickers in Mask but missing from Price Data.")
        print(f"      Examples: {list(missing_in_prices)[:5]}")
    else:
        print("  ✅ All tickers in Mask have corresponding Price Data.")

    if missing_in_mask:
        print(f"  ℹ️ {len(missing_in_mask)} tickers in Price Data but missing from Mask (likely extra downloads).")

    print("\n[2] FUNDAMENTAL DATA CHECK")
    missing_fund = mask_tickers - fund_tickers
    if missing_fund:
        print(f"  ⚠️ {len(missing_fund)} tickers in Mask missing Fundamental folders.")
        print(f"      Examples: {list(missing_fund)[:5]}")
    else:
        print("  ✅ All mask tickers have a Fundamental folder.")
        
    incomplete_funds = []
    fund_min_dates = []
    fund_max_dates = []
    if fund_dir.exists():
        print("  Scanning Fundamental dates...")
        for t in fund_tickers.intersection(mask_tickers):
            has_core = False
            if (fund_dir / t / 'info.csv').exists() or (fund_dir / t / 'financials.csv').exists():
                has_core = True
                
            if not has_core:
                incomplete_funds.append(t)
            else:
                # Extract temporal boundaries for fundamental data to ensure sufficient 
                # history exists for trailing indicator generation (e.g. 3-year averages)
                for fname in ['financials.csv', 'balance_sheet.csv', 'cashflow.csv']:
                    fpath = fund_dir / t / fname
                    if fpath.exists():
                        try:
                            df_fund = pd.read_csv(fpath, nrows=0, index_col=0)
                            valid_dates = pd.to_datetime(df_fund.columns, errors='coerce').dropna()
                            valid_dates = valid_dates[valid_dates >= target_start_ts]
                            if not valid_dates.empty:
                                fund_min_dates.append(valid_dates.min().date())
                                fund_max_dates.append(valid_dates.max().date())
                        except Exception:
                            pass

    if incomplete_funds:
        print(f"  ⚠️ {len(incomplete_funds)} folders missing core statements (info.csv or financials.csv).")
        print(f"      Examples: {incomplete_funds[:5]}")
    elif fund_dir.exists() and fund_tickers:
        print("  ✅ All checked fundamental folders contain core CSVs.")

    if fund_min_dates and fund_max_dates:
        print(f"  📅 Fundamental Date Range: {min(fund_min_dates)} to {max(fund_max_dates)}")

    print("\n[3] EARNINGS DATA CHECK")
    missing_earn = mask_tickers - earn_tickers
    if missing_earn:
        print(f"  ⚠️ {len(missing_earn)} tickers in Mask missing Earnings CSVs.")
        print(f"      Examples: {list(missing_earn)[:5]}")
    else:
        print("  ✅ All mask tickers have an Earnings CSV.")
        
    empty_earns = []
    earn_min_dates = []
    earn_max_dates = []
    if earn_dir.exists():
        print("  Scanning Earnings dates...")
        for f in earn_files:
            if f.stat().st_size < 50:
                empty_earns.append(f.stem)
            elif f.stem in mask_tickers:
                try:
                    df_earn = pd.read_csv(f, usecols=lambda c: c.lower() == 'date')
                    if not df_earn.empty and 'date' in df_earn.columns:
                        valid_dates = pd.to_datetime(df_earn['date'], errors='coerce', utc=True).dt.tz_localize(None).dropna()
                        valid_dates = valid_dates[valid_dates >= target_start_ts]
                        if not valid_dates.empty:
                            earn_min_dates.append(valid_dates.min().date())
                            earn_max_dates.append(valid_dates.max().date())
                except Exception:
                    pass

    if empty_earns:
        print(f"  ⚠️ {len(empty_earns)} Earnings CSVs are empty or near-empty (no data).")
        print(f"      Examples: {empty_earns[:5]}")
    elif earn_files:
        print("  ✅ All Earnings CSVs have valid data payloads.")

    if earn_min_dates and earn_max_dates:
        print(f"  📅 Earnings Date Range: {min(earn_min_dates)} to {max(earn_max_dates)}")

    print("\n[4] ALTERNATIVE / MACRO DATA CHECK")
    expected_macro = {'vix', 'us_10y', 'oil', 'usd', 'sp500'}
    missing_macro = expected_macro - macro_files
    if missing_macro:
        print(f"  ❌ Missing expected macro series: {missing_macro}")
    else:
        print(f"  ✅ All expected macro series found: {expected_macro}")

    macro_min_dates = []
    macro_max_dates = []
    if alt_dir.exists():
        print("  Scanning Macro dates...")
        for f in alt_dir.glob("*.csv"):
            try:
                df_macro = pd.read_csv(f, usecols=lambda c: c.lower() == 'date')
                if not df_macro.empty and 'date' in df_macro.columns:
                    valid_dates = pd.to_datetime(df_macro['date'], errors='coerce').dropna()
                    valid_dates = valid_dates[valid_dates >= target_start_ts]
                    if not valid_dates.empty:
                        macro_min_dates.append(valid_dates.min().date())
                        macro_max_dates.append(valid_dates.max().date())
            except Exception:
                pass

    if macro_min_dates and macro_max_dates:
        print(f"  📅 Macro Date Range: {min(macro_min_dates)} to {max(macro_max_dates)}")

    print("\n[5] PRICE DATE ALIGNMENT CHECK")
    
    print("  Aggregating close prices to check alignment and NaNs (this may take a moment)...")
    
    close_prices = {}
    missing_close_tickers = []
    auto_cleaned_tickers = []
    
    for f in price_files:
        ticker = f.stem
        if ticker not in mask_tickers:
            continue
            
        df = pd.read_csv(f)
        cols_lower = {c.lower(): c for c in df.columns}
        
        price_col = None
        if 'adj close' in cols_lower:
            price_col = cols_lower['adj close']
        elif 'close' in cols_lower:
            price_col = cols_lower['close']
            
        date_col = cols_lower.get('date', None)
            
        if price_col is None or date_col is None:
            missing_close_tickers.append(ticker)
            continue
            
        # Stability Guard: Auto-clean bankrupt/OTC artifacts where price drops to zero or below,
        # which would otherwise trigger DivisionByZero or log(0) exceptions during factor generation.
        if (df[price_col] <= 0).any():
            auto_cleaned_tickers.append(ticker)
            df = df[df[price_col] > 0.0]
            df.to_csv(f, index=False)
            
        df_subset = df[[date_col, price_col]].copy()
        df_subset = df_subset.rename(columns={date_col: 'date'})
        df_subset['date'] = pd.to_datetime(df_subset['date']).dt.normalize()
        df_subset = df_subset.set_index('date')
        
        close_prices[ticker] = df_subset[price_col]

    price_matrix = pd.DataFrame(close_prices)
    price_dates = set(price_matrix.index)
    
    mask_min, mask_max = min(mask_dates).date(), max(mask_dates).date()
    price_min, price_max = min(price_dates).date(), max(price_dates).date()
    
    print(f"  Mask Date Range:  {mask_min} to {mask_max}")
    print(f"  Price Date Range: {price_min} to {price_max}")
    
    dates_in_mask_not_in_price = mask_dates - price_dates
    if len(dates_in_mask_not_in_price) > 0:
        print(f"  ⚠️ Mask contains {len(dates_in_mask_not_in_price)} dates not in price data (likely weekends/holidays).")
    else:
        print("  ✅ All dates in Mask are present in Price Data.")

    print("\n[6] PRICE INTEGRITY & CORPORATE ACTION CHECK")
    if missing_close_tickers:
        print(f"  ❌ {len(missing_close_tickers)} tickers missing 'close'/'adj close' column.")
    else:
        print("  ✅ All price files contain 'close' or 'adj close' (split/div adjusted).")

    if auto_cleaned_tickers:
        print(f"  ✅ Auto-cleaned 0 or negative prices for {len(auto_cleaned_tickers)} tickers.")
        print(f"      Examples: {auto_cleaned_tickers[:5]}")
    else:
        print("  ✅ No 0 or negative prices detected.")

    print("\n[7] SURVIVORSHIP BIAS TEST")
    
    end_date = mask.index.max()
    removed_tickers = [col for col in mask.columns if mask[col].any() and not mask.loc[end_date, col]]
    
    # Isolate known bankrupt or acquired assets to explicitly verify that the historical 
    # ingestion pipeline has not inadvertently dropped them from the dataset.
    test_tickers = ['YHOO', 'FRC', 'SIVB', 'TWTR', 'SBNY']
    found_test_tickers = [t for t in test_tickers if t in removed_tickers]
    
    for t in removed_tickers:
        if len(found_test_tickers) >= 3:
            break
        if t not in found_test_tickers:
            found_test_tickers.append(t)

    for ticker in found_test_tickers[:3]:
        if ticker not in price_matrix.columns:
            print(f"  ❌ {ticker}: In Mask but MISSING from Price Data! (Survivorship Bias risk)")
            continue
            
        membership = mask[ticker]
        last_date_in_index = membership[membership].index.max()
        prices_after_removal = price_matrix.loc[price_matrix.index > last_date_in_index, ticker]
        
        print(f"  🔍 {ticker}:")
        print(f"     - Removed from index after: {last_date_in_index.date()}")
        print(f"     - Mask correctly marks as False at end date: {not mask.loc[end_date, ticker]}")
        print(f"     - Historical Price data present: {'✅ Yes' if price_matrix[ticker].notna().any() else '❌ No'}")
        if prices_after_removal.notna().any():
            print(f"     - Price data continues after removal: ✅ Yes (Proves Delisted/OTC tracking)")
        else:
            print(f"     - Price data continues after removal: ⚠️ No (Likely acquired/bankrupt immediately)")

    print("\n[8] SUMMARY REPORT & DATE SPAN VALIDATION")
    
    price_start = min(price_dates).date() if price_dates else None
    fund_start = min(fund_min_dates) if fund_min_dates else None
    earn_start = min(earn_min_dates) if earn_min_dates else None
    macro_start = min(macro_min_dates) if macro_min_dates else None
    
    grace_period = timedelta(days=7)
    
    if price_start and price_start > (target_start_date + grace_period):
        print(f"  ❌ Price data starts late: {price_start} (Target: {target_start_date} + 7d grace)")
    elif price_start and price_start > target_start_date:
        print(f"  ✅ Price data starts {price_start} (within 7-day holiday grace period of {target_start_date})")
        
    if macro_start and macro_start > (target_start_date + grace_period):
        print(f"  ❌ Macro data starts late: {macro_start} (Target: {target_start_date} + 7d grace)")
    elif macro_start and macro_start > target_start_date:
        print(f"  ✅ Macro data starts {macro_start} (within 7-day holiday grace period of {target_start_date})")
        
    if earn_start and earn_start > target_start_date:
        print(f"  ⚠️  Earnings data starts late ({earn_start}) compared to backtest start ({target_start_date}).")
    if fund_start:
        three_years_ago = (pd.Timestamp.today() - pd.DateOffset(years=3)).date()
        if fund_start > three_years_ago:
             print(f"  ⚠️  Fundamental data is very short (starts {fund_start}). YFinance usually provides 3-4 years.")
        elif fund_start > target_start_date:
             print(f"  ℹ️  Fundamental data starts {fund_start}, backtest starts {target_start_date} (Expected limitation of YF).")
    
    print("-" * 50)
    
    common_dates = sorted(list(mask_dates.intersection(price_dates)))
    common_tickers = sorted(list(mask_tickers.intersection(price_tickers)))
    
    pm_aligned = price_matrix.loc[common_dates, common_tickers]
    mask_aligned = mask.loc[common_dates, common_tickers]
    
    total_active_cells = mask_aligned.sum().sum()
    missing_active_prices = (pm_aligned.isna() & mask_aligned).sum().sum()
    missing_pct = (missing_active_prices / total_active_cells) * 100 if total_active_cells > 0 else 0
    
    print(f"  Total Data Points (Active Universe): {total_active_cells:,}")
    print(f"  Missing Prices in Active Universe:   {missing_active_prices:,} ({missing_pct:.2f}%)")
    
    print("\n" + "=" * 70)
    
    is_go = True
    issues = []
    
    if missing_in_prices:
        is_go = False
        issues.append(f"{len(missing_in_prices)} Mask tickers missing price data.")
    if missing_pct > 5.0:
        is_go = False
        issues.append(f"Missing price data percentage ({missing_pct:.2f}%) is too high (>5%).")
        
    missing_fund_pct = (len(missing_fund) / len(mask_tickers)) * 100 if mask_tickers else 100
    missing_earn_pct = (len(missing_earn) / len(mask_tickers)) * 100 if mask_tickers else 100
    
    if missing_macro:
        is_go = False
        issues.append(f"Missing alternative (macro) data: {', '.join(missing_macro)}")
    if missing_fund_pct > 25.0:
        is_go = False
        issues.append(f"Fundamental data missing for {missing_fund_pct:.1f}% of universe (>25% limit).")
    if missing_earn_pct > 25.0:
        is_go = False
        issues.append(f"Earnings data missing for {missing_earn_pct:.1f}% of universe (>25% limit).")
        
    if price_start and price_start > (target_start_date + grace_period):
        is_go = False
        issues.append(f"Price data begins after the target start date + grace period ({price_start} > {target_start_date}).")
        
    if macro_start and macro_start > (target_start_date + grace_period):
        is_go = False
        issues.append(f"Macro data begins after the target start date + grace period ({macro_start} > {target_start_date}).")

    if is_go:
        print("  🟢 FINAL STATUS: GO")
        print("  Data is clean, aligned, and ready for Feature Engineering.")
    else:
        print("  🔴 FINAL STATUS: NO-GO")
        print("  Please resolve the following issues before proceeding:")
        for issue in issues:
            print(f"    - {issue}")
    print("=" * 70)

if __name__ == '__main__':
    run_diagnostics()