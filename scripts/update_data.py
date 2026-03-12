"""
Incremental Data Ingestion Engine
=================================
Orchestrates the delta-update of the Data Lake, ensuring all financial datasets
(Prices, Fundamentals, Earnings, Macro) remain synchronized with the latest
market information.

Purpose
-------
This module serves as the **Maintenance Layer** of the quantitative pipeline.
It implements an "Update-in-Place" strategy to minimize bandwidth and processing
overhead. Instead of full-reloads, it identifies stale or incomplete records
and patches them incrementally, preserving existing history while appending
new observations.

Usage:
------
Executed via CLI as a scheduled cron job or ad-hoc update.

.. code-block:: bash

    # 1. Full Synchronization (All Data Types)
    python scripts/update_data.py

    # 2. Specific Domain Update (e.g., Prices only)
    python scripts/update_data.py --mode prices --workers 12

    # 3. Deep Refresh of Fundamentals (Force check > 30 days)
    python scripts/update_data.py --mode fundamentals --fund-days 30

Importance
----------
- **Data Continuity**: Ensures strictly monotonic time-series data without gaps,
  critical for accurate look-back window calculations (e.g., Volatility$_{60d}$).
- **Bandwidth Optimization**: Reduces API load by $O(1)$ (fetching only delta)
  rather than $O(T)$ (fetching full history).
- **Error Resilience**: Implements granular error handling to prevent a single
  failed ticker from aborting the entire batch process.

Tools & Frameworks
------------------
- **Pandas**: Time-series alignment, deduplication, and Parquet/CSV I/O.
- **YFinance**: Upstream data provider interface.
- **ThreadPoolExecutor**: Parallel concurrency for I/O-bound network requests.
- **Pathlib**: Object-oriented filesystem manipulation.
"""

import sys
import time
import random
import logging
import argparse
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
from config.settings import config
from quant_alpha.utils import setup_logging
import download_data as dd

# Logging Configuration: Set to ERROR to suppress transient retry noise during bulk operations.
setup_logging(default_level=logging.ERROR)
log = logging.getLogger("Quant_Alpha")

PRICE_DIR    = config.PRICES_DIR
FUND_DIR     = config.FUNDAMENTALS_DIR
EARNINGS_DIR = config.EARNINGS_DIR
ALT_DIR      = config.ALTERNATIVE_DIR

MACRO_TICKERS     = dd.MACRO_TICKERS
DEFAULT_WORKERS   = 10
DEFAULT_FUND_DAYS = 90


# =========================================================
# 1.  PRICES  (incremental append)
# =========================================================
def _update_price_ticker(file_path: Path, today: date) -> str:
    """
    Worker Task: Performs an incremental update (delta-patching) for a single ticker's price history.

    Logic:
    1.  Load existing CSV.
    2.  Determine missing head (history) or tail (recent).
    3.  Fetch only required ranges.
    4.  Merge, deduplicate, and persist.
    """
    ticker = file_path.stem
    try:
        df = pd.read_csv(file_path)
        if df.empty or "date" not in df.columns:
            return "error"

        # Timezone Normalization: Ensure UTC-naive for consistent comparisons
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Data Hygiene: Prune rows where all price columns are NaN (artifacts from failed prior fetches)
        price_cols = [c for c in df.columns if c != "date"]
        df = df.dropna(subset=price_cols, how="all")

        if df.empty:
            return "error"

        earliest_date = df["date"].min().date()
        last_date = df["date"].max().date()
        required_start = pd.to_datetime(config.BACKTEST_START_DATE).date()

        # Gap Detection Logic:
        # 1. Backfill: Missing history at the start (allow 7-day buffer for IPOs/listings).
        need_history = earliest_date > (required_start + timedelta(days=7))
        # 2. Forward-fill: Missing recent data up to T-0.
        need_update = last_date < today

        if not need_history and not need_update:
            return "uptodate"

        dfs_to_merge = [df]
        data_added = False

        # --- 1. Fetch Missing History ---
        if need_history:
            hist_data = yf.download(
                ticker,
                start=str(required_start),
                end=str(earliest_date),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if hist_data is not None and not hist_data.empty:
                if isinstance(hist_data.columns, pd.MultiIndex):
                    hist_data.columns = hist_data.columns.droplevel(1)
                hist_data = hist_data.reset_index()
                hist_data.columns = [str(c).lower() for c in hist_data.columns]
                hist_data = hist_data.loc[:, ~hist_data.columns.duplicated()]
                
                if "date" in hist_data.columns:
                    hist_data["date"] = pd.to_datetime(hist_data["date"]).dt.tz_localize(None)
                    common = [c for c in df.columns if c in hist_data.columns]
                    if "date" in common:
                        dfs_to_merge.append(hist_data[common])
                        data_added = True

        # --- 2. Fetch Recent Data ---
        if need_update:
            new_data = yf.download(
                ticker,
                start=str(last_date),
                end=None,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if new_data is not None and not new_data.empty:
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.droplevel(1)
                new_data = new_data.reset_index()
                new_data.columns = [str(c).lower() for c in new_data.columns]
                new_data = new_data.loc[:, ~new_data.columns.duplicated()]

                if "date" in new_data.columns:
                    new_data["date"] = pd.to_datetime(new_data["date"]).dt.tz_localize(None)
                    common = [c for c in df.columns if c in new_data.columns]
                    if "date" in common:
                        dfs_to_merge.append(new_data[common])
                        data_added = True

        if not data_added:
            return "uptodate"

        # Merge Strategy: Concatenate -> Deduplicate on Date (keep new) -> Sort
        full_df = (
            pd.concat(dfs_to_merge, ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        full_df.to_csv(file_path, index=False)
        return "updated"

    except Exception as e:
        # Rate Limit Handling: Exponential backoff/jitter simulation if 429 encountered
        if "Rate limited" in str(e):
            time.sleep(random.uniform(2.0, 5.0))
        log.debug(f"{ticker}: price update failed — {e}")
        return "error"


def update_prices(workers: int = DEFAULT_WORKERS) -> None:
    """
    Orchestrator: Executes threaded price updates across the universe.
    Uses ThreadPoolExecutor for I/O bound efficiency.
    """
    dd._section("📈 PRICES — Incremental Update")
    if not PRICE_DIR.exists():
        print("❌ Price directory missing. Run download_data.py first.")
        return

    files = list(PRICE_DIR.glob("*.csv"))
    if not files:
        print("⚠️  No CSV files found.")
        return

    today = datetime.now().date()
    print(f"📂 {len(files)} files  |  {workers} workers  |  today={today}")
    counts = {"updated": 0, "uptodate": 0, "error": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_update_price_ticker, f, today): f for f in files}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(files), desc="Prices", unit="ticker"):
            counts[future.result()] += 1

    print(f"\n✅ Prices done.  Updated: {counts['updated']}  "
          f"Up-to-date: {counts['uptodate']}  Errors: {counts['error']}")


# =========================================================
# 2.  FUNDAMENTALS  (re-fetch tickers whose info.csv is stale)
# =========================================================
def _info_age_days(ticker: str) -> int:
    """Calculates file age in days based on filesystem modification time ($mtime$)."""
    info_path = FUND_DIR / ticker / "info.csv"
    if not info_path.exists():
        return 9999
    mtime = datetime.fromtimestamp(info_path.stat().st_mtime).date()
    return (datetime.now().date() - mtime).days


def _fund_data_incomplete(ticker: str) -> bool:
    """
    Audit: Checks if financial statements (Balance Sheet, Income, Cash Flow) exist
    and cover a sufficient historical window.
    """
    for fname in ("financials.csv", "balance_sheet.csv", "cashflow.csv"):
        path = FUND_DIR / ticker / fname
        if not path.exists():
            return True
        try:
            df = pd.read_csv(path, index_col=0)
            if df.empty:
                return True
            # Date Parsing: Columns represent fiscal period ends (e.g., "2019-09-28")
            col_dates = pd.to_datetime(df.columns, errors="coerce").dropna()
            if col_dates.empty:
                return True
            earliest = col_dates.min().date()
            
            # Vendor Constraint: Yahoo Finance typically provides ~4 years of history.
            # Heuristic: Consider data 'complete' if it extends back at least 3 years.
            three_years_ago = datetime.now().date() - timedelta(days=365*3)
            if earliest > three_years_ago:
                return True
        except Exception:
            return True
    return False


def update_fundamentals(workers: int = 4, stale_days: int = DEFAULT_FUND_DAYS) -> None:
    """
    Orchestrator: Refresh fundamental data based on staleness criteria.
    
    Concurrency Note:
        Worker count is intentionally throttled (default=4) to mitigate
        HTTP 429/401 errors from the upstream provider.
    """
    dd._section("📊 FUNDAMENTALS — Staleness-Based Update")
    if not PRICE_DIR.exists():
        print("❌ Run download_data.py first."); return

    all_tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]

    stale = [
        t for t in all_tickers
        if _info_age_days(t) >= stale_days or _fund_data_incomplete(t)
    ]
    fresh = len(all_tickers) - len(stale)

    age_stale  = [t for t in all_tickers if _info_age_days(t) >= stale_days]
    incomplete = [t for t in all_tickers if _fund_data_incomplete(t)]

    print(f"📂 {len(all_tickers)} tickers total")
    print(f"   ✅ Fresh & complete:      {fresh}")
    print(f"   🕐 Age-stale (≥{stale_days}d):   {len(age_stale)}")
    print(f"   📉 Incomplete statements: {len(incomplete)}")
    print(f"   🔄 Total to re-fetch:     {len(stale)}")
    print(f"   ⚠️  Low worker count ({workers}) prevents Yahoo 401 crumb errors")

    if not stale:
        print("⏭️  All fundamentals are fresh and complete. Nothing to do.")
        return

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(dd._fetch_fundamental, t, True): t for t in stale}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(stale), desc="Fundamentals", unit="ticker"):
            result = future.result()
            if result.startswith("❌"):
                errors.append(result)

    print(f"✅ Fundamentals done. → {FUND_DIR}")
    if errors:
        print(f"   ⚠️  {len(errors)} failed — re-run to retry.")


# =========================================================
# 3.  EARNINGS  (re-fetch tickers with no future dates on record)
# =========================================================
def _fetch_earnings_safe(ticker: str) -> str:
    """
    Worker Task: Atomically fetch and persist earnings calendar data.
    Note: Bypasses standard retry logic as `earnings_dates` KeyErrors are often deterministic (data missing).
    """
    save_path = EARNINGS_DIR / f"{ticker}.csv"
    try:
        # Direct property access (no retry wrapper)
        earnings = yf.Ticker(ticker).earnings_dates

        if earnings is None or earnings.empty:
            return f"⚠️  {ticker} (no data)"

        # Normalization: Promote index to column, standardize naming conventions
        earnings = earnings.reset_index()
        earnings = earnings.rename(columns={
            "Earnings Date": "date",
            "EPS Estimate":  "eps_estimate",
            "Reported EPS":  "eps_actual",
            "Surprise(%)":   "surprise_pct",
        })

        # Timezone: Remove offsets for compatibility
        earnings["date"] = pd.to_datetime(earnings["date"]).dt.tz_localize(None)
        earnings.to_csv(save_path, index=False)
        return f"✅ {ticker}"

    except KeyError:
        # Vendor Quirk: yfinance raises KeyError for tickers with no earnings history
        return f"⚠️  {ticker} (no earnings data)"
    except Exception as e:
        log.debug(f"Earnings failed for {ticker}: {e}")
        return f"❌ {ticker}"


def _earnings_needs_update(ticker: str) -> bool:
    """
    Audit: Determines if the local earnings cache is actionable.
    
    Criteria:
    1. File existence / integrity (size > 50 bytes).
    2. Staleness: Contains future dates (upcoming earnings).
    3. History: Contains at least 1 year of past data.
    """
    path = EARNINGS_DIR / f"{ticker}.csv"
    if not path.exists() or path.stat().st_size < 50:
        return True
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns:
            return True
        dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
        now   = pd.Timestamp.now().tz_localize(None)

        # Check for future dates (if none, we need an update)
        if dates[dates >= now].empty:
            return True

        # Check for historical completeness
        if (now - dates.min()).days < 365:
            return True

        return False
    except Exception:
        return True


def update_earnings(workers: int = 8) -> None:
    """Orchestrator: Updates earnings records for tickers with missing or stale data."""
    dd._section("📅 EARNINGS — Smart Update")
    if not PRICE_DIR.exists():
        print("❌ Run download_data.py first."); return

    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]
    needs_update = [t for t in all_tickers if _earnings_needs_update(t)]
    fresh = len(all_tickers) - len(needs_update)

    print(f"📂 {len(all_tickers)} tickers total")
    print(f"   ✅ Has future dates: {fresh}")
    print(f"   🔄 Needs update:     {len(needs_update)}")

    if not needs_update:
        print("⏭️  All earnings are current. Nothing to do.")
        return

    counts = {"updated": 0, "no_data": 0, "error": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_earnings_safe, t): t for t in needs_update}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(needs_update), desc="Earnings", unit="ticker"):
            result = future.result()
            if result.startswith("✅"):
                counts["updated"] += 1
            elif result.startswith("⚠️"):
                counts["no_data"] += 1
            else:
                counts["error"] += 1

    print(f"✅ Earnings done. → {EARNINGS_DIR}")
    print(f"   Updated: {counts['updated']}  No data: {counts['no_data']}  Errors: {counts['error']}")


# =========================================================
# 4.  MACRO  (incremental append, same logic as prices)
# =========================================================
def _update_macro_series(name: str, ticker: str, today: date = None) -> str:
    """
    Worker Task: Updates global macro-economic indicators.
    Handles distinct data schema and source idiosyncrasies compared to equity tickers.
    
    Args:
        name: Canonical internal name (e.g., 'sp500', 'vix')
        ticker: Yahoo Finance ticker (e.g., 'SPY', '^VIX')
        today: Reference date for staleness check
    
    Returns:
        Status: 'updated', 'uptodate', or 'error'
    """
    if today is None:
        today = date.today()
    
    try:
        csv_path = ALT_DIR / f"{name}.csv"
        
        # Staleness Check
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(csv_path)
                
                if len(df) > 0:
                    # Parse the date column
                    if 'date' in df.columns:
                        last_date = pd.to_datetime(df['date'].iloc[-1]).date()
                    else:
                        # Try first column
                        first_col = df.columns[0]
                        last_date = pd.to_datetime(df[first_col].iloc[-1]).date()
                    
                    # Short-circuit if data is already current
                    if last_date >= today:
                        return "uptodate"
            except Exception as e:
                log.warning(f"Could not parse {csv_path}: {e}")
        
        # Fetch Strategy:
        # 1. Try 'download_data' module (with retry logic).
        # 2. Fallback to direct yfinance call.
        new_data = None
        
        if dd is not None:
            try:
                hist = dd._yf_ticker(ticker)
                new_data = dd._retry(lambda: hist.history(start=str(config.BACKTEST_START_DATE), end=None, auto_adjust=True), retries=3, delay=4.0)
            except Exception as e:
                log.debug(f"dd module failed: {e}")
        
        if new_data is None:
            if yf is None:
                log.error("Neither dd nor yfinance available")
                return "error"
            
            try:
                hist = yf.Ticker(ticker)
                new_data = hist.history(period="max", auto_adjust=True)
            except Exception as e:
                log.error(f"Failed to download {name}: {e}")
                return "error"
        
        if new_data is None or len(new_data) == 0:
            log.warning(f"No data received for {name}")
            return "error"
        
        # Normalization
        new_data = new_data.reset_index()
        new_data.columns = [str(col).lower().replace(' ', '_') for col in new_data.columns]
        
        # Schema alignment: Prefix columns to avoid collisions in downstream joins
        rename_map = {}
        if 'close' in new_data.columns:
            rename_map['close'] = f'{name.lower()}_close'
        if 'volume' in new_data.columns:
            rename_map['volume'] = f'{name.lower()}_volume'
        
        if rename_map:
            new_data.rename(columns=rename_map, inplace=True)
            
        # Timezone: Force naive
        if 'date' in new_data.columns:
             new_data['date'] = pd.to_datetime(new_data['date']).dt.tz_localize(None)
        
        # Persistence: Merge -> Deduplicate -> Save
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                existing = pd.read_csv(csv_path)
                if 'date' in existing.columns:
                    existing['date'] = pd.to_datetime(existing['date'])
                
                merged = pd.concat([existing, new_data], ignore_index=True)
                merged = merged.drop_duplicates(subset=['date'], keep='last')
                new_data = merged.sort_values('date').reset_index(drop=True)
            except Exception as e:
                log.warning(f"Could not merge {name} data: {e}")
        
        # Save to CSV
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_data.to_csv(csv_path, index=False)
        
        return "updated"
        
    except Exception as e:
        log.debug(f"Error updating {name}: {e}")
        return "error"


def update_macro() -> None:
    """Orchestrator: Sequentially updates all defined macro-economic indicators."""
    dd._section("🌍 MACRO — Incremental Update")
    ALT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()

    counts = {"updated": 0, "uptodate": 0, "error": 0}
    for name, ticker in MACRO_TICKERS.items():
        print(f"   ⬇️  {name}…", end=" ", flush=True)
        result = _update_macro_series(name, ticker, today)
        counts[result] += 1
        print({"updated": "✅", "uptodate": "⏭️ ", "error": "❌"}[result])

    print(f"\n✅ Macro done.  Updated: {counts['updated']}  "
          f"Up-to-date: {counts['uptodate']}  Errors: {counts['error']}")


# =========================================================
# MAIN
# =========================================================
MODES = {
    "prices":       lambda args: update_prices(args.workers),
    "fundamentals": lambda args: update_fundamentals(workers=4, stale_days=args.fund_days),
    "earnings":     lambda args: update_earnings(workers=args.workers),
    "macro":        lambda args: update_macro(),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental updater — all data types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all"] + list(MODES.keys()),
        help="Which data type to update (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Thread pool size for prices/earnings (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--fund-days", type=int, default=DEFAULT_FUND_DAYS,
        dest="fund_days",
        help=f"Re-fetch fundamentals older than N days (default: {DEFAULT_FUND_DAYS})",
    )
    args = parser.parse_args()

    print("🚀 INCREMENTAL DATA UPDATE")
    print(f"   Mode     : {args.mode}")
    print(f"   Workers  : {args.workers}")
    print(f"   Fund days: {args.fund_days}")

    targets = list(MODES.keys()) if args.mode == "all" else [args.mode]
    for mode in targets:
        try:
            MODES[mode](args)
        except Exception as e:
            print(f"\n❌ {mode} update crashed: {e}")
            print("   Skipping to next step…")

    print("\n✅ Update complete.")


if __name__ == "__main__":
    main()