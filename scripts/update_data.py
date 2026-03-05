"""
update_data.py
==============
Incremental Daily Updater — all data types
------------------------------------------
Updates only what's stale. Never re-downloads from scratch.

Usage:
    python update_data.py                   # Update everything
    python update_data.py --mode prices     # Prices only
    python update_data.py --mode earnings   # Earnings only
    python update_data.py --mode macro      # Macro only
    python update_data.py --workers 8       # Override thread count
    python update_data.py --fund-days 30    # Re-fetch fundamentals older than 30 days
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from config.settings import config
from config.logging_config import setup_logging
import download_data as dd

# ERROR level only — retry noise is not actionable
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
    ticker = file_path.stem
    try:
        df = pd.read_csv(file_path)
        if df.empty or "date" not in df.columns:
            return "error"

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # Drop rows where ALL price columns are NaN (empty rows from bad prior download)
        price_cols = [c for c in df.columns if c != "date"]
        df = df.dropna(subset=price_cols, how="all")

        if df.empty:
            return "error"

        earliest_date = df["date"].min().date()
        last_date = df["date"].max().date()
        required_start = pd.to_datetime(config.BACKTEST_START_DATE).date()

        # Check for gaps:
        # 1. Missing history (allow 7 days buffer)
        need_history = earliest_date > (required_start + timedelta(days=7))
        # 2. Missing recent data
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

        full_df = (
            pd.concat(dfs_to_merge, ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        full_df.to_csv(file_path, index=False)
        return "updated"

    except Exception as e:
        if "Rate limited" in str(e):
            time.sleep(random.uniform(2.0, 5.0))
        log.debug(f"{ticker}: price update failed — {e}")
        return "error"


def update_prices(workers: int = DEFAULT_WORKERS) -> None:
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
    """Days since info.csv was last modified, or 9999 if missing."""
    info_path = FUND_DIR / ticker / "info.csv"
    if not info_path.exists():
        return 9999
    mtime = datetime.fromtimestamp(info_path.stat().st_mtime).date()
    return (datetime.now().date() - mtime).days


def _fund_data_incomplete(ticker: str) -> bool:
    """
    True if financial statements are missing OR don't reach back to
    BACKTEST_START_DATE. financials.csv columns are fiscal period dates.
    """
    for fname in ("financials.csv", "balance_sheet.csv", "cashflow.csv"):
        path = FUND_DIR / ticker / fname
        if not path.exists():
            return True
        try:
            df = pd.read_csv(path, index_col=0)
            if df.empty:
                return True
            # Columns are date strings like "2019-09-28 00:00:00"
            col_dates = pd.to_datetime(df.columns, errors="coerce").dropna()
            if col_dates.empty:
                return True
            earliest = col_dates.min().date()
            
            # Fix: Yahoo only provides ~4 years of history. 
            # If we have at least 3 years of data from today, consider it complete.
            three_years_ago = datetime.now().date() - timedelta(days=365*3)
            if earliest > three_years_ago:
                return True
        except Exception:
            return True
    return False


def update_fundamentals(workers: int = 4, stale_days: int = DEFAULT_FUND_DAYS) -> None:
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
    Fetch and save earnings for a single ticker.
    Does NOT use _retry — earnings_dates raises KeyError internally on some
    tickers (yfinance bug), retrying just wastes time.
    """
    save_path = EARNINGS_DIR / f"{ticker}.csv"
    try:
        # earnings_dates is a property; access it directly, no retry wrapper
        earnings = yf.Ticker(ticker).earnings_dates

        if earnings is None or earnings.empty:
            return f"⚠️  {ticker} (no data)"

        # reset_index() promotes DatetimeIndex ("Earnings Date") → column
        earnings = earnings.reset_index()
        earnings = earnings.rename(columns={
            "Earnings Date": "date",
            "EPS Estimate":  "eps_estimate",
            "Reported EPS":  "eps_actual",
            "Surprise(%)":   "surprise_pct",
        })

        # Normalize to timezone-naive datetime
        earnings["date"] = pd.to_datetime(earnings["date"]).dt.tz_localize(None)
        earnings.to_csv(save_path, index=False)
        return f"✅ {ticker}"

    except KeyError:
        # yfinance raises KeyError(['Earnings Date']) for some tickers — not retryable
        return f"⚠️  {ticker} (no earnings data)"
    except Exception as e:
        log.debug(f"Earnings failed for {ticker}: {e}")
        return f"❌ {ticker}"


def _earnings_needs_update(ticker: str) -> bool:
    """
    True if:
    - file missing / too small
    - no future earnings dates (stale)
    - oldest date is less than 1 year of history (incomplete)
    Note: yfinance earnings_dates only provides ~2 years max — that's a
    Yahoo Finance API limit, not something we can fix by re-fetching.
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

        # Stale: no future dates recorded
        if dates[dates >= now].empty:
            return True

        # Incomplete: less than 1 year of history (probably a bad prior fetch)
        if (now - dates.min()).days < 365:
            return True

        return False
    except Exception:
        return True


def update_earnings(workers: int = 8) -> None:
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
def _update_macro_series(name: str, ticker: str, today: date) -> str:
    out = ALT_DIR / f"{name}.csv"
    close_col = f"{name.lower()}_close"

    def _full_download():
        df = dd._retry(lambda: dd._yf_ticker(ticker).history(
            start=str(config.BACKTEST_START_DATE), end=None, auto_adjust=True
        ), retries=3, delay=4.0)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        keep = ["date", "close"] + (["volume"] if "volume" in df.columns else [])
        return df[keep].rename(columns={
            "close": close_col,
            "volume": f"{name.lower()}_volume",
        })

    # File missing → full download
    if not out.exists():
        try:
            df = _full_download()
            if df is None:
                return "error"
            df.to_csv(out, index=False)
            return "updated"
        except Exception as e:
            log.debug(f"{name}: full download failed — {e}")
            return "error"

    try:
        df = pd.read_csv(out)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        last_date = df["date"].max().date()

        if last_date >= today:
            return "uptodate"

        time.sleep(random.uniform(1.0, 2.5))

        new_data = dd._retry(lambda: dd._yf_ticker(ticker).history(
            start=str(last_date), end=None, auto_adjust=True
        ), retries=3, delay=4.0)

        if new_data is None or new_data.empty:
            return "uptodate"

        new_data = new_data.reset_index()
        new_data.columns = [str(c).lower() for c in new_data.columns]

        if "date" not in new_data.columns or "close" not in new_data.columns:
            return "error"

        new_data["date"] = pd.to_datetime(new_data["date"]).dt.tz_localize(None)
        new_data = new_data[["date", "close"]].rename(columns={"close": close_col})

        full_df = (
            pd.concat([df, new_data], ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        full_df.to_csv(out, index=False)
        return "updated"

    except Exception as e:
        log.debug(f"{name}: macro update failed — {e}")
        return "error"


def update_macro() -> None:
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