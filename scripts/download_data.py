"""
download_data.py
================
Standalone Master Data Pipeline
--------------------------------
Usage:
    python download_data.py                  # Run all steps
    python download_data.py --step prices    # Run one step
    python download_data.py --force          # Re-download existing files
    python download_data.py --step validate  # Just validate

Steps:
    1. prices        — S&P 500 OHLCV via vectorized yfinance
    2. fundamentals  — Balance Sheet, Income Stmt, Cashflow (threaded)
    3. earnings      — Historical EPS dates & estimates (threaded)
    4. macro         — VIX, Yields, Oil, USD, SP500
    5. validate      — Health check & universe diagnosis
"""

import sys
import io
import time
import random
import warnings
import concurrent.futures
import argparse
import logging
from pathlib import Path
from typing import Callable

import yfinance as yf
import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PROJECT SETUP
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import setup_logging

# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
setup_logging(default_level=logging.WARNING)
log = logging.getLogger("Quant_Alpha")

PRICE_DIR    = config.PRICES_DIR
FUND_DIR     = config.FUNDAMENTALS_DIR
EARNINGS_DIR = config.EARNINGS_DIR
ALT_DIR      = config.ALTERNATIVE_DIR
START_DATE   = config.BACKTEST_START_DATE
END_DATE     = config.BACKTEST_END_DATE

# ⚠️  Keep FUND_WORKERS low — Yahoo's crumb/session expires fast under
#     high concurrency, causing 401 "Invalid Crumb" floods.
FUND_WORKERS     = 4
EARNINGS_WORKERS = 8

MACRO_TICKERS = {
    "VIX":   "^VIX",
    "US_10Y":"^TNX",
    "OIL":   "CL=F",
    "USD":   "DX-Y.NYB",
    "SP500": "^GSPC",
}

# Browser-like headers to avoid 401s on .info calls
_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _retry(fn, retries: int = 3, delay: float = 3.0):
    """Call fn(), retrying on exception with exponential back-off."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            wait = delay * attempt
            log.warning(f"Attempt {attempt} failed ({e}). Retrying in {wait:.0f}s…")
            time.sleep(wait)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _yf_ticker(symbol: str) -> yf.Ticker:
    """Return a Ticker with browser-like headers to reduce 401s."""
    t = yf.Ticker(symbol)
    t.session = requests.Session()
    t.session.headers.update(_YF_HEADERS)
    return t


# =========================================================
# 1.  S&P 500 TICKER LIST
# =========================================================
def get_sp500_tickers() -> list[str]:
    """Scrapes Wikipedia for current S&P 500 constituents."""
    print("📋 Fetching S&P 500 tickers from Wikipedia…")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = _retry(lambda: requests.get(url, headers=_YF_HEADERS, timeout=15))
        resp.raise_for_status()
        tickers = pd.read_html(io.StringIO(resp.text))[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"✅ Found {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        print(f"❌ Critical Error fetching tickers: {e}")
        sys.exit(1)


# =========================================================
# 2.  PRICE DOWNLOAD
# =========================================================
def download_prices(force: bool = False) -> None:
    """Downloads OHLCV data for all S&P 500 tickers."""
    _section("📈 STEP 1 / 4 — PRICE DATA")
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = get_sp500_tickers()

    if not force:
        tickers = [t for t in tickers if not (PRICE_DIR / f"{t}.csv").exists()]
        if not tickers:
            print("⏭️  All price files exist. Use --force to re-download.")
            return
        print(f"⬇️  {len(tickers)} new tickers to download.")
    else:
        print(f"🚀 Downloading {len(tickers)} stocks ({START_DATE} → {END_DATE})…")

    try:
        data = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True,
        )

        print("\n💾 Saving CSVs…")
        count, skipped = 0, 0
        for ticker in tickers:
            try:
                df = (data[ticker] if len(tickers) > 1 else data).copy()
                df = df.dropna(how="all")
                if df.empty:
                    skipped += 1
                    continue
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                df.to_csv(PRICE_DIR / f"{ticker}.csv", index=False)
                count += 1
            except KeyError:
                log.warning(f"Data missing for {ticker}")
                skipped += 1

        print(f"🏆 Prices done!  Saved: {count}  |  Empty/Missing: {skipped} → {PRICE_DIR}")

    except Exception as e:
        print(f"❌ Critical Error: {e}")


# =========================================================
# 3.  FUNDAMENTAL DOWNLOAD
# =========================================================
def _fetch_fundamental(ticker: str, force: bool = False) -> str:
    save_path = FUND_DIR / ticker
    if not force and save_path.exists() and (save_path / "info.csv").exists():
        return f"⏭️  {ticker}"

    # Jitter: spread requests to avoid crumb invalidation under concurrency
    time.sleep(random.uniform(0.3, 1.2))

    try:
        def _pull():
            stock = _yf_ticker(ticker)
            save_path.mkdir(parents=True, exist_ok=True)

            # Guard: stock.info can return None or an empty/minimal dict
            info = stock.info
            if not info or not isinstance(info, dict) or len(info) < 5:
                raise ValueError(f"info dict empty or invalid for {ticker}")
            pd.DataFrame([info]).to_csv(save_path / "info.csv", index=False)

            for attr, fname in [
                ("financials",    "financials.csv"),
                ("balance_sheet", "balance_sheet.csv"),
                ("cashflow",      "cashflow.csv"),
            ]:
                df = getattr(stock, attr)
                if df is not None and not df.empty:
                    df.to_csv(save_path / fname)

        _retry(_pull, retries=3, delay=4.0)
        return f"✅ {ticker}"

    except Exception as e:
        log.warning(f"Fundamentals failed for {ticker}: {e}")
        return f"❌ {ticker}"


def download_fundamentals(force: bool = False) -> None:
    """Downloads fundamental data (threaded, rate-limited)."""
    _section("📊 STEP 2 / 4 — FUNDAMENTAL DATA")
    if not PRICE_DIR.exists():
        print("❌ Run prices first."); return

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]
    print(f"🚀 {len(tickers)} stocks  |  {FUND_WORKERS} threads")
    print(f"   (Low worker count intentional — prevents Yahoo 401 crumb errors)")

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=FUND_WORKERS) as ex:
        futures = {ex.submit(_fetch_fundamental, t, force): t for t in tickers}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tickers)):
            result = future.result()
            if result.startswith("❌"):
                errors.append(result)

    print(f"🏆 Fundamentals done! → {FUND_DIR}")
    if errors:
        print(f"   ⚠️  {len(errors)} failed. Re-run to retry (skips already saved).")


# =========================================================
# 4.  EARNINGS DOWNLOAD
# =========================================================
def _fetch_earnings(ticker: str, force: bool = False) -> str:
    save_path = EARNINGS_DIR / f"{ticker}.csv"
    if not force and save_path.exists():
        return f"⏭️  {ticker}"

    time.sleep(random.uniform(0.2, 0.8))

    try:
        def _pull():
            try:
                return _yf_ticker(ticker).earnings_dates
            except KeyError:
                return None
        earnings = _retry(_pull, retries=3, delay=3.0)

        if earnings is not None and not earnings.empty:
            (
                earnings.reset_index()
                .rename(columns={
                    "Earnings Date": "date",
                    "EPS Estimate":  "eps_estimate",
                    "Reported EPS":  "eps_actual",
                    "Surprise(%)":   "surprise_pct",
                })
                .to_csv(save_path, index=False)
            )
            return f"✅ {ticker}"
        return f"⚠️  {ticker} (no data)"

    except Exception as e:
        log.warning(f"Earnings failed for {ticker}: {e}")
        return f"❌ {ticker}"


def download_earnings(force: bool = False) -> None:
    """Downloads earnings history (threaded)."""
    _section("📅 STEP 3 / 4 — EARNINGS DATA")
    if not PRICE_DIR.exists():
        print("❌ Run prices first."); return

    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]
    print(f"🚀 {len(tickers)} stocks  |  {EARNINGS_WORKERS} threads…")

    with concurrent.futures.ThreadPoolExecutor(max_workers=EARNINGS_WORKERS) as ex:
        futures = {ex.submit(_fetch_earnings, t, force): t for t in tickers}
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(tickers)):
            pass

    print(f"🏆 Earnings done! → {EARNINGS_DIR}")


# =========================================================
# 5.  MACRO / ALTERNATIVE DOWNLOAD
# =========================================================
def download_macro(force: bool = False) -> None:
    """Downloads macro-economic indicators."""
    _section("🌍 STEP 4 / 4 — MACRO / ALTERNATIVE DATA")
    ALT_DIR.mkdir(parents=True, exist_ok=True)

    for name, ticker in MACRO_TICKERS.items():
        out = ALT_DIR / f"{name}.csv"
        if not force and out.exists():
            print(f"⏭️  {name} (skipped)")
            continue

        print(f"⬇️  {name} ({ticker})…")
        time.sleep(random.uniform(2.0, 4.0))   # macro calls are sequential; be polite

        try:
            df = _retry(lambda t=ticker: _yf_ticker(t).history(
                start=START_DATE, end=END_DATE, auto_adjust=True
            ), retries=3, delay=4.0)

            if df.empty:
                print(f"⚠️  No data for {name}"); continue

            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if "close" in df.columns:
                keep = ["date", "close"] + (["volume"] if "volume" in df.columns else [])
                df = df[keep].rename(columns={
                    "close":  f"{name.lower()}_close",
                    "volume": f"{name.lower()}_volume",
                })
                df.to_csv(out, index=False)
                print(f"✅ {name}.csv — {len(df)} rows")

        except Exception as e:
            print(f"❌ Failed {name}: {e}")

    print(f"🏆 Macro done! → {ALT_DIR}")


# =========================================================
# 6.  VALIDATION
# =========================================================
def validate_all(force: bool = False) -> None:
    """Health check across all data directories."""
    _section("🏥 MASTER DATA HEALTH CHECK")

    print("\n1️⃣  PRICE DATA")
    if not PRICE_DIR.exists():
        print("❌ Price directory missing."); return
    price_files   = list(PRICE_DIR.glob("*.csv"))
    price_tickers = {f.stem for f in price_files}
    empty_prices  = {f.stem for f in price_files if f.stat().st_size < 100}
    valid_prices  = price_tickers - empty_prices
    print(f"   • Files found:     {len(price_files)}")
    print(f"   • Valid (>100 B):  {len(valid_prices)}")
    if empty_prices:
        print(f"   ⚠️  Empty:          {len(empty_prices)}  e.g. {list(empty_prices)[:3]}")

    print("\n2️⃣  FUNDAMENTAL DATA")
    if not FUND_DIR.exists():
        print("   ❌ Directory missing.")
        good_funds = set()
    else:
        fund_dirs  = [f for f in FUND_DIR.glob("*") if f.is_dir()]
        good_funds = {
            d.name for d in fund_dirs
            if (d / "info.csv").exists() or (d / "financials.csv").exists()
        }
        print(f"   • Folders found:   {len(fund_dirs)}")
        print(f"   • Usable content:  {len(good_funds)}")

    print("\n3️⃣  EARNINGS DATA")
    if not EARNINGS_DIR.exists():
        print("   ⚠️  Directory missing.")
        earn_tickers = set()
    else:
        earn_files   = list(EARNINGS_DIR.glob("*.csv"))
        earn_tickers = {f.stem for f in earn_files if f.stat().st_size > 50}
        print(f"   • Files found:     {len(earn_files)}")
        print(f"   • Valid content:   {len(earn_tickers)}")

    print("\n4️⃣  ALTERNATIVE DATA")
    found   = [m for m in MACRO_TICKERS if (ALT_DIR / f"{m}.csv").exists()]
    missing = [m for m in MACRO_TICKERS if m not in found]
    print(f"   • Found:           {found}")
    if missing:
        print(f"   ❌ Missing:         {missing}")
    else:
        print("   ✅ All macro indicators present.")

    print(f"\n{'=' * 60}\n🏆 UNIVERSE DIAGNOSIS")
    partial = valid_prices & good_funds
    full    = partial & earn_tickers
    print(f"   ✅ Price + Fundamentals:            {len(partial)} stocks")
    print(f"   🌟 Price + Fundamentals + Earnings: {len(full)} stocks  ← Golden Universe")

    if len(full) > 400:
        print("\n🟢 STATUS: EXCELLENT — ready for full alpha modelling.")
    elif len(partial) > 400:
        print("\n🟡 STATUS: GOOD — earnings limited, core pipeline solid.")
    else:
        print("\n🔴 STATUS: POOR — significant data gaps detected.")


# =========================================================
# MAIN
# =========================================================
STEPS: dict[str, Callable] = {
    "prices":       download_prices,
    "fundamentals": download_fundamentals,
    "earnings":     download_earnings,
    "macro":        download_macro,
    "validate":     validate_all,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step", default="all",
        choices=["all"] + list(STEPS.keys()),
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download data even if files already exist",
    )
    args = parser.parse_args()

    print("🚀 MASTER DATA PIPELINE")
    print(f"   Step : {args.step}")
    print(f"   Force: {args.force}")
    print(f"   Range: {START_DATE} → {END_DATE}")

    targets = list(STEPS.keys()) if args.step == "all" else [args.step]
    for step in targets:
        STEPS[step](force=args.force)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()