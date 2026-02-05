import yfinance as yf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import warnings

# Suppress "No data" warnings from yfinance
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
EARNINGS_DIR = PROJECT_ROOT / "data" / "raw" / "earnings"
EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "sp500_prices"

# 🚀 Speed Config
MAX_WORKERS = 20  # Fast but polite

def fetch_earnings(ticker):
    save_path = EARNINGS_DIR / f"{ticker}.csv"
    
    # Skip if we already have it (Delete this check to force update)
    if save_path.exists():
        return f"⏭️ {ticker}"

    try:
        stock = yf.Ticker(ticker)
        
        # This pulls historical earnings dates + estimates + actuals
        # Note: This data can be sparse for some stocks
        earnings = stock.earnings_dates
        
        if earnings is not None and not earnings.empty:
            # Clean up
            earnings = earnings.reset_index()
            earnings = earnings.rename(columns={
                'Earnings Date': 'date',
                'EPS Estimate': 'eps_estimate',
                'Reported EPS': 'eps_actual',
                'Surprise(%)': 'surprise_pct'
            })
            
            # Save
            earnings.to_csv(save_path, index=False)
            return f"✅ {ticker}"
        else:
            return f"⚠️ {ticker} (No Data)"
            
    except Exception:
        return f"❌ {ticker} (Error)"

def run_earnings_download():
    # 1. Get Ticker List
    if not PRICE_DIR.exists():
        print("❌ Error: No price data found.")
        return
        
    tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]
    print(f"🚀 Downloading Earnings History for {len(tickers)} stocks...")
    
    # 2. Parallel Download
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_earnings, t): t for t in tickers}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tickers)):
            # Just consuming the iterator to update progress bar
            pass

    print(f"\n🏆 Earnings Download Complete!")
    print(f"📂 Data stored in: {EARNINGS_DIR}")

if __name__ == "__main__":
    run_earnings_download()