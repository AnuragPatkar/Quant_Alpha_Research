import yfinance as yf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import concurrent.futures # The secret weapon for speed

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

RAW_FUND_DIR = config.FUNDAMENTALS_DIR
PRICE_DIR = config.PRICES_DIR

# Safety Limit: Too many workers = IP Ban from Yahoo
MAX_WORKERS = 10 

def fetch_single_stock_fundamental(ticker):
    """
    Fetches data for a single stock. 
    This function will be run in parallel.
    """
    save_path = RAW_FUND_DIR / ticker
    
    # Skip if already downloaded (remove this check if you want to force update)
    if save_path.exists() and (save_path / "info.csv").exists():
        return f"⏭️ {ticker} (Skipped)"

    try:
        stock = yf.Ticker(ticker)
        save_path.mkdir(exist_ok=True)
        
        # 1. Get Info (Key Ratios)
        info = stock.info
        pd.DataFrame([info]).to_csv(save_path / "info.csv", index=False)
        
        # 2. Get Financial Statements
        # We only save if not empty
        fin = stock.financials
        if not fin.empty: fin.to_csv(save_path / "financials.csv")
        
        bs = stock.balance_sheet
        if not bs.empty: bs.to_csv(save_path / "balance_sheet.csv")
        
        cf = stock.cashflow
        if not cf.empty: cf.to_csv(save_path / "cashflow.csv")
        
        return f"✅ {ticker}"
        
    except Exception as e:
        return f"❌ {ticker} (Error)"

def download_fundamentals_fast():
    # Get list of tickers from the Price directory
    if not PRICE_DIR.exists():
        print("❌ Run the price downloader first!")
        return
        
    tickers = [f.stem for f in PRICE_DIR.glob("*.csv")]
    print(f"🚀 Starting Parallel Fundamental Download for {len(tickers)} stocks...")
    print(f"⚡ Using {MAX_WORKERS} threads for maximum speed.")

    # Execute in Parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_single_stock_fundamental, t): t for t in tickers}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tickers)):
            result = future.result()
            # print(result) # Uncomment to see individual status logs (spammy)

    print(f"\n🏆 Fundamentals Download Complete!")
    print(f"📂 Data stored in: {RAW_FUND_DIR}")

if __name__ == "__main__":
    download_fundamentals_fast()