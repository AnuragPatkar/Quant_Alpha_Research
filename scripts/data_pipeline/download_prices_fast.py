import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import io
import sys

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config

DATA_DIR = config.PRICES_DIR
START_DATE = config.BACKTEST_START_DATE
END_DATE = config.BACKTEST_END_DATE

def get_sp500_tickers():
    print("📋 Fetching S&P 500 ticker list from Wikipedia (Stealth Mode)...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # 1. Fake the User-Agent (Pretend to be Chrome)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 2. Get the HTML content manually with headers
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for errors (403/404)
        
        # 3. Parse with Pandas
        # we wrap the text in StringIO because pandas expects a file-like object
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0] # The first table is the constituents list
        
        # 4. Extract symbols
        tickers = df['Symbol'].tolist()
        
        # 5. Clean symbols for Yahoo (BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        
        print(f"✅ Success! Found {len(tickers)} tickers.")
        return tickers

    except Exception as e:
        print(f"❌ Critical Error fetching tickers: {e}")
        print("💡 Try checking your internet connection or the URL.")
        sys.exit(1) # Stop the script if we can't get the list

def download_prices_fast():
    tickers = get_sp500_tickers()
    print(f"🚀 Downloading 8 years of data for {len(tickers)} stocks...")
    
    # We download EVERYTHING in one massive vectorized request
    # yfinance handles the threading internally for this function
    try:
        data = yf.download(
            tickers, 
            start=START_DATE, 
            end=END_DATE, 
            group_by='ticker', 
            auto_adjust=True, # Adjusts for splits/dividends automatically
            threads=True,     # ENABLE MULTI-THREADING
            progress=True
        )
        
        print("\n💾 Saving individual CSVs...")
        
        # The data comes as a MultiIndex DataFrame. We need to split it.
        count = 0
        for ticker in tickers:
            try:
                # Extract specific ticker dataframe
                df = data[ticker].copy()
                
                # Drop rows with all NaNs (non-trading days for that stock)
                df = df.dropna(how='all')
                
                if not df.empty:
                    # Clean up
                    df = df.reset_index()
                    df.columns = [c.lower() for c in df.columns] # Lowercase cols
                    df = df.rename(columns={'date': 'date', 'close': 'close', 'volume': 'volume'})
                    
                    # Save
                    df.to_csv(DATA_DIR / f"{ticker}.csv", index=False)
                    count += 1
            except KeyError:
                print(f"⚠️ Data missing for {ticker}")
                continue
                
        print(f"\n🏆 Price Download Complete! Saved {count} stocks to {DATA_DIR}")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    download_prices_fast()