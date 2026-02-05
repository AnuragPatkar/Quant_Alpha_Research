import yfinance as yf
import pandas as pd
from pathlib import Path
import time
import random

# Paths - Professional structure
PROJECT_ROOT = Path(__file__).parent.parent
ALT_DIR = PROJECT_ROOT / "data" / "raw" / "alternative"
ALT_DIR.mkdir(parents=True, exist_ok=True)

# Key Macro Assets
MACRO_TICKERS = {
    "VIX": "^VIX",         # Volatility (Fear Index)
    "US_10Y": "^TNX",      # 10-Year Treasury Yield
    "OIL": "CL=F",         # Crude Oil Futures
    "USD": "DX-Y.NYB",     # US Dollar Index
    "SP500": "^GSPC"       # S&P 500 Index
}

def download_macro():
    print("🚀 Starting Alternative Data Download (Enhanced Mode)...")
    
    start_date = "2016-01-01"
    end_date = "2024-01-01"
    
    for name, ticker in MACRO_TICKERS.items():
        print(f"⬇️ Fetching {name} ({ticker})...")
        
        # 1. Anti-Blocking: Rate limiting se bachne ke liye
        time.sleep(random.uniform(1.5, 3.0))
        
        try:
            # Ticker object create karke history fetch karna zyada stable hai
            asset = yf.Ticker(ticker)
            df = asset.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                print(f"⚠️ Warning: No data returned for {name}")
                continue

            # 2. Data Cleaning & Normalization
            df = df.reset_index()
            
            # Column names ko lowercase karna (taaki matching mein dikkat na ho)
            df.columns = [str(c).lower() for c in df.columns]
            
            # Timezone handling: Date se timezone hatao (Quant models ke liye simple date best hai)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            # 3. Specific Columns Selection
            # Hum sirf Date, Close aur Volume par focus kar rahe hain
            if 'close' in df.columns:
                cols_to_keep = ['date', 'close']
                if 'volume' in df.columns:
                    cols_to_keep.append('volume')
                
                df = df[cols_to_keep]
                
                # Column names ko clear banayein: e.g., 'vix_close'
                rename_map = {
                    'close': f'{name.lower()}_close',
                    'volume': f'{name.lower()}_volume'
                }
                df = df.rename(columns=rename_map)

                # 4. Save to CSV
                output_file = ALT_DIR / f"{name}.csv"
                df.to_csv(output_file, index=False)
                print(f"✅ Saved {name}.csv ({len(df)} rows)")
            
        except Exception as e:
            print(f"❌ Failed to fetch {name}: {str(e)}")

    print(f"\n🏆 Alternative Data Update Complete!")
    print(f"📂 Location: {ALT_DIR}")

if __name__ == "__main__":
    download_macro()