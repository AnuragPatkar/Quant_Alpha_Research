import pandas as pd
from pathlib import Path
import sys

# Setup Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PRICE_DIR = RAW_DIR / "sp500_prices"
FUND_DIR = RAW_DIR / "fundamentals"
EARN_DIR = RAW_DIR / "earnings"
ALT_DIR = RAW_DIR / "alternative"

def validate_all():
    print("🏥 Starting MASTER DATA HEALTH CHECK...")
    print("=" * 60)
    
    # 1. Price Validation
    print("\n1️⃣  PRICE DATA (OHLCV)")
    if not PRICE_DIR.exists():
        print("❌ CRITICAL: Price directory missing!")
        return
        
    price_files = list(PRICE_DIR.glob("*.csv"))
    price_tickers = set([f.stem for f in price_files])
    
    # Check for empty files
    empty_prices = [f.stem for f in price_files if f.stat().st_size < 100]
    valid_prices = price_tickers - set(empty_prices)
    
    print(f"   • Files Found:      {len(price_files)}")
    print(f"   • Valid (Non-0kb):  {len(valid_prices)}")
    if empty_prices:
        print(f"   ⚠️ Empty Files:      {len(empty_prices)} (e.g., {empty_prices[:3]})")
    
    # 2. Fundamental Validation
    print("\n2️⃣  FUNDAMENTAL DATA")
    if not FUND_DIR.exists():
        print("❌ CRITICAL: Fundamental directory missing!")
        return

    fund_folders = list(FUND_DIR.glob("*"))
    fund_tickers = set([f.name for f in fund_folders if f.is_dir()])
    
    # Check deep content
    good_funds = set()
    for t in fund_tickers:
        path = FUND_DIR / t
        if (path / "info.csv").exists() or (path / "financials.csv").exists():
            good_funds.add(t)
            
    print(f"   • Folders Found:    {len(fund_folders)}")
    print(f"   • Usable Content:   {len(good_funds)}")
    
    # 3. Earnings Validation
    print("\n3️⃣  EARNINGS DATA")
    if not EARN_DIR.exists():
        print("   ⚠️ Earnings directory missing. (Did you run download_earnings.py?)")
        earn_tickers = set()
    else:
        earn_files = list(EARN_DIR.glob("*.csv"))
        earn_tickers = set([f.stem for f in earn_files if f.stat().st_size > 50])
        print(f"   • Files Found:      {len(earn_files)}")
        print(f"   • Valid Content:    {len(earn_tickers)}")

    # 4. Alternative/Macro Validation
    print("\n4️⃣  ALTERNATIVE DATA")
    required_macro = ["VIX", "US_10Y", "OIL", "USD", "SP500"]
    missing_macro = []
    found_macro = []
    
    for m in required_macro:
        if (ALT_DIR / f"{m}.csv").exists():
            found_macro.append(m)
        else:
            missing_macro.append(m)
            
    print(f"   • Found:            {found_macro}")
    if missing_macro:
        print(f"   ❌ Missing:          {missing_macro}")
    else:
        print("   ✅ All Macro Indicators Present.")

    # 5. THE INTERSECTION (The Golden Universe)
    print("\n" + "=" * 60)
    print("🏆 UNIVERSE DIAGNOSIS")
    
    # Who has EVERYTHING?
    full_coverage = valid_prices.intersection(good_funds).intersection(earn_tickers)
    partial_coverage = valid_prices.intersection(good_funds)
    
    print(f"✅ PRICE + FUNDAMENTALS:       {len(partial_coverage)} stocks")
    print(f"🌟 PRICE + FUND + EARNINGS:    {len(full_coverage)} stocks (The 'Golden' Universe)")
    
    if len(full_coverage) > 400:
        print("\n🟢 STATUS: EXCELLENT. Ready for full Alpha Modeling.")
    elif len(partial_coverage) > 400:
        print("\n🟡 STATUS: GOOD. Strategy will focus on Price/Fundamentals (Earnings limited).")
    else:
        print("\n🔴 STATUS: POOR. Significant data gaps detected.")

if __name__ == "__main__":
    validate_all()