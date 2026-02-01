#!/usr/bin/env python3
"""
Data Validation Script
======================
Uses the quant_alpha package to validate data.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import print_welcome
from quant_alpha.data import DataLoader, DataValidationError

def main():
    print_welcome()
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    try:
        # Initialize Loader with validation enabled
        print("\n[INFO] Initializing DataLoader...")
        loader = DataLoader(validate=True, verbose=True)
        
        # Load and Validate
        print("[INFO] Running validation checks...")
        df = loader.load()
        
        # Get Summary
        stats = loader.summary_stats()
        
        print("\n" + "="*60)
        print("[SUCCESS] DATA IS VALID AND READY FOR ML")
        print("="*60)
        print(f"   Stocks:       {df['ticker'].nunique()}")
        print(f"   Total Rows:   {len(df):,}")
        print(f"   Date Range:   {df['date'].min().date()} to {df['date'].max().date()}")
        print("-" * 60)
        
    except DataValidationError as e:
        print("\n" + "="*60)
        print("[ERROR] DATA VALIDATION FAILED")
        print("="*60)
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print("\n[ERROR] Data file not found. Run scripts/fetch_data.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()