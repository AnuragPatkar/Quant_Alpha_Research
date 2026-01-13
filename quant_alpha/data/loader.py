"""
Data Loader
===========
Loads Stooq data from data/processed/ folder.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Add project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings


class DataLoader:
    """
    Load stock data from processed files.
    
    Handles both pickle and CSV formats.
    """
    
    def __init__(self):
        """Initialize DataLoader."""
        self.data_path = settings.data.panel_path
        self._data = None
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazy load data."""
        if self._data is None:
            self._data = self.load()
        return self._data
    
    def load(self) -> pd.DataFrame:
        """
        Load data from file.
        
        Returns:
            DataFrame with: date, ticker, open, high, low, close, volume
        """
        print("\n" + "="*60)
        print("📊 LOADING DATA")
        print("="*60)
        
        # Try pickle first since it's faster, fall back to CSV if needed
        pkl_path = self.data_path
        csv_path = self.data_path.with_suffix('.csv')
        
        df = None
        
        # Attempt to load from pickle
        if pkl_path.exists():
            try:
                print(f"   📁 Trying pickle: {pkl_path.name}")
                df = pd.read_pickle(pkl_path)
                print(f"   ✅ Loaded from pickle")
            except Exception as e:
                print(f"   ⚠️ Pickle failed: {e}")
                print(f"   📁 Trying CSV instead...")
        
        # If pickle didn't work, try CSV
        if df is None and csv_path.exists():
            try:
                print(f"   📁 Loading CSV: {csv_path.name}")
                df = pd.read_csv(csv_path)
                print(f"   ✅ Loaded from CSV")
                
                # Save as pickle for next time (much faster)
                try:
                    df.to_pickle(pkl_path)
                    print(f"   💾 Re-saved as pickle for faster loading")
                except:
                    pass  # Not critical if this fails
                    
            except Exception as e:
                print(f"   ❌ CSV failed: {e}")
        
        # Make sure we actually loaded something
        if df is None:
            raise FileNotFoundError(
                f"\n❌ Could not load data!\n"
                f"   Tried:\n"
                f"   - {pkl_path}\n"
                f"   - {csv_path}\n\n"
                f"   Run your data download script again."
            )
        
        # Clean up the data
        df = self._prepare(df)
        
        # Show what we loaded
        self._print_summary(df)
        
        return df
    
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data."""
        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove NaN
        df = df.dropna()
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print data summary."""
        n_stocks = df['ticker'].nunique()
        n_rows = len(df)
        n_days = df['date'].nunique()
        date_min = df['date'].min()
        date_max = df['date'].max()
        
        print(f"\n   📈 Stocks: {n_stocks}")
        print(f"   📅 Date Range: {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
        print(f"   📊 Trading Days: {n_days}")
        print(f"   📁 Total Rows: {n_rows:,}")
        
        # List stocks
        stocks = sorted(df['ticker'].unique())
        print(f"\n   📋 Available Stocks ({len(stocks)}):")
        for i in range(0, len(stocks), 10):
            print(f"      {', '.join(stocks[i:i+10])}")
    
    def get_stock(self, ticker: str) -> pd.DataFrame:
        """Get data for single stock."""
        return self.data[self.data['ticker'] == ticker].copy()
    
    def get_tickers(self) -> List[str]:
        """Get list of all tickers."""
        return sorted(self.data['ticker'].unique().tolist())
    
    def get_date_range(self) -> tuple:
        """Get (min_date, max_date)."""
        return self.data['date'].min(), self.data['date'].max()


# Test
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load()
    print("\n📊 Sample Data:")
    print(data.head(10))