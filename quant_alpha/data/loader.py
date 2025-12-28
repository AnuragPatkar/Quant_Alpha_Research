"""
Data Loader Module - Fixed Version
-----------------------------------
With retry logic, delays, and better error handling.
"""

import sys
from pathlib import Path

# Path fix
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings, get_universe


class DataLoader:
    """
    Stock data loader with retry logic.
    """
    
    def __init__(self, universe: str = None):
        """
        Initialize DataLoader.
        
        Args:
            universe: 'nifty50' ya 'sp500' (sp500 recommended)
        """
        self.universe = universe or settings.data.universe
        self.start_date = settings.data.start_date
        self.end_date = settings.data.end_date
        self.tickers = get_universe(self.universe)
        
        print(f"📊 DataLoader initialized")
        print(f"   Universe: {self.universe.upper()}")
        print(f"   Stocks: {len(self.tickers)}")
        print(f"   Period: {self.start_date} to {self.end_date}")
    
    def download_single_stock(self, ticker: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Single stock download with retry logic.
        """
        for attempt in range(retries):
            try:
                # Add delay to avoid rate limiting
                time.sleep(0.5)
                
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    timeout=10
                )
                
                if df.empty or len(df) < settings.data.min_history_days:
                    return None
                
                # Handle multi-level columns (yfinance sometimes returns this)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize column names
                df.columns = [c.lower() for c in df.columns]
                
                # Rename if needed
                col_mapping = {
                    'adj close': 'adj_close',
                    'adjclose': 'adj_close'
                }
                df = df.rename(columns=col_mapping)
                
                # Ensure we have required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                if 'adj_close' in df.columns:
                    df['close'] = df['adj_close']
                
                for col in required:
                    if col not in df.columns:
                        return None
                
                df = df[required]
                df['ticker'] = ticker
                
                return df
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return None
        
        return None
    
    def download_batch(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download multiple stocks at once (faster).
        """
        print(f"\n🔽 Batch downloading {len(tickers)} stocks...")
        
        try:
            # Download all at once
            data = yf.download(
                tickers,
                start=self.start_date,
                end=self.end_date,
                progress=True,
                group_by='ticker',
                threads=True
            )
            
            stock_data = {}
            
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        df = data.copy()
                    else:
                        df = data[ticker].copy()
                    
                    if df.empty or len(df) < settings.data.min_history_days:
                        continue
                    
                    # Clean columns
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    if 'adj_close' in df.columns:
                        df['close'] = df['adj_close']
                    
                    required = ['open', 'high', 'low', 'close', 'volume']
                    df = df[required].dropna()
                    
                    if len(df) >= settings.data.min_history_days:
                        df['ticker'] = ticker
                        stock_data[ticker] = df
                        
                except Exception as e:
                    continue
            
            return stock_data
            
        except Exception as e:
            print(f"❌ Batch download failed: {e}")
            return {}
    
    def download_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Download all stocks - tries batch first, then individual.
        """
        # Try batch download first
        stock_data = self.download_batch(self.tickers)
        
        if len(stock_data) >= len(self.tickers) * 0.5:  # At least 50% success
            print(f"\n✅ Batch download successful: {len(stock_data)} stocks")
            return stock_data
        
        # Fallback to individual download
        print(f"\n⚠️ Batch failed, trying individual downloads...")
        
        stock_data = {}
        failed = []
        
        for ticker in tqdm(self.tickers, desc="Downloading"):
            df = self.download_single_stock(ticker)
            if df is not None:
                stock_data[ticker] = df
            else:
                failed.append(ticker)
        
        print(f"\n✅ Downloaded: {len(stock_data)} stocks")
        if failed:
            print(f"❌ Failed: {len(failed)} stocks")
        
        return stock_data
    
    def create_panel_data(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create panel data from stock dictionary.
        """
        if not stock_data:
            raise ValueError("No stock data to create panel!")
        
        print(f"\n📊 Creating panel data from {len(stock_data)} stocks...")
        
        all_dfs = []
        for ticker, df in stock_data.items():
            df = df.copy()
            df['date'] = df.index
            df = df.reset_index(drop=True)
            all_dfs.append(df)
        
        panel = pd.concat(all_dfs, ignore_index=True)
        panel['date'] = pd.to_datetime(panel['date'])
        panel = panel.set_index(['date', 'ticker']).sort_index()
        
        print(f"   Shape: {panel.shape}")
        print(f"   Date range: {panel.index.get_level_values('date').min().date()} to "
              f"{panel.index.get_level_values('date').max().date()}")
        print(f"   Stocks: {panel.index.get_level_values('ticker').nunique()}")
        
        return panel
    
    def load(self) -> pd.DataFrame:
        """
        Main load function with caching.
        """
        cache_path = ROOT / f"data/processed/panel_{self.universe}.pkl"
        
        # Try cache first
        try:
            panel = pd.read_pickle(cache_path)
            print(f"📁 Loaded from cache: {cache_path}")
            return panel
        except FileNotFoundError:
            pass
        
        # Download fresh
        stock_data = self.download_all_stocks()
        
        if not stock_data:
            raise ValueError("Failed to download any stocks! Check your internet connection.")
        
        panel = self.create_panel_data(stock_data)
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_pickle(cache_path)
        print(f"💾 Saved to: {cache_path}")
        
        return panel
    
    def get_benchmark(self) -> pd.DataFrame:
        """
        Download benchmark index.
        """
        ticker = "^NSEI" if self.universe == "nifty50" else "^GSPC"
        print(f"\n📈 Downloading benchmark: {ticker}")
        
        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            if 'adj_close' in df.columns:
                df['close'] = df['adj_close']
            
            df['returns'] = df['close'].pct_change()
            
            return df[['close', 'returns']]
        except Exception as e:
            print(f"⚠️ Benchmark download failed: {e}")
            return pd.DataFrame()


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print(f"Project Root: {ROOT}")
    print("="*50)
    
    # USE SP500 - More reliable!
    loader = DataLoader(universe="sp500")
    
    try:
        panel = loader.load()
        print("\n" + "="*50)
        print("✅ SUCCESS!")
        print(f"Panel shape: {panel.shape}")
        print("\nSample data:")
        print(panel.head(10))
    except Exception as e:
        print(f"\n❌ Error: {e}")