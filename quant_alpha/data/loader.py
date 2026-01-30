"""
Data Loader
===========
Robust data loading with validation, filtering, and quality checks.

Features:
- Date range filtering based on config
- Universe validation
- Data quality checks
- Proper error handling
- Logging instead of prints

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from datetime import datetime
import logging
import warnings

# Proper relative imports (no sys.path hacking)
try:
    from config.settings import settings, get_universe
except ImportError:
    # Fallback for when running as script
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings, get_universe


# Set up module logger
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class DataLoader:
    """
    Load and validate stock data from processed files.
    
    Features:
        - Automatic date range filtering from config
        - Universe validation against config
        - Data quality checks (missing data, price sanity)
        - Support for pickle and CSV formats
        - Lazy loading with caching
    
    Usage:
        >>> loader = DataLoader()
        >>> df = loader.load()
        >>> aapl = loader.get_stock('AAPL')
        >>> tickers = loader.get_tickers()
    
    Attributes:
        data_path: Path to data file
        config: Reference to settings
    """
    
    # Expected columns in data file
    REQUIRED_COLUMNS = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume'}
    OPTIONAL_COLUMNS = {'adj_close', 'adj_volume', 'dividends', 'splits'}
    NUMERIC_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        universe: Optional[List[str]] = None,
        validate: bool = True,
        verbose: bool = True
    ):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Custom path to data file (default: from config)
            start_date: Override start date (default: from config)
            end_date: Override end date (default: from config)
            universe: Override stock universe (default: from config)
            validate: Whether to run data validation checks
            verbose: Whether to print loading summary
        """
        self.data_path = data_path or settings.data.panel_path
        self.start_date = pd.to_datetime(start_date or settings.data.start_date)
        self.end_date = pd.to_datetime(end_date or settings.data.effective_end_date)
        self.universe = universe or get_universe(settings.data.universe)
        self.validate = validate
        self.verbose = verbose
        
        # Internal state
        self._data: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._ticker_index: Optional[Dict[str, pd.DataFrame]] = None
        self._load_stats: Dict = {}
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Lazy load data on first access.
        
        Returns:
            DataFrame with OHLCV data
        """
        if self._data is None:
            self._data = self.load()
        return self._data
    
    @property
    def returns(self) -> pd.DataFrame:
        """
        Get returns data (lazy calculated).
        
        Returns:
            DataFrame with daily returns
        """
        if self._returns is None:
            self._returns = self._calculate_returns()
        return self._returns
    
    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load data from file with full validation.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Cleaned and validated DataFrame
            
        Raises:
            FileNotFoundError: If data file not found
            DataValidationError: If data fails validation
        """
        if self._data is not None and not force_reload:
            return self._data
        
        self._log_header("LOADING DATA")
        
        # Load raw data
        df = self._load_file()
        
        # Prepare and clean
        df = self._prepare(df)
        
        # Filter by date range
        df = self._filter_dates(df)
        
        # Filter by universe
        df = self._filter_universe(df)
        
        # Validate data quality
        if self.validate:
            self._validate_data(df)
        
        # Print summary
        if self.verbose:
            self._print_summary(df)
        
        # Cache and return
        self._data = df
        self._ticker_index = None  # Reset index cache
        self._returns = None  # Reset returns cache
        
        return df
    
    def _load_file(self) -> pd.DataFrame:
        """
        Load data from pickle or CSV file.
        
        Returns:
            Raw DataFrame from file
            
        Raises:
            FileNotFoundError: If no valid file found
        """
        pkl_path = Path(self.data_path)
        csv_path = pkl_path.with_suffix('.csv')
        
        df = None
        load_source = None
        
        # Try pickle first (faster)
        if pkl_path.exists() and pkl_path.suffix == '.pkl':
            try:
                logger.info(f"Loading pickle: {pkl_path.name}")
                df = pd.read_pickle(pkl_path)
                load_source = 'pickle'
            except Exception as e:
                logger.warning(f"Pickle load failed: {e}")
        
        # Fallback to CSV
        if df is None and csv_path.exists():
            try:
                logger.info(f"Loading CSV: {csv_path.name}")
                df = pd.read_csv(csv_path, parse_dates=['date'])
                load_source = 'csv'
                
                # Cache as pickle for next time
                self._save_pickle_cache(df, pkl_path)
                
            except Exception as e:
                logger.error(f"CSV load failed: {e}")
        
        # Also try if data_path points directly to CSV
        if df is None and pkl_path.suffix == '.csv' and pkl_path.exists():
            try:
                df = pd.read_csv(pkl_path, parse_dates=['date'])
                load_source = 'csv'
            except Exception as e:
                logger.error(f"Direct CSV load failed: {e}")
        
        if df is None:
            raise FileNotFoundError(
                f"\n❌ Could not load data!\n"
                f"   Looked for:\n"
                f"   - {pkl_path}\n"
                f"   - {csv_path}\n\n"
                f"   Please run data download script first:\n"
                f"   python scripts/download_data.py"
            )
        
        self._load_stats['source'] = load_source
        self._load_stats['raw_rows'] = len(df)
        
        logger.info(f"Loaded {len(df):,} rows from {load_source}")
        
        return df
    
    def _save_pickle_cache(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame as pickle for faster future loads."""
        try:
            df.to_pickle(path)
            logger.info(f"Cached as pickle: {path.name}")
        except Exception as e:
            logger.debug(f"Could not cache pickle: {e}")
    
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Standardize column names
        df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
        
        # Validate required columns exist
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Standardize ticker format
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        
        # Convert numeric columns
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Track missing data before dropping
        missing_before = df.isna().sum().sum()
        rows_before = len(df)
        
        # Remove rows with NaN in critical columns
        critical_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        critical_cols = [c for c in critical_cols if c in df.columns]
        df = df.dropna(subset=critical_cols)
        
        rows_after = len(df)
        rows_dropped = rows_before - rows_after
        
        if rows_dropped > 0:
            drop_pct = rows_dropped / rows_before * 100
            logger.info(f"Dropped {rows_dropped:,} rows ({drop_pct:.2f}%) with missing data")
            self._load_stats['rows_dropped_missing'] = rows_dropped
        
        # Remove invalid data
        df = self._remove_invalid_data(df)
        
        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        
        return df
    
    def _remove_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with invalid/corrupt data.
        
        Checks for:
        - Negative prices
        - Zero prices
        - High < Low
        - Zero volume (non-trading days)
        - Extreme price jumps (potential split issues)
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        rows_before = len(df)
        invalid_masks = []
        
        # Check 1: Negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                mask = df[col] < 0
                if mask.any():
                    logger.warning(f"Found {mask.sum()} rows with negative {col}")
                    invalid_masks.append(mask)
        
        # Check 2: Zero prices
        for col in price_cols:
            if col in df.columns:
                mask = df[col] == 0
                if mask.any():
                    logger.warning(f"Found {mask.sum()} rows with zero {col}")
                    invalid_masks.append(mask)
        
        # Check 3: High < Low (data error)
        if 'high' in df.columns and 'low' in df.columns:
            mask = df['high'] < df['low']
            if mask.any():
                logger.warning(f"Found {mask.sum()} rows with high < low")
                invalid_masks.append(mask)
        
        # Check 4: Zero volume (likely non-trading day or halt)
        if 'volume' in df.columns:
            mask = df['volume'] == 0
            if mask.any():
                logger.info(f"Found {mask.sum()} rows with zero volume (removing)")
                invalid_masks.append(mask)
        
        # Check 5: Extreme daily moves (>50% in one day, likely data error or unadjusted split)
        if 'close' in df.columns:
            df_sorted = df.sort_values(['ticker', 'date'])
            daily_return = df_sorted.groupby('ticker')['close'].pct_change()
            mask = daily_return.abs() > 0.50  # 50% move
            if mask.any():
                extreme_count = mask.sum()
                logger.warning(
                    f"Found {extreme_count} rows with >50% daily move. "
                    f"Check for stock splits or data errors."
                )
                # Don't remove these automatically - just warn
                # User should investigate and use adjusted data
        
        # Combine all invalid masks
        if invalid_masks:
            combined_mask = pd.concat(invalid_masks, axis=1).any(axis=1)
            df = df[~combined_mask].copy()
            
            rows_removed = rows_before - len(df)
            logger.info(f"Removed {rows_removed:,} invalid rows")
            self._load_stats['rows_dropped_invalid'] = rows_removed
        
        return df
    
    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to configured date range.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Date-filtered DataFrame
        """
        rows_before = len(df)
        
        # Apply date filter
        mask = (df['date'] >= self.start_date) & (df['date'] <= self.end_date)
        df = df[mask].copy()
        
        rows_after = len(df)
        rows_filtered = rows_before - rows_after
        
        if rows_filtered > 0:
            logger.info(
                f"Date filter [{self.start_date.date()} to {self.end_date.date()}]: "
                f"kept {rows_after:,} of {rows_before:,} rows"
            )
        
        self._load_stats['rows_after_date_filter'] = rows_after
        
        # Validate we have data in range
        if len(df) == 0:
            raise DataValidationError(
                f"No data found in date range "
                f"{self.start_date.date()} to {self.end_date.date()}"
            )
        
        return df
    
    def _filter_universe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to configured stock universe.
        
        Args:
            df: DataFrame to filter
            
        Returns:
            Universe-filtered DataFrame
        """
        available_tickers = set(df['ticker'].unique())
        requested_tickers = set(self.universe)
        
        # Find matches and mismatches
        matched = available_tickers & requested_tickers
        missing_from_data = requested_tickers - available_tickers
        extra_in_data = available_tickers - requested_tickers
        
        # Log warnings
        if missing_from_data:
            logger.warning(
                f"Tickers in config but not in data ({len(missing_from_data)}): "
                f"{sorted(missing_from_data)[:10]}{'...' if len(missing_from_data) > 10 else ''}"
            )
        
        if extra_in_data and self.verbose:
            logger.info(
                f"Tickers in data but not in config ({len(extra_in_data)}): "
                f"will be excluded"
            )
        
        # Filter to matched tickers
        df = df[df['ticker'].isin(matched)].copy()
        
        self._load_stats['tickers_matched'] = len(matched)
        self._load_stats['tickers_missing'] = len(missing_from_data)
        
        # Validate minimum stocks
        if len(matched) < settings.data.min_stocks:
            raise DataValidationError(
                f"Only {len(matched)} stocks matched (minimum: {settings.data.min_stocks})\n"
                f"Missing tickers: {sorted(missing_from_data)}"
            )
        
        logger.info(f"Universe filter: {len(matched)} of {len(requested_tickers)} tickers matched")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Run comprehensive data validation checks.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        errors = []
        warnings_list = []
        
        # Check 1: Minimum history per stock
        min_days = settings.data.min_history_days
        stock_counts = df.groupby('ticker')['date'].nunique()
        stocks_short_history = stock_counts[stock_counts < min_days]
        
        if len(stocks_short_history) > 0:
            warnings_list.append(
                f"{len(stocks_short_history)} stocks have < {min_days} days of history: "
                f"{list(stocks_short_history.index[:5])}"
            )
        
        # Check 2: Missing data percentage
        total_possible = df['ticker'].nunique() * df['date'].nunique()
        actual_rows = len(df)
        missing_pct = 1 - (actual_rows / total_possible)
        
        if missing_pct > settings.data.max_missing_pct:
            errors.append(
                f"Missing data ({missing_pct:.1%}) exceeds threshold "
                f"({settings.data.max_missing_pct:.1%})"
            )
        
        # Check 3: Date gaps
        date_range = pd.date_range(
            df['date'].min(), 
            df['date'].max(), 
            freq='B'  # Business days
        )
        actual_dates = df['date'].unique()
        missing_dates = set(date_range) - set(actual_dates)
        
        # More than 5% missing business days is suspicious
        if len(missing_dates) / len(date_range) > 0.05:
            warnings_list.append(
                f"Found {len(missing_dates)} missing business days "
                f"({len(missing_dates)/len(date_range):.1%} of expected)"
            )
        
        # Check 4: Duplicate rows
        duplicates = df.duplicated(subset=['date', 'ticker'], keep=False)
        if duplicates.any():
            dup_count = duplicates.sum() // 2  # Each duplicate counted twice
            errors.append(f"Found {dup_count} duplicate (date, ticker) pairs")
        
        # Log warnings
        for warn in warnings_list:
            logger.warning(warn)
        
        # Raise if errors
        if errors:
            raise DataValidationError(
                "Data validation failed:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
        
        logger.info("Data validation passed ✓")
    
    def _calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns for all stocks.
        
        Returns:
            DataFrame with returns added
        """
        df = self.data.copy()
        
        # Sort for proper calculation
        df = df.sort_values(['ticker', 'date'])
        
        # Calculate returns within each ticker
        df['return'] = df.groupby('ticker')['close'].pct_change()
        
        # Also calculate log returns (useful for some models)
        df['log_return'] = np.log1p(df['return'])
        
        # Forward returns (target variable)
        fwd_days = settings.features.forward_return_days
        df['forward_return'] = df.groupby('ticker')['close'].pct_change(fwd_days).shift(-fwd_days)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print comprehensive data summary."""
        n_stocks = df['ticker'].nunique()
        n_rows = len(df)
        n_days = df['date'].nunique()
        date_min = df['date'].min()
        date_max = df['date'].max()
        
        print(f"\n   📈 Loaded Data Summary:")
        print(f"   {'─' * 40}")
        print(f"   Stocks:        {n_stocks}")
        print(f"   Date Range:    {date_min.strftime('%Y-%m-%d')} → {date_max.strftime('%Y-%m-%d')}")
        print(f"   Trading Days:  {n_days:,}")
        print(f"   Total Rows:    {n_rows:,}")
        print(f"   Rows/Stock:    {n_rows // n_stocks:,} avg")
        
        # Data quality summary
        if self._load_stats:
            print(f"\n   📊 Data Quality:")
            if 'rows_dropped_missing' in self._load_stats:
                print(f"   Dropped (missing): {self._load_stats['rows_dropped_missing']:,}")
            if 'rows_dropped_invalid' in self._load_stats:
                print(f"   Dropped (invalid): {self._load_stats['rows_dropped_invalid']:,}")
            if 'tickers_missing' in self._load_stats:
                print(f"   Tickers missing:   {self._load_stats['tickers_missing']}")
        
        # List stocks
        stocks = sorted(df['ticker'].unique())
        print(f"\n   📋 Stocks ({len(stocks)}):")
        for i in range(0, len(stocks), 10):
            print(f"      {', '.join(stocks[i:i+10])}")
    
    def _log_header(self, title: str) -> None:
        """Print section header."""
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"📊 {title}")
            print("=" * 60)
    
    # ==========================================
    # PUBLIC METHODS - Data Access
    # ==========================================
    
    def get_stock(
        self, 
        ticker: str, 
        include_returns: bool = False
    ) -> pd.DataFrame:
        """
        Get data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            include_returns: Whether to include return columns
            
        Returns:
            DataFrame for specified ticker
            
        Raises:
            ValueError: If ticker not found
        """
        ticker = ticker.upper().strip()
        
        # Validate ticker exists
        available = self.get_tickers()
        if ticker not in available:
            raise ValueError(
                f"Ticker '{ticker}' not found. "
                f"Available: {available[:10]}..."
            )
        
        # Use returns data if requested
        source = self.returns if include_returns else self.data
        
        return source[source['ticker'] == ticker].copy()
    
    def get_stocks(
        self, 
        tickers: List[str],
        include_returns: bool = False
    ) -> pd.DataFrame:
        """
        Get data for multiple stocks.
        
        Args:
            tickers: List of ticker symbols
            include_returns: Whether to include return columns
            
        Returns:
            DataFrame for specified tickers
        """
        tickers = [t.upper().strip() for t in tickers]
        
        # Validate all tickers
        available = set(self.get_tickers())
        invalid = set(tickers) - available
        if invalid:
            logger.warning(f"Tickers not found (skipping): {invalid}")
            tickers = [t for t in tickers if t in available]
        
        source = self.returns if include_returns else self.data
        
        return source[source['ticker'].isin(tickers)].copy()
    
    def get_tickers(self) -> List[str]:
        """
        Get list of all available tickers.
        
        Returns:
            Sorted list of ticker symbols
        """
        return sorted(self.data['ticker'].unique().tolist())
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get data date range.
        
        Returns:
            Tuple of (min_date, max_date)
        """
        return self.data['date'].min(), self.data['date'].max()
    
    def get_dates(self) -> pd.DatetimeIndex:
        """
        Get all unique dates in data.
        
        Returns:
            DatetimeIndex of trading dates
        """
        return pd.DatetimeIndex(sorted(self.data['date'].unique()))
    
    def get_cross_section(self, date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get all stock data for a specific date.
        
        Args:
            date: Date string or datetime
            
        Returns:
            DataFrame with all stocks for that date
        """
        date = pd.to_datetime(date)
        df = self.data[self.data['date'] == date].copy()
        
        if len(df) == 0:
            # Find nearest date
            all_dates = self.get_dates()
            nearest = all_dates[all_dates.get_indexer([date], method='nearest')[0]]
            logger.warning(f"No data for {date.date()}, using nearest: {nearest.date()}")
            df = self.data[self.data['date'] == nearest].copy()
        
        return df
    
    def get_pivot(
        self, 
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Get data in pivot table format (dates × tickers).
        
        Args:
            column: Column to pivot ('close', 'volume', etc.)
            
        Returns:
            DataFrame with dates as index, tickers as columns
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(self.data.columns)}")
        
        return self.data.pivot(
            index='date', 
            columns='ticker', 
            values=column
        )
    
    def get_returns_pivot(self) -> pd.DataFrame:
        """
        Get returns in pivot table format.
        
        Returns:
            DataFrame with dates as index, tickers as columns, values as returns
        """
        _ = self.returns  # Ensure calculated
        return self.returns.pivot(
            index='date',
            columns='ticker',
            values='return'
        )
    
    def summary_stats(self) -> pd.DataFrame:
        """
        Calculate summary statistics for all stocks.
        
        Returns:
            DataFrame with statistics per ticker
        """
        _ = self.returns  # Ensure returns calculated
        
        stats = self.returns.groupby('ticker').agg({
            'close': ['first', 'last', 'mean', 'std', 'min', 'max'],
            'volume': ['mean', 'std'],
            'return': ['mean', 'std', 'min', 'max'],
            'date': ['min', 'max', 'count']
        })
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns]
        
        # Add derived metrics
        stats['total_return'] = stats['close_last'] / stats['close_first'] - 1
        stats['ann_return'] = stats['return_mean'] * 252
        stats['ann_volatility'] = stats['return_std'] * np.sqrt(252)
        stats['sharpe'] = stats['ann_return'] / stats['ann_volatility']
        
        return stats.round(4)
    
    def __repr__(self) -> str:
        """String representation."""
        if self._data is None:
            return f"DataLoader(path='{self.data_path}', loaded=False)"
        
        return (
            f"DataLoader("
            f"stocks={self.data['ticker'].nunique()}, "
            f"rows={len(self.data):,}, "
            f"dates={self.data['date'].min().date()} to {self.data['date'].max().date()}"
            f")"
        )


# ==========================================
# MODULE TEST
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING DATA LOADER")
    print("=" * 60)
    
    try:
        # Test basic loading
        loader = DataLoader()
        data = loader.load()
        
        print(f"\n✅ Basic load successful")
        print(f"   {loader}")
        
        # Test single stock
        print("\n📊 Testing get_stock('AAPL'):")
        aapl = loader.get_stock('AAPL', include_returns=True)
        print(f"   Rows: {len(aapl)}")
        print(f"   Columns: {list(aapl.columns)}")
        
        # Test pivot
        print("\n📊 Testing get_pivot():")
        pivot = loader.get_pivot('close')
        print(f"   Shape: {pivot.shape}")
        
        # Test summary stats
        print("\n📊 Testing summary_stats():")
        stats = loader.summary_stats()
        print(stats.head())
        
        # Test invalid ticker
        print("\n📊 Testing invalid ticker:")
        try:
            loader.get_stock('INVALID_TICKER')
        except ValueError as e:
            print(f"   ✅ Correctly raised: {e}")
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\n⚠️ No data file found (expected for fresh install)")
        print(f"   {e}")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()