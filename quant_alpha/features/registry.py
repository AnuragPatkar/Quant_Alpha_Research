"""
Factor Registry
===============
Central registry for managing all factors and computing features.

Features:
- Register and organize factors by category
- Batch computation for all stocks
- Cross-sectional normalization
- Winsorization
- Feature caching
- Target variable creation

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable, Union, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import pickle
from datetime import datetime

try:
    from quant_alpha.features.base import (
        BaseFactor,
        FactorInfo,
        FactorCategory,
        FactorGroup,
        FactorValidationError,
    )
    from quant_alpha.features.momentum import get_momentum_factors
    from quant_alpha.features.mean_reversion import get_mean_reversion_factors
    from quant_alpha.features.microstructure import get_microstructure_factors
    from config.settings import settings
except ImportError:
    from .base import (
        BaseFactor,
        FactorInfo,
        FactorCategory,
        FactorGroup,
        FactorValidationError,
    )
    from .momentum import get_momentum_factors
    from .mean_reversion import get_mean_reversion_factors
    from .microstructure import get_microstructure_factors
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings


logger = logging.getLogger(__name__)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def winsorize_series(
    series: pd.Series,
    limits: Tuple[float, float] = (0.01, 0.99)
) -> pd.Series:
    """
    Winsorize a series by clipping extreme values to quantile limits.
    
    Args:
        series: Input series
        limits: (lower_quantile, upper_quantile) e.g., (0.01, 0.99)
        
    Returns:
        Winsorized series with extreme values clipped
        
    Example:
        >>> s = pd.Series([1, 2, 3, 100, 5])
        >>> winsorize_series(s, (0.1, 0.9))
    """
    if series.isna().all():
        return series
    
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    
    return series.clip(lower=lower, upper=upper)


def winsorize_by_group(
    df: pd.DataFrame,
    columns: List[str],
    group_column: str = 'date',
    limits: Tuple[float, float] = (0.01, 0.99)
) -> pd.DataFrame:
    """
    Winsorize columns within each group - OPTIMIZED.
    """
    if not columns:
        return df
    
    df_result = df.copy()
    valid_cols = [c for c in columns if c in df_result.columns]
    
    if not valid_cols:
        return df_result
    
    # Vectorized quantile computation
    def clip_group(group):
        for col in valid_cols:
            lower = group[col].quantile(limits[0])
            upper = group[col].quantile(limits[1])
            group[col] = group[col].clip(lower=lower, upper=upper)
        return group
    
    df_result = df_result.groupby(group_column, group_keys=False).apply(clip_group)
    
    return df_result


def normalize_cross_section(
    df: pd.DataFrame,
    method: str = 'rank',
    date_column: str = 'date',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Normalize features cross-sectionally - OPTIMIZED.
    """
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found")
        return df
    
    df_result = df.copy()
    
    # Auto-detect feature columns
    if columns is None:
        non_feature_cols = {
            date_column, 'ticker', 'symbol', 
            'forward_return', 'return', 'log_return',
            'close', 'open', 'high', 'low', 'volume', 
            'adj_close', 'adj_volume', 'dividends', 'splits'
        }
        columns = [c for c in df.columns if c not in non_feature_cols]
    
    if not columns:
        return df
    
    # Vectorized normalization
    if method == 'rank':
        # Process all columns at once
        df_result[columns] = df.groupby(date_column)[columns].transform(
            lambda x: x.rank(pct=True, na_option='keep')
        )
    elif method == 'zscore':
        df_result[columns] = df.groupby(date_column)[columns].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    elif method == 'minmax':
        df_result[columns] = df.groupby(date_column)[columns].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.debug(f"Normalized {len(columns)} features using {method}")
    
    return df_result


def create_target_variable(
    df: pd.DataFrame,
    forward_days: Optional[int] = None,
    target_column: str = 'forward_return',
    price_column: str = 'close',
    ticker_column: str = 'ticker'
) -> pd.DataFrame:
    """
    Create target variable (forward returns) for ML training.
    
    IMPORTANT: This creates the target by looking forward in time.
    The target at time t is the return from t to t+forward_days.
    
    Args:
        df: DataFrame with [date, ticker, close]
        forward_days: Forward return horizon in trading days (default: from config)
        target_column: Name for target column
        price_column: Column to use for price
        ticker_column: Column to group by
        
    Returns:
        DataFrame with target column added
        
    Warning:
        The last `forward_days` rows for each stock will have NaN target.
        This is correct - we don't know future returns!
    """
    if forward_days is None:
        forward_days = settings.features.forward_return_days
    
    df_result = df.copy()
    
    # Sort by ticker and date
    df_result = df_result.sort_values([ticker_column, 'date'])
    
    # Calculate forward return for each ticker
    # shift(-forward_days) looks into the future
    df_result[target_column] = df_result.groupby(ticker_column)[price_column].transform(
        lambda x: x.pct_change(periods=forward_days).shift(-forward_days)
    )
    
    # Count NaN targets (expected at the end of each stock's history)
    n_nan = df_result[target_column].isna().sum()
    expected_nan = df_result[ticker_column].nunique() * forward_days
    
    logger.info(f"Created target '{target_column}' with {forward_days}-day forward return")
    logger.debug(f"NaN targets: {n_nan} (expected ~{expected_nan})")
    
    return df_result


def calculate_forward_returns_multi(
    df: pd.DataFrame,
    horizons: List[int] = [5, 10, 21, 63],
    price_column: str = 'close',
    ticker_column: str = 'ticker'
) -> pd.DataFrame:
    """
    Calculate forward returns for multiple horizons.
    
    Useful for analyzing alpha decay across different holding periods.
    
    Args:
        df: Input DataFrame
        horizons: List of forward return horizons
        price_column: Price column to use
        ticker_column: Ticker column for grouping
        
    Returns:
        DataFrame with multiple forward return columns
    """
    df_result = df.copy()
    df_result = df_result.sort_values([ticker_column, 'date'])
    
    for horizon in horizons:
        col_name = f'forward_return_{horizon}d'
        df_result[col_name] = df_result.groupby(ticker_column)[price_column].transform(
            lambda x: x.pct_change(periods=horizon).shift(-horizon)
        )
    
    logger.info(f"Created forward returns for horizons: {horizons}")
    
    return df_result


# ============================================
# FACTOR REGISTRY CLASS
# ============================================

class FactorRegistry:
    """
    Central registry for managing all alpha factors.
    
    The FactorRegistry is the main interface for:
    1. Registering factors
    2. Computing features for all stocks
    3. Applying cross-sectional normalization
    4. Managing factor metadata
    
    Usage:
        >>> # Create registry and register factors
        >>> registry = FactorRegistry()
        >>> registry.register_defaults()
        
        >>> # Compute features
        >>> features = registry.compute_all(price_data)
        
        >>> # Check registered factors
        >>> print(registry.summary())
    
    Attributes:
        _factors: Dictionary of registered factors
        _groups: Dictionary of factor groups
        _cache: Cache for computed features
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._factors: Dict[str, BaseFactor] = {}
        self._groups: Dict[str, FactorGroup] = {}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._computation_stats: Dict = {}
    
    # ==========================================
    # REGISTRATION METHODS
    # ==========================================
    
    def register(self, factor: BaseFactor) -> None:
        """
        Register a single factor.
        
        Args:
            factor: Factor instance to register
            
        Raises:
            TypeError: If factor is not a BaseFactor instance
        """
        if not isinstance(factor, BaseFactor):
            raise TypeError(f"Expected BaseFactor, got {type(factor).__name__}")
        
        name = factor.info.name
        
        if name in self._factors:
            logger.warning(f"Overwriting existing factor: {name}")
        
        self._factors[name] = factor
        logger.debug(f"Registered factor: {name}")
    
    def register_many(self, factors: List[BaseFactor]) -> None:
        """
        Register multiple factors at once.
        
        Args:
            factors: List of factor instances
        """
        for factor in factors:
            self.register(factor)
        
        logger.debug(f"Registered {len(factors)} factors")
    
    def register_defaults(
        self,
        include_momentum: bool = True,
        include_mean_reversion: bool = True,
        include_microstructure: bool = True
    ) -> None:
        """
        Register all default factors from each category.
        
        Args:
            include_momentum: Include momentum factors
            include_mean_reversion: Include mean reversion factors
            include_microstructure: Include microstructure factors
        """
        logger.info("Registering default factors...")
        
        initial_count = len(self._factors)
        
        if include_momentum:
            momentum_factors = get_momentum_factors()
            self.register_many(momentum_factors)
            logger.info(f"  Momentum: {len(momentum_factors)} factors")
        
        if include_mean_reversion:
            mr_factors = get_mean_reversion_factors()
            self.register_many(mr_factors)
            logger.info(f"  Mean Reversion: {len(mr_factors)} factors")
        
        if include_microstructure:
            micro_factors = get_microstructure_factors()
            self.register_many(micro_factors)
            logger.info(f"  Microstructure: {len(micro_factors)} factors")
        
        total_added = len(self._factors) - initial_count
        logger.info(f"Registered {total_added} total factors (total: {len(self._factors)})")
    
    def register_custom(
        self, 
        name: str, 
        compute_func: Callable[[pd.DataFrame], pd.Series],
        category: FactorCategory = FactorCategory.CUSTOM,
        lookback: int = 1,
        description: str = ""
    ) -> None:
        """
        Register a custom factor using a function.
        
        Convenience method for adding simple factors without creating a class.
        
        Args:
            name: Factor name
            compute_func: Function that takes DataFrame, returns Series
            category: Factor category
            lookback: Lookback period
            description: Factor description
            
        Example:
            >>> registry.register_custom(
            ...     name='my_factor',
            ...     compute_func=lambda df: df['close'].pct_change(5),
            ...     lookback=5,
            ...     description='5-day return'
            ... )
        """
        # Create a dynamic factor class
        class CustomFactor(BaseFactor):
            def __init__(self, n, f, cat, lb, desc):
                self._name = n
                self._func = f
                self._category = cat
                self._lookback = lb
                self._description = desc
            
            @property
            def info(self) -> FactorInfo:
                return FactorInfo(
                    name=self._name,
                    category=self._category,
                    description=self._description or f"Custom factor: {self._name}",
                    lookback=self._lookback
                )
            
            def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
                return self._func(df)
        
        factor = CustomFactor(name, compute_func, category, lookback, description)
        self.register(factor)
    
    def unregister(self, name: str) -> bool:
        """
        Remove a factor from registry.
        
        Args:
            name: Factor name to remove
            
        Returns:
            True if factor was removed, False if not found
        """
        if name in self._factors:
            del self._factors[name]
            logger.debug(f"Unregistered factor: {name}")
            return True
        
        logger.warning(f"Factor not found: {name}")
        return False
    
    def clear(self) -> None:
        """Remove all registered factors."""
        self._factors.clear()
        self._groups.clear()
        self._cache.clear()
        logger.info("Registry cleared")
    
    # ==========================================
    # RETRIEVAL METHODS
    # ==========================================
    
    def get(self, name: str) -> Optional[BaseFactor]:
        """
        Get a factor by name.
        
        Args:
            name: Factor name
            
        Returns:
            Factor instance or None if not found
        """
        return self._factors.get(name)
    
    def __getitem__(self, name: str) -> BaseFactor:
        """
        Get factor by name using index notation.
        
        Args:
            name: Factor name
            
        Returns:
            Factor instance
            
        Raises:
            KeyError: If factor not found
        """
        if name not in self._factors:
            raise KeyError(f"Factor not found: {name}")
        return self._factors[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if factor is registered."""
        return name in self._factors
    
    def list_factors(self) -> List[str]:
        """
        Get list of all registered factor names.
        
        Returns:
            Sorted list of factor names
        """
        return sorted(self._factors.keys())
    
    def list_by_category(self, category: FactorCategory) -> List[str]:
        """
        Get factors of a specific category.
        
        Args:
            category: FactorCategory enum value
            
        Returns:
            List of factor names in that category
        """
        return sorted([
            name for name, factor in self._factors.items()
            if factor.info.category == category
        ])
    
    def get_category_counts(self) -> Dict[str, int]:
        """
        Get count of factors in each category.
        
        Returns:
            Dictionary of {category_name: count}
        """
        counts = {}
        for factor in self._factors.values():
            cat = factor.info.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items()))
    
    # ==========================================
    # COMPUTATION METHODS
    # ==========================================
    
    def compute_single_stock(
        self,
        df: pd.DataFrame,
        ticker: str = None,
        factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute factors for a single stock.
        
        Args:
            df: OHLCV DataFrame for one stock (must be sorted by date)
            ticker: Ticker symbol (for logging)
            factors: Specific factors to compute (None = all)
            
        Returns:
            DataFrame with all factor values (same index as input)
        """
        results = {}
        errors = []
        
        # Determine which factors to compute
        factors_to_compute = factors or list(self._factors.keys())
        
        for name in factors_to_compute:
            if name not in self._factors:
                logger.warning(f"Factor not found: {name}")
                continue
                
            factor = self._factors[name]
            
            try:
                results[name] = factor.compute(df)
            except Exception as e:
                errors.append((name, str(e)))
                logger.debug(f"Error computing {name} for {ticker}: {e}")
                results[name] = pd.Series(np.nan, index=df.index)
        
        if errors and ticker:
            logger.warning(f"{ticker}: {len(errors)} factor computation errors")
        
        result_df = pd.DataFrame(results, index=df.index)
        
        return result_df
    
    def compute_all(
        self,
        panel_data: pd.DataFrame,
        ticker_column: str = 'ticker',
        date_column: str = 'date',
        normalize: Optional[bool] = None,
        normalization_method: str = 'rank',
        winsorize: bool = True,
        winsorize_limits: Optional[Tuple[float, float]] = None,
        add_target: bool = True,
        target_horizon: Optional[int] = None,
        parallel: bool = False,
        n_workers: int = 4,
        factors: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compute all factors for all stocks in panel data.
        
        This is the main method for feature engineering. It:
        1. Computes all registered factors for each stock
        2. Optionally winsorizes extreme values
        3. Optionally normalizes cross-sectionally
        4. Optionally adds target variable
        
        Args:
            panel_data: DataFrame with columns [date, ticker, OHLCV]
            ticker_column: Name of ticker column
            date_column: Name of date column
            normalize: Whether to normalize cross-sectionally (default: from config)
            normalization_method: 'rank', 'zscore', or 'minmax'
            winsorize: Whether to winsorize extreme values
            winsorize_limits: Quantile limits for winsorization (default: from config)
            add_target: Whether to add forward return target
            target_horizon: Forward return horizon (default: from config)
            parallel: Whether to use parallel processing
            n_workers: Number of parallel workers
            factors: Specific factors to compute (None = all)
            verbose: Whether to print progress
            
        Returns:
            DataFrame with original data + all factor columns + target
            
        Example:
            >>> registry = FactorRegistry()
            >>> registry.register_defaults()
            >>> features = registry.compute_all(price_data)
        """
        start_time = datetime.now()
        
        # Get defaults from config
        if normalize is None:
            normalize = settings.features.normalize_cross_section
        if winsorize_limits is None:
            winsorize_limits = settings.features.winsorize_limits
        if target_horizon is None:
            target_horizon = settings.features.forward_return_days
        
        n_tickers = panel_data[ticker_column].nunique()
        n_factors = len(factors) if factors else len(self._factors)
        
        if verbose:
            logger.info(f"Computing {n_factors} factors for {n_tickers} stocks...")
        
        # Validate input
        self._validate_panel_data(panel_data, ticker_column, date_column)
        
        # Get tickers
        tickers = panel_data[ticker_column].unique()
        
        # Compute factors for each stock
        if parallel and len(tickers) > 10:
            all_results = self._compute_parallel(
                panel_data, ticker_column, date_column, 
                tickers, n_workers, factors
            )
        else:
            all_results = self._compute_sequential(
                panel_data, ticker_column, date_column,
                tickers, factors, verbose
            )
        
        # Combine all stocks
        result = pd.concat(all_results, ignore_index=True)
        result = result.sort_values([date_column, ticker_column]).reset_index(drop=True)
        
        # Get feature columns
        feature_cols = self._get_feature_columns(result, date_column, ticker_column)
        
        # Winsorize extreme values
        if winsorize and feature_cols:
            if verbose:
                logger.info(f"Winsorizing {len(feature_cols)} features at {winsorize_limits}")
            result = winsorize_by_group(
                result, feature_cols, date_column, winsorize_limits
            )
        
        # Normalize cross-sectionally
        if normalize and feature_cols:
            if verbose:
                logger.info(f"Normalizing features using {normalization_method}")
            result = normalize_cross_section(
                result, 
                method=normalization_method, 
                date_column=date_column,
                columns=feature_cols
            )
        
        # Add target variable
        if add_target:
            if verbose:
                logger.info(f"Adding {target_horizon}-day forward return target")
            result = create_target_variable(
                result,
                forward_days=target_horizon,
                ticker_column=ticker_column
            )
        
        # Record statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        self._computation_stats = {
            'n_tickers': n_tickers,
            'n_factors': n_factors,
            'n_rows': len(result),
            'elapsed_seconds': elapsed,
            'normalize': normalize,
            'winsorize': winsorize,
            'target_horizon': target_horizon if add_target else None
        }
        
        if verbose:
            logger.info(f"Feature computation complete: {result.shape} in {elapsed:.1f}s")
        
        return result
    
    def _validate_panel_data(
        self,
        df: pd.DataFrame,
        ticker_column: str,
        date_column: str
    ) -> None:
        """Validate panel data structure."""
        required_cols = {ticker_column, date_column, 'close'}
        missing = required_cols - set(df.columns)
        
        if missing:
            raise FactorValidationError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        
        if len(df) == 0:
            raise FactorValidationError("Empty DataFrame provided")
    
    def _get_feature_columns(
        self,
        df: pd.DataFrame,
        date_column: str,
        ticker_column: str
    ) -> List[str]:
        """Get list of feature columns (excluding metadata and target)."""
        non_feature_cols = {
            date_column, ticker_column, 'symbol',
            'forward_return', 'return', 'log_return',
            'close', 'open', 'high', 'low', 'volume',
            'adj_close', 'adj_volume', 'dividends', 'splits'
        }
        return [c for c in df.columns if c not in non_feature_cols]
    
    def _compute_sequential(
        self,
        panel_data: pd.DataFrame,
        ticker_column: str,
        date_column: str,
        tickers: np.ndarray,
        factors: Optional[List[str]],
        verbose: bool
    ) -> List[pd.DataFrame]:
        """Compute factors sequentially for all stocks."""
        all_results = []
        
        for i, ticker in enumerate(tickers):
            if verbose and (i + 1) % 10 == 0:
                logger.debug(f"Processing {i + 1}/{len(tickers)}: {ticker}")
            
            # Get stock data
            stock_data = panel_data[panel_data[ticker_column] == ticker].copy()
            stock_data = stock_data.sort_values(date_column)
            
            # Compute factors
            factor_values = self.compute_single_stock(stock_data, ticker, factors)
            
            # Combine with original data
            stock_result = stock_data.copy()
            for col in factor_values.columns:
                stock_result[col] = factor_values[col].values
            
            all_results.append(stock_result)
        
        return all_results
    
    def _compute_parallel(
        self,
        panel_data: pd.DataFrame,
        ticker_column: str,
        date_column: str,
        tickers: np.ndarray,
        n_workers: int,
        factors: Optional[List[str]]
    ) -> List[pd.DataFrame]:
        """Compute factors in parallel using ThreadPoolExecutor."""
        logger.info(f"Using {n_workers} workers for parallel computation")
        
        results = []
        errors = []
        
        def process_ticker(ticker):
            stock_data = panel_data[panel_data[ticker_column] == ticker].copy()
            stock_data = stock_data.sort_values(date_column)
            
            factor_values = self.compute_single_stock(stock_data, ticker, factors)
            
            stock_result = stock_data.copy()
            for col in factor_values.columns:
                stock_result[col] = factor_values[col].values
            
            return stock_result
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_ticker, t): t for t in tickers}
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append((ticker, str(e)))
                    logger.error(f"Error processing {ticker}: {e}")
        
        if errors:
            logger.warning(f"Parallel computation had {len(errors)} errors")
        
        return results
    
    # ==========================================
    # ANALYSIS METHODS
    # ==========================================
    
    def get_max_lookback(self) -> int:
        """
        Get maximum lookback period across all factors.
        
        This is important for determining how much history is needed.
        
        Returns:
            Maximum lookback in periods
        """
        if not self._factors:
            return 0
        return max(f.info.lookback for f in self._factors.values())
    
    def get_min_lookback(self) -> int:
        """Get minimum lookback period across all factors."""
        if not self._factors:
            return 0
        return min(f.info.lookback for f in self._factors.values())
    
    def summary(self) -> pd.DataFrame:
        """
        Get summary of all registered factors.
        
        Returns:
            DataFrame with factor information
        """
        if not self._factors:
            return pd.DataFrame()
        
        records = []
        for name, factor in self._factors.items():
            info = factor.info
            records.append({
                'name': info.name,
                'category': info.category.value,
                'description': info.description,
                'lookback': info.lookback,
                'is_rank': info.is_rank,
                'higher_is_better': info.higher_is_better
            })
        
        df = pd.DataFrame(records)
        return df.sort_values(['category', 'name']).reset_index(drop=True)
    
    def print_summary(self) -> None:
        """Print formatted summary of registered factors."""
        print("\n" + "=" * 70)
        print("📊 FACTOR REGISTRY SUMMARY")
        print("=" * 70)
        
        counts = self.get_category_counts()
        print(f"\nTotal Factors: {len(self._factors)}")
        print(f"Max Lookback: {self.get_max_lookback()} periods")
        
        print("\nBy Category:")
        for cat, count in counts.items():
            print(f"  {cat}: {count}")
        
        print("\nFactors:")
        for category in FactorCategory:
            factors = self.list_by_category(category)
            if factors:
                print(f"\n  {category.value.upper()}:")
                for f in factors[:10]:  # Show first 10
                    factor = self._factors[f]
                    print(f"    - {f} (lookback={factor.info.lookback})")
                if len(factors) > 10:
                    print(f"    ... and {len(factors) - 10} more")
        
        print("=" * 70)
    
    def get_computation_stats(self) -> Dict:
        """Get statistics from last compute_all() call."""
        return self._computation_stats.copy()
    
    # ==========================================
    # PERSISTENCE METHODS
    # ==========================================
    
    def save_features(
        self,
        features: pd.DataFrame,
        path: Union[str, Path],
        format: str = 'pickle'
    ) -> None:
        """
        Save computed features to disk.
        
        Args:
            features: DataFrame with features
            path: Output path
            format: 'pickle', 'parquet', or 'csv'
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'pickle':
            features.to_pickle(path)
        elif format == 'parquet':
            features.to_parquet(path)
        elif format == 'csv':
            features.to_csv(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved features to {path}")
    
    def load_features(
        self,
        path: Union[str, Path]
    ) -> pd.DataFrame:
        """
        Load features from disk.
        
        Args:
            path: Input path
            
        Returns:
            DataFrame with features
        """
        path = Path(path)
        
        if path.suffix == '.pkl':
            return pd.read_pickle(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.csv':
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
    
    # ==========================================
    # CACHE METHODS
    # ==========================================
    
    def cache_features(
        self,
        key: str,
        features: pd.DataFrame
    ) -> None:
        """Cache computed features in memory."""
        self._cache[key] = features
        logger.debug(f"Cached features: {key}")
    
    def get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached features."""
        return self._cache.get(key)
    
    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()
        logger.debug("Factor cache cleared")
    
    # ==========================================
    # MAGIC METHODS
    # ==========================================
    
    def __len__(self) -> int:
        """Return number of registered factors."""
        return len(self._factors)
    
    def __iter__(self):
        """Iterate over factor names."""
        return iter(self._factors)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FactorRegistry(n_factors={len(self._factors)})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        counts = self.get_category_counts()
        cats = ", ".join(f"{k}={v}" for k, v in counts.items())
        return f"FactorRegistry with {len(self._factors)} factors: {cats}"


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compute_all_features(
    panel_data: pd.DataFrame,
    normalize: Optional[bool] = None,
    winsorize: bool = True,
    add_target: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to compute all default features.
    
    Creates a registry, registers all default factors, and computes features.
    
    Args:
        panel_data: DataFrame with [date, ticker, OHLCV]
        normalize: Whether to normalize cross-sectionally (default: from config)
        winsorize: Whether to winsorize extreme values
        add_target: Whether to add forward return target
        verbose: Whether to print progress
        
    Returns:
        DataFrame with all features added
        
    Example:
        >>> from quant_alpha.features import compute_all_features
        >>> features = compute_all_features(price_data)
    """
    registry = FactorRegistry()
    registry.register_defaults()
    
    return registry.compute_all(
        panel_data,
        normalize=normalize,
        winsorize=winsorize,
        add_target=add_target,
        verbose=verbose
    )


def get_default_registry() -> FactorRegistry:
    """
    Get a registry with all default factors registered.
    
    Returns:
        FactorRegistry with default factors
    """
    registry = FactorRegistry()
    registry.register_defaults()
    return registry


# ============================================
# MODULE TEST
# ============================================

def test_registry():
    """Test factor registry with sample data."""
    print("\n" + "=" * 70)
    print("🧪 TESTING FACTOR REGISTRY")
    print("=" * 70)
    
    # Create sample panel data
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    records = []
    for ticker in tickers:
        # Generate realistic price series
        returns = np.random.randn(len(dates)) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            records.append({
                'date': date,
                'ticker': ticker,
                'open': prices[i] * (1 + np.random.randn() * 0.005),
                'high': prices[i] * (1 + abs(np.random.randn()) * 0.01),
                'low': prices[i] * (1 - abs(np.random.randn()) * 0.01),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000)
            })
    
    panel_data = pd.DataFrame(records)
    
    # Fix high/low
    panel_data['high'] = panel_data[['open', 'high', 'close']].max(axis=1) * 1.001
    panel_data['low'] = panel_data[['open', 'low', 'close']].min(axis=1) * 0.999
    
    print(f"\n📊 Sample data: {len(panel_data)} rows, {len(tickers)} stocks")
    
    # Test registry
    registry = FactorRegistry()
    
    print("\n1️⃣ Testing factor registration...")
    registry.register_defaults()
    print(f"   ✅ Registered {len(registry)} factors")
    
    print("\n2️⃣ Testing factor retrieval...")
    print(f"   Categories: {registry.get_category_counts()}")
    print(f"   Max lookback: {registry.get_max_lookback()}")
    
    print("\n3️⃣ Testing feature computation...")
    features = registry.compute_all(
        panel_data,
        normalize=True,
        winsorize=True,
        add_target=True,
        verbose=False
    )
    print(f"   ✅ Computed features: {features.shape}")
    
    # Check for NaN
    feature_cols = [c for c in features.columns if c not in 
                   ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    nan_pct = features[feature_cols].isna().mean().mean() * 100
    print(f"   NaN percentage: {nan_pct:.1f}%")
    
    print("\n4️⃣ Testing cross-sectional normalization...")
    # Check that ranks are between 0 and 1
    sample_feature = feature_cols[0]
    min_val = features[sample_feature].min()
    max_val = features[sample_feature].max()
    print(f"   {sample_feature}: range [{min_val:.3f}, {max_val:.3f}]")
    
    print("\n5️⃣ Testing summary...")
    summary = registry.summary()
    print(f"   Summary shape: {summary.shape}")
    print(summary.head())
    
    print("\n6️⃣ Testing computation stats...")
    stats = registry.get_computation_stats()
    print(f"   Stats: {stats}")
    
    print("\n✅ All registry tests passed!")
    
    return registry, features


if __name__ == "__main__":
    test_registry()