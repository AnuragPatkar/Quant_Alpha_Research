"""
Base Factor Classes
===================
Abstract base class and utilities for all alpha factors.

This module provides:
- FactorCategory enum for categorizing factors
- FactorInfo dataclass for factor metadata
- BaseFactor abstract class that all factors must inherit
- Validation utilities

Author: [Your Name]
Last Updated: 2024
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class FactorValidationError(Exception):
    """Custom exception for factor validation failures."""
    pass


class FactorCategory(Enum):
    """
    Factor categories for organization and analysis.
    
    Categories:
        MOMENTUM: Trend-following factors
        MEAN_REVERSION: Counter-trend factors
        VOLATILITY: Risk/volatility-based factors
        VOLUME: Volume and liquidity factors
        MICROSTRUCTURE: Market microstructure factors
        FUNDAMENTAL: Fundamental/value factors (future)
        SENTIMENT: Sentiment-based factors (future)
    """
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    MICROSTRUCTURE = "microstructure"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"


@dataclass
class FactorInfo:
    """
    Factor metadata container.
    
    Attributes:
        name: Unique factor identifier (lowercase, underscores only)
        category: Factor category from FactorCategory enum
        description: Human-readable description
        lookback: Number of periods needed for calculation
        is_rank: Whether this is a cross-sectional rank factor
        higher_is_better: Whether higher values indicate stronger signal
    """
    name: str
    category: FactorCategory
    description: str
    lookback: int
    is_rank: bool = False
    higher_is_better: bool = True
    
    def __post_init__(self):
        """Validate factor info after initialization."""
        self._validate_name()
        self._validate_lookback()
    
    def _validate_name(self) -> None:
        """Ensure name follows conventions."""
        # Only lowercase, numbers, underscores allowed
        pattern = r'^[a-z][a-z0-9_]*$'
        if not re.match(pattern, self.name):
            raise FactorValidationError(
                f"Invalid factor name: '{self.name}'. "
                f"Must be lowercase, start with letter, use only a-z, 0-9, underscore."
            )
    
    def _validate_lookback(self) -> None:
        """Ensure lookback is valid."""
        if self.lookback < 1:
            raise FactorValidationError(
                f"Invalid lookback: {self.lookback}. Must be >= 1."
            )


class BaseFactor(ABC):
    """
    Abstract base class for all alpha factors.
    
    All factor implementations must:
    1. Inherit from BaseFactor
    2. Implement the `info` property returning FactorInfo
    3. Implement the `compute()` method
    
    The base class provides:
    - Input validation
    - Error handling
    - Logging
    - Common utilities
    
    Example:
        >>> class MyFactor(BaseFactor):
        ...     def __init__(self, window: int = 20):
        ...         self.window = window
        ...     
        ...     @property
        ...     def info(self) -> FactorInfo:
        ...         return FactorInfo(
        ...             name=f"my_factor_{self.window}",
        ...             category=FactorCategory.CUSTOM,
        ...             description=f"My custom {self.window}-day factor",
        ...             lookback=self.window
        ...         )
        ...     
        ...     def compute(self, df: pd.DataFrame) -> pd.Series:
        ...         return df['close'].rolling(self.window).mean()
    """
    
    # Required columns for factor computation
    REQUIRED_COLUMNS = {'close'}
    
    @property
    @abstractmethod
    def info(self) -> FactorInfo:
        """
        Return factor metadata.
        
        Returns:
            FactorInfo object with name, category, description, lookback
        """
        pass
    
    @abstractmethod
    def _compute_impl(self, df: pd.DataFrame) -> pd.Series:
        """
        Internal computation implementation.
        
        Args:
            df: Validated OHLCV DataFrame for single stock
            
        Returns:
            Series of factor values (same index as input)
        """
        pass
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute factor values with validation and error handling.
        
        This is the public method that should be called.
        It wraps _compute_impl with validation.
        
        Args:
            df: OHLCV DataFrame for single stock
            
        Returns:
            Series of factor values
            
        Raises:
            FactorValidationError: If input validation fails
        """
        # Validate input
        self._validate_input(df)
        
        # Compute factor
        try:
            result = self._compute_impl(df)
        except Exception as e:
            logger.error(f"Error computing {self.info.name}: {e}")
            # Return NaN series on error
            return pd.Series(np.nan, index=df.index, name=self.info.name)
        
        # Validate output
        result = self._validate_output(result, df)
        
        return result
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            FactorValidationError: If validation fails
        """
        # Check if DataFrame
        if not isinstance(df, pd.DataFrame):
            raise FactorValidationError(
                f"Expected DataFrame, got {type(df).__name__}"
            )
        
        # Check if empty
        if len(df) == 0:
            raise FactorValidationError("Empty DataFrame provided")
        
        # Check required columns
        df_columns = set(df.columns.str.lower())
        missing = self.REQUIRED_COLUMNS - df_columns
        if missing:
            raise FactorValidationError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        
        # Warn if insufficient data for lookback
        if len(df) < self.info.lookback:
            logger.warning(
                f"{self.info.name}: DataFrame has {len(df)} rows, "
                f"but lookback is {self.info.lookback}. "
                f"Result will have many NaN values."
            )
    
    def _validate_output(
        self, 
        result: pd.Series, 
        df: pd.DataFrame
    ) -> pd.Series:
        """
        Validate and clean output Series.
        
        Args:
            result: Computed factor values
            df: Original input DataFrame
            
        Returns:
            Validated and cleaned Series
        """
        # Ensure Series
        if not isinstance(result, pd.Series):
            result = pd.Series(result, index=df.index)
        
        # Set name
        result.name = self.info.name
        
        # Ensure same index as input
        if len(result) != len(df):
            logger.warning(
                f"{self.info.name}: Output length ({len(result)}) != "
                f"input length ({len(df)}). Reindexing."
            )
            result = result.reindex(df.index)
        
        # Replace infinities with NaN
        inf_count = np.isinf(result).sum()
        if inf_count > 0:
            logger.warning(f"{self.info.name}: Replacing {inf_count} inf values with NaN")
            result = result.replace([np.inf, -np.inf], np.nan)
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.info.name}', lookback={self.info.lookback})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return f"{self.info.name}: {self.info.description}"


class FactorGroup:
    """
    Group of related factors for batch computation.
    
    Useful for organizing factors by category or strategy.
    """
    
    def __init__(self, name: str, factors: List[BaseFactor] = None):
        """
        Initialize factor group.
        
        Args:
            name: Group name
            factors: List of factors (optional)
        """
        self.name = name
        self.factors: List[BaseFactor] = factors or []
    
    def add(self, factor: BaseFactor) -> None:
        """Add a factor to the group."""
        self.factors.append(factor)
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all factors in the group.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with all factor values
        """
        results = {}
        for factor in self.factors:
            try:
                results[factor.info.name] = factor.compute(df)
            except Exception as e:
                logger.error(f"Error in {factor.info.name}: {e}")
                results[factor.info.name] = pd.Series(np.nan, index=df.index)
        
        return pd.DataFrame(results)
    
    def get_names(self) -> List[str]:
        """Get list of factor names."""
        return [f.info.name for f in self.factors]
    
    def __len__(self) -> int:
        return len(self.factors)
    
    def __repr__(self) -> str:
        return f"FactorGroup('{self.name}', n_factors={len(self.factors)})"


# Utility functions for factor computation
def safe_divide(
    numerator: Union[pd.Series, np.ndarray, float, int],
    denominator: Union[pd.Series, np.ndarray, float, int],
    fill_value: float = 0.0
) -> Union[pd.Series, np.ndarray]:
    """
    Safe division handling zero and near-zero denominators.
    
    Args:
        numerator: Numerator values (can be scalar, array, or Series)
        denominator: Denominator values (can be scalar, array, or Series)
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result of safe division (same type as input Series/array)
    """
    # Create mask for valid denominators
    is_valid = np.abs(denominator) > 1e-8
    
    # Determine result type - prefer Series if either input is Series
    if isinstance(numerator, pd.Series):
        result = pd.Series(fill_value, index=numerator.index)
        result[is_valid] = numerator[is_valid] / denominator[is_valid]
    elif isinstance(denominator, pd.Series):
        # Numerator is scalar/array but denominator is Series
        result = pd.Series(fill_value, index=denominator.index)
        result[is_valid] = numerator / denominator[is_valid]
    else:
        # Both are arrays or scalars
        result = np.where(is_valid, numerator / denominator, fill_value)
    
    return result


def clip_values(
    series: Union[pd.Series, np.ndarray],
    lower: Optional[float] = None,
    upper: Optional[float] = None
) -> Union[pd.Series, np.ndarray]:
    """
    Clip values to specified range.
    Works with both pandas Series and numpy arrays.
    
    Args:
        series: Input series or array
        lower: Lower bound (optional)
        upper: Upper bound (optional)
        
    Returns:
        Clipped series/array
    """
    if isinstance(series, pd.Series):
        result = series.copy()
        if lower is not None:
            result = result.clip(lower=lower)
        if upper is not None:
            result = result.clip(upper=upper)
    else:
        # numpy array - use np.clip with a_min/a_max
        result = np.clip(series, a_min=lower, a_max=upper)
    
    return result