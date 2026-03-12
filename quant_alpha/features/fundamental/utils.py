"""
Fundamental Feature Utilities
=============================
Abstract base classes and schema validation logic for fundamental factor generation.

Purpose
-------
This module serves as the **Data Abstraction Layer (DAL)** between raw fundamental datasets
(e.g., Compustat, FMP) and the factor engineering pipeline. It resolves schema heterogeneity
via fuzzy column matching and provides standardized design patterns for common factor types
(Single-Column and Ratio-based metrics).

Usage
-----
These utilities are primarily consumed by specific factor implementations in the `fundamental` subpackage.

.. code-block:: python

    from .utils import RatioFactor

    class MyNewRatio(RatioFactor):
        def __init__(self):
            super().__init__('my_ratio', num_key='operating_income', den_key='interest_expense')

Importance
----------
- **Schema Agnosticism**: Decouples factor logic from vendor-specific column naming conventions,
  allowing for seamless vendor switching or data updates.
- **Robustness**: Enforces safe division (using machine epsilon) and standardized error handling
  for missing data, preventing pipeline crashes in production.
- **Code DRYness**: Reduces boilerplate code for the hundreds of simple arithmetic factors
  common in fundamental analysis.

Tools & Frameworks
------------------
- **Pandas**: DataFrame schema introspection and vectorized operations.
- **NumPy**: Handling of `NaN` propagation and numerical stability ($EPS$).
"""

import numpy as np
import pandas as pd
from typing import Optional
from config.logging_config import logger
from config.mappings import COLUMN_MAPPINGS
from ..base import FundamentalFactor, EPS

class FundamentalColumnValidator:
    """
    Schema Abstraction Engine.
    
    Provides a robust mechanism to resolve column names against a dataset, prioritizing
    exact matches before falling back to configuration aliases and heuristic matching.
    """
    @classmethod
    def find_column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        """
        Locates a column in the DataFrame using a tiered lookup strategy.
        
        Priority:
        1. **Exact Match**: The key exists directly in the columns.
        2. **Alias Map**: Checks `COLUMN_MAPPINGS` for known vendor variations.
        3. **Case-Insensitive**: Checks for case variance (e.g., 'Ebitda' vs 'EBITDA').
        
        Returns:
            The actual column name in the DataFrame, or None if not found.
        """
        # 1. Exact Direct Check ($O(1)$)
        if key in df.columns: return key
        
        # 2. Config-based Alias Check
        if key in COLUMN_MAPPINGS:
            for variant in COLUMN_MAPPINGS[key]:
                if variant in df.columns: return variant
                
        # 3. Heuristic Case-Insensitive Check ($O(C)$ where C is columns)
        col_map_lower = {c.lower(): c for c in df.columns}
        if key.lower() in col_map_lower: return col_map_lower[key.lower()]
        
        return None

# ==================== SHARED FACTOR BASES ====================

class SingleColumnFactor(FundamentalFactor):
    """
    Base class for atomic factors derived from a single fundamental metric.
    
    Handles column resolution, optional sign inversion (e.g., for Risk metrics like Leverage),
    and standardized error logging.
    """
    def __init__(self, name: str, col_key: str, invert: bool = False, description: str = ""):
        super().__init__(name=name, description=description)
        self.col_key = col_key
        self.invert = invert

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = FundamentalColumnValidator.find_column(df, self.col_key)
        if col:
            val = df[col]
            # Apply inversion if defined (e.g., Low Debt -> High Score)
            return -1.0 * val if self.invert else val
        
        logger.warning(f"⚠️  {self.name}: Missing '{self.col_key}' column")
        return pd.Series(np.nan, index=df.index)

class RatioFactor(FundamentalFactor):
    """
    Base class for relative valuation and efficiency factors ($A / B$).
    
    Implements vectorized division with numerical stability checks.
    """
    def __init__(self, name: str, num_key: str, den_key: str, description: str = ""):
        super().__init__(name=name, description=description)
        self.num_key = num_key
        self.den_key = den_key

    def compute(self, df: pd.DataFrame) -> pd.Series:
        num = FundamentalColumnValidator.find_column(df, self.num_key)
        den = FundamentalColumnValidator.find_column(df, self.den_key)
        
        if num and den:
            # Vectorized Division: $$ Factor = \frac{Num}{Den + \epsilon} $$
            # Adds machine epsilon to denominator to prevent ZeroDivisionError.
            return df[num] / (df[den] + EPS)
            
        logger.warning(f"⚠️  {self.name}: Missing {self.num_key} or {self.den_key}")
        return pd.Series(np.nan, index=df.index)