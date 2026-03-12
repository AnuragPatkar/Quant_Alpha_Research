"""
Base Factor Architecture
========================
Abstract base classes defining the contract and lifecycle for all alpha factors.

Purpose
-------
This module establishes the **Template Method** design pattern for the feature engineering
pipeline. It provides a robust, standardized execution flow for factor computation, ensuring
that every signal undergoes consistent validation, outlier mitigation (winsorization),
and distribution scaling (cross-sectional normalization) before entering the model.

Usage
-----
Developers should inherit from specific category bases (e.g., `TechnicalFactor`,
`FundamentalFactor`) rather than `BaseFactor` directly.

.. code-block:: python

    class MyMomentum(TechnicalFactor):
        def compute(self, data: pd.DataFrame) -> pd.Series:
            return data['close'].pct_change(10)

Importance
----------
- **Signal Hygiene**: Enforces statistical rigor by automatically handling `NaN` propagation,
  infinite values, and outliers ($3\sigma$ clipping) centrally.
- **Pipeline Uniformity**: Guarantees that all 100+ factors output aligned, standardized
  Pandas Series/DataFrames, facilitating vectorized operations downstream.
- **Complexity Abstraction**: Hides the complexity of cross-sectional alignment and
  error logging from individual factor logic.

Tools & Frameworks
------------------
- **ABC**: Enforces the implementation of the `compute` method in subclasses.
- **Pandas**: Core data structure for time-series and cross-sectional data.
- **NumPy**: Efficient handling of numerical singularities ($\epsilon$).
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from config.logging_config import logger
from .utils import winsorize, cross_sectional_normalize

EPS = 1e-9  # Machine epsilon for numerical stability in division operations

class BaseFactor(ABC):
    """
    Abstract base class implementing the Factor Lifecycle.
    
    This class dictates the skeleton of the algorithm (Template Method), delegating
    the specific `compute` logic to subclasses.
    
    Capabilities:
    - **Input Validation**: Ensures required columns (e.g., 'date', 'ticker') exist.
    - **Statistical Processing**: Optional Winsorization and Z-Score Normalization.
    - **Sanitization**: Automatic handling of `inf` and `NaN` values.
    - **Metadata**: Tracks computation time and versioning.
    """
    
    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        lookback_period: Optional[int] = None,
        normalize: bool = True,
        winsorize: bool = True,
        fill_na: bool = True
    ):
        self.name = name
        self.category = category
        self.description = description
        self.lookback_period = lookback_period
        self.normalize = normalize
        self.winsorize_flag = winsorize
        self.fill_na = fill_na
        
        # Metadata
        self.created_at = datetime.now()
        self.version = '1.0'
        
        # Statistics
        self.computation_time = 0.0
        self.last_computed = None
        self.num_computations = 0
    
    # ==================== PUBLIC API (Template Method) ====================
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the factor computation pipeline.
        
        Pipeline Steps:
        1. **Validate**: Check input schema integrity.
        2. **Compute**: Execute subclass-specific logic via `compute()`.
        3. **Align**: Ensure output dimensions match input metadata.
        4. **Sanitize**: Clean infinite values and apply statistical transforms.
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Validate input
            self._validate_input(data)
            
            # Step 2: Compute raw factor
            factor_values = self.compute(data)
            
            # Memory Optimization:
            # Instead of deep-copying the entire input DataFrame, we construct a 
            # lightweight result container with only necessary metadata.
            meta_cols = [c for c in ['date', 'ticker'] if c in data.columns]
            result = data[meta_cols].copy()
            
            # Type Handling: Support both pd.Series and single-column pd.DataFrame outputs
            if isinstance(factor_values, pd.Series):
                result[self.name] = factor_values.values
            elif isinstance(factor_values, pd.DataFrame):
                # If it returns a DF, merge or assign
                if self.name in factor_values.columns:
                    result[self.name] = factor_values[self.name].values
                else:
                    # Fallback: assume single column DF
                    result[self.name] = factor_values.iloc[:, 0].values
            else:
                # Numpy array
                result[self.name] = factor_values
            
            # Step 3: Apply transformations
            # 3.0: Sanitize (Inf -> NaN) BEFORE statistical processing to prevent mean corruption
            if np.isinf(result[self.name]).any():
                 # Optimization: Vectorized boolean indexing is significantly faster than .replace()
                 result.loc[np.isinf(result[self.name]), self.name] = np.nan

            if self.winsorize_flag:
                result = winsorize(result, [self.name])
            
            if self.normalize:
                result = cross_sectional_normalize(result, [self.name])
            
            if self.fill_na:
                result[self.name] = result[self.name].fillna(0)
            
            # Step 4: Validate output
            self._validate_output(result)
            
            # Update statistics
            self.computation_time = time.perf_counter() - start_time
            self.last_computed = datetime.now()
            self.num_computations += 1
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error computing {self.name}: {str(e)}")
            raise
    
    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Abstract method for factor logic. Subclasses must implement the specific signal generation.
        """
        pass
    
    # ==================== VALIDATION ====================
    
    def _validate_input(self, data: pd.DataFrame):
        required_cols = ['date', 'ticker']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Factor {self.name} requires columns: {missing_cols}")
        
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")
    
    def _validate_output(self, result: pd.DataFrame):
        if self.name not in result.columns:
            raise ValueError(f"Factor {self.name} failed to generate output column")
        
        # Post-computation sanity check
        if np.isinf(result[self.name]).any():
            logger.warning(f"⚠️ Factor {self.name} contains infinite values. Replacing with NaN.")
            result[self.name] = result[self.name].replace([np.inf, -np.inf], np.nan)

# ==================== SPECIALIZED SUBCLASSES ====================

class TechnicalFactor(BaseFactor):
    def __init__(self, name: str, description: str, lookback_period: int, **kwargs):
        super().__init__(name, 'technical', description, lookback_period, **kwargs)

class FundamentalFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'fundamental', description, lookback_period=None, normalize=False, winsorize=False, fill_na=False, **kwargs)
    
    def _validate_input(self, data: pd.DataFrame):
        """
        Override: Fundamental datasets often use wide formats or sparse indexing,
        relaxing the strict (date, ticker) requirement found in time-series data.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")

class EarningsFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'earnings', description, lookback_period=None, **kwargs)

class AlternativeFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        # Default Behavior: Macro/Alternative data is often absolute (e.g., VIX, Sentiment)
        # and does not require cross-sectional normalization (Z-Scoring).
        kwargs.setdefault('normalize', False) 
        super().__init__(name, 'alternative', description, lookback_period=None, **kwargs)
    
    def _validate_input(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")

class CompositeFactor(BaseFactor):
    """Composite factors blend multiple signal types for regime-aware ranking strategies."""
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'composite', description, lookback_period=None, **kwargs)
    
    def _validate_input(self, data: pd.DataFrame):
        """
        Override: Composite calculations may involve heterogeneous data sources 
        (Macro + Ticker-level), requiring flexible validation.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")