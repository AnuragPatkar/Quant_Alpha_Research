"""
Base Factor Classes
All factors inherit from these abstract classes
Design Pattern: Template Method + Strategy
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from config.logging_config import logger
from .utils import winsorize, cross_sectional_normalize

EPS = 1e-9  # Centralized small epsilon to prevent division by zero

class BaseFactor(ABC):
    """
    Abstract base class for all factors with built-in:
    - Validation
    - Winsorization (Outlier Clipping)
    - Normalization (Z-Score)
    - Metadata Logging
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
        
        # logger.debug(f"Initialized factor: {self.name}")
    
    # ==================== PUBLIC API (Template Method) ====================
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Orchestrates the calculation flow.
        1. Validate -> 2. Compute -> 3. Clean -> 4. Normalize
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Validate input
            self._validate_input(data)
            
            # Step 2: Compute raw factor
            # logger.debug(f"Computing factor: {self.name}")
            factor_values = self.compute(data)
            
            # Create Result DataFrame (Optimized)
            # Instead of copying the whole dataframe (data.copy()), we only keep 
            # metadata columns needed for alignment.
            meta_cols = [c for c in ['date', 'ticker'] if c in data.columns]
            result = data[meta_cols].copy()
            
            # Handle if compute returns Series or DataFrame
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
            # 3.0: Sanitize (Inf -> NaN) BEFORE processing to prevent corruption
            if np.isinf(result[self.name]).any():
                 # Optimization: Boolean indexing is faster than replace()
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
        CORE LOGIC HERE. Subclasses must implement this.
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
        
        # Check for infinite values
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
        Override validation for fundamental factors.
        Fundamental data is static (no date/ticker index required).
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")

class EarningsFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'earnings', description, lookback_period=None, **kwargs)

class AlternativeFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        # FIX: Single correct initialization with macro-specific defaults
        # Macro data usually doesn't need cross-sectional normalization
        kwargs.setdefault('normalize', False) 
        super().__init__(name, 'alternative', description, lookback_period=None, **kwargs)
    
    def _validate_input(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")

class CompositeFactor(BaseFactor):
    """Composite factors blend multiple signal types for regime-aware trading"""
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'composite', description, lookback_period=None, **kwargs)
    
    def _validate_input(self, data: pd.DataFrame):
        """
        Override validation for composite factors.
        Composite data may mix macro-level (no ticker) with ticker-level data.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")