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
            
            # Create Result DataFrame
            result = data.copy()
            
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
            if self.winsorize_flag:
                result[self.name] = self._winsorize(result[self.name])
            
            if self.normalize:
                result = self._cross_sectional_normalize(result)
            
            if self.fill_na:
                result[self.name] = result[self.name].fillna(0)
            
            # Step 4: Validate output
            self._validate_output(result)
            
            # Update statistics
            self.computation_time = time.time() - start_time
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
    
    # ==================== TRANSFORMATIONS ====================
    
    def _winsorize(self, series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        """Clip extreme outliers (1st and 99th percentile)"""
        lower_val = series.quantile(lower)
        upper_val = series.quantile(upper)
        return series.clip(lower=lower_val, upper=upper_val)
    
    def _cross_sectional_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Robust Z-Score Normalization per DATE.
        Handles edge cases:
        - Less than 3 stocks on a given date (return 0)
        - Zero Variance (std=0) (return 0)
        - NaN handling
        """
        
        def robust_zscore(x):
            # 1. Safety Check: Need at least 3 stocks to calculate meaningful stats
            if len(x.dropna()) < 3:
                return 0.0 
            
            std = x.std()
            
            # 2. Safety Check: Zero Variance or NaN Std
            if std == 0 or pd.isna(std):
                return 0.0
                
            # 3. Standard Z-Score
            return (x - x.mean()) / std

        # Apply robust transformation
        # transform() keeps the index aligned, which is crucial
        data[self.name] = data.groupby('date')[self.name].transform(robust_zscore)
        
        # Final cleanup ensures no NaNs slip through from the transform
        data[self.name] = data[self.name].fillna(0.0)
        
        return data

# ==================== SPECIALIZED SUBCLASSES ====================

class TechnicalFactor(BaseFactor):
    def __init__(self, name: str, description: str, lookback_period: int, **kwargs):
        super().__init__(name, 'technical', description, lookback_period, **kwargs)

class FundamentalFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'fundamental', description, lookback_period=None, **kwargs)

class EarningsFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'earnings', description, lookback_period=None, **kwargs)

class AlternativeFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name, 'alternative', description, lookback_period=None, **kwargs)