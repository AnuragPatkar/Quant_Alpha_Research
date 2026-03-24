"""
Base Factor Architecture
========================

Abstract base classes defining the contract and strict lifecycle evaluation bounds for all alpha factors.

Purpose
-------
This module establishes the mathematical **Template Method** design pattern mapping 
the feature engineering pipeline. It guarantees robust, standardized execution 
flows explicitly shielding mathematical states against out-of-sample leakage and 
index misalignment.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Index alignment manipulation and absolute NaN normalization vectors.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import time
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
    the specific continuous `compute` logic mapping directly to subclasses.
    
    Forces default scaling bypass configurations (`normalize=False`, `winsorize=False`) 
    strictly limiting full-panel distributional leakage. Downstream scalers inherently 
    process these state thresholds distinctly.
    """

    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        lookback_period: Optional[int] = None,
        normalize: bool = False,
        winsorize: bool = False,
        fill_na: bool = True,
    ):
        """
        Initializes the abstract factor boundary states.
        
        Args:
            name (str): Strict string identifier for the explicit factor parameter.
            category (str): Topological grouping category (e.g., 'technical', 'fundamental').
            description (str): Verbose documentation bounding exact mathematical extraction.
            lookback_period (Optional[int]): Sequential tracking evaluation lengths. Defaults to None.
            normalize (bool): Execution flag bounding standard cross-sectional Z-scoring. Defaults to False.
            winsorize (bool): Execution flag triggering structural distribution trims. Defaults to False.
            fill_na (bool): Limits null vectors casting natively back to 0. Defaults to True.
        """
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
        Orchestrates the rigid sequential factor computation pipeline limits.

        Pipeline Steps:
        1. Validate: Asserts input matrix strict schema integrity.
        2. Compute: Executes specialized subclass topology via compute().
        3. Align: Enforces output sequence structural matches bypassing vector collisions.
        4. Sanitize: Binds numeric boundaries eliminating infinite numerical evaluations.
        
        Args:
            data (pd.DataFrame): Systemic raw market historical dataframe.
            
        Returns:
            pd.DataFrame: Computed boundaries integrated cleanly with isolated factor limits.
        """
        # Records explicit fractional latency metrics evaluating processing optimizations
        start_time = time.perf_counter()

        try:
            self._validate_input(data)

            factor_values = self.compute(data)

            meta_cols = [c for c in ['date', 'ticker'] if c in data.columns]
            result = data[meta_cols].copy()

            # Enforces absolute dimensional alignment resolving potential index permutations 
            # inherently shifted dynamically during cross-sectional dataframe grouping operators.
            if isinstance(factor_values, pd.Series):
                result[self.name] = factor_values.reindex(data.index).values
            elif isinstance(factor_values, pd.DataFrame):
                if self.name in factor_values.columns:
                    result[self.name] = (
                        factor_values[self.name].reindex(data.index).values
                    )
                else:
                    result[self.name] = (
                        factor_values.iloc[:, 0].reindex(data.index).values
                    )
            else:
                result[self.name] = factor_values

            # Systematically captures implicit zero-division remnants scaling inf values natively down to NaN limits
            col = result[self.name]
            if np.isinf(col).any():
                result.loc[np.isinf(col), self.name] = np.nan

            if self.winsorize_flag:
                result = winsorize(result, [self.name])

            if self.normalize:
                result = cross_sectional_normalize(result, [self.name])

            if self.fill_na:
                result[self.name] = result[self.name].fillna(0)

            self._validate_output(result)

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
        Abstract topological mathematical interface bounding signal generation parameters.
        
        Args:
            data (pd.DataFrame): Foundational executing variables mapping limits.
            
        Returns:
            pd.Series: Extracted structural definitions strictly indexed symmetrically mapping limits.
        """
        pass

    # ==================== VALIDATION ====================

    def _validate_input(self, data: pd.DataFrame):
        """
        Strictly validates temporal intersection mapping execution boundaries.
        
        Args:
            data (pd.DataFrame): Systemic input map bounds.
            
        Raises:
            ValueError: If target schema permutations intrinsically omit date sequences.
        """
        required_cols = ['date', 'ticker']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Factor {self.name} requires columns: {missing_cols}")

        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")

    def _validate_output(self, result: pd.DataFrame):
        """
        Strictly verifies output dimensional arrays binding evaluated sequence lengths.
        
        Args:
            result (pd.DataFrame): Symmetrically projected factor distribution sequences.
            
        Raises:
            ValueError: If internal variable matrices failed explicit derivation rendering limits.
        """
        if self.name not in result.columns:
            raise ValueError(f"Factor {self.name} failed to generate output column")

        if np.isinf(result[self.name]).any():
            logger.warning(
                f"⚠️ Factor {self.name} contains infinite values. Replacing with NaN."
            )
            result[self.name] = result[self.name].replace([np.inf, -np.inf], np.nan)


# ==================== SPECIALIZED SUBCLASSES ====================

class TechnicalFactor(BaseFactor):
    """Base subclass strictly targeting price-derived quantitative geometric extraction parameters."""
    def __init__(self, name: str, description: str, lookback_period: int, **kwargs):
        super().__init__(name, 'technical', description, lookback_period, **kwargs)


class FundamentalFactor(BaseFactor):
    """Base subclass mapping sparse quarterly and annual accounting metrics to structured time-series."""
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name, 'fundamental', description,
            lookback_period=None,
            normalize=False,
            winsorize=False,
            fill_na=False,
            **kwargs,
        )

    def _validate_input(self, data: pd.DataFrame):
        """
        Relaxes validation strictness supporting unstructured or categorical broad data bounds explicitly.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")


class EarningsFactor(BaseFactor):
    """Base subclass explicitly mapping trailing structural consensus parameters and surprise beats."""
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(
            name, 'earnings', description,
            lookback_period=None,
            normalize=False,
            winsorize=False,
            **kwargs,
        )


class AlternativeFactor(BaseFactor):
    """Base subclass accommodating unstructured exogenous metric limits bridging non-standard datasets."""
    def __init__(self, name: str, description: str, **kwargs):
        kwargs.setdefault('normalize', False)
        kwargs.setdefault('winsorize', False)
        super().__init__(name, 'alternative', description, lookback_period=None, **kwargs)

    def _validate_input(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")


class CompositeFactor(BaseFactor):
    """Base subclass structurally integrating distinct primitive limits enforcing conditional bounds."""

    def __init__(self, name: str, description: str, **kwargs):
        kwargs.setdefault('normalize', False)
        kwargs.setdefault('winsorize', False)
        super().__init__(name, 'composite', description, lookback_period=None, **kwargs)

    def _validate_input(self, data: pd.DataFrame):
        """
        Accommodates complex heterogeneous temporal bounds bounding integrated limits efficiently.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")