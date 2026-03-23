"""
Base Factor Architecture
========================
Abstract base classes defining the contract and lifecycle for all alpha factors.

Purpose
-------
This module establishes the **Template Method** design pattern for the feature engineering
pipeline. It provides a robust, standardized execution flow for factor computation.

FIXES:
  BUG-026: Removed full-panel winsorize() and cross_sectional_normalize() from
           BaseFactor.calculate(). These create data leakage when called on the full
           panel that includes test dates. Per-fold preprocessing is handled in
           trainer.py via WinsorisationScaler (confirmed architectural decision).
           Defaults changed to normalize=False, winsorize_flag=False.

  BUG-029: Fixed mixed time.time() / time.perf_counter() — now uses perf_counter
           consistently for accurate sub-second timing.

  BUG-044: Fixed .values assignment that strips index. Now uses .reindex(data.index)
           before .values to ensure correct row alignment when groupby re-orders.
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
    the specific `compute` logic to subclasses.

    NOTE on winsorize / normalize defaults:
        Both default to False. Full-panel normalization creates data leakage when
        compute_all() is called on the entire dataset (including test dates).
        Per-fold preprocessing is handled by WinsorisationScaler in trainer.py.
        If you explicitly set normalize=True / winsorize=True on a subclass, you
        are responsible for ensuring only in-sample data is passed to compute_all().
    """

    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        lookback_period: Optional[int] = None,
        # FIX BUG-026: Default both to False to prevent full-panel leakage.
        normalize: bool = False,
        winsorize: bool = False,
        fill_na: bool = True,
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
        1. Validate: Check input schema integrity.
        2. Compute: Execute subclass-specific logic via compute().
        3. Align: Ensure output dimensions match input via reindex (not .values).
        4. Sanitize: Clean infinite values; apply optional statistical transforms.
        """
        # FIX BUG-029: Use perf_counter consistently throughout.
        start_time = time.perf_counter()

        try:
            # Step 1: Validate input
            self._validate_input(data)

            # Step 2: Compute raw factor
            factor_values = self.compute(data)

            # Lightweight result container (avoid deep-copying full input DF)
            meta_cols = [c for c in ['date', 'ticker'] if c in data.columns]
            result = data[meta_cols].copy()

            # FIX BUG-044: Use .reindex(data.index) before .values to maintain
            # correct row alignment after groupby operations that may re-order rows.
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
                # Numpy array — assume aligned to data.index (caller's responsibility)
                result[self.name] = factor_values

            # Step 3: Sanitize (Inf → NaN) BEFORE any statistical processing
            col = result[self.name]
            if np.isinf(col).any():
                result.loc[np.isinf(col), self.name] = np.nan

            # Step 4: Optional transforms (only if explicitly enabled on subclass).
            # NOTE: Both are False by default (BUG-026). Do not enable unless you
            # are certain only in-sample data is flowing through this path.
            if self.winsorize_flag:
                result = winsorize(result, [self.name])

            if self.normalize:
                result = cross_sectional_normalize(result, [self.name])

            if self.fill_na:
                result[self.name] = result[self.name].fillna(0)

            # Step 5: Validate output
            self._validate_output(result)

            # Update statistics
            self.computation_time = time.perf_counter() - start_time  # FIX BUG-029
            self.last_computed = datetime.now()
            self.num_computations += 1

            return result

        except Exception as e:
            logger.error(f"❌ Error computing {self.name}: {str(e)}")
            raise

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Abstract method for factor logic. Subclasses implement the signal generation.
        Must return a pd.Series with the same index as data.index.
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

        if np.isinf(result[self.name]).any():
            logger.warning(
                f"⚠️ Factor {self.name} contains infinite values. Replacing with NaN."
            )
            result[self.name] = result[self.name].replace([np.inf, -np.inf], np.nan)


# ==================== SPECIALIZED SUBCLASSES ====================

class TechnicalFactor(BaseFactor):
    def __init__(self, name: str, description: str, lookback_period: int, **kwargs):
        super().__init__(name, 'technical', description, lookback_period, **kwargs)


class FundamentalFactor(BaseFactor):
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
        Override: Fundamental datasets often use wide formats or sparse indexing,
        relaxing the strict (date, ticker) requirement found in time-series data.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")


class EarningsFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        # FIX BUG-026: normalize=False, winsorize=False (default) — per-fold only.
        super().__init__(
            name, 'earnings', description,
            lookback_period=None,
            normalize=False,
            winsorize=False,
            **kwargs,
        )


class AlternativeFactor(BaseFactor):
    def __init__(self, name: str, description: str, **kwargs):
        # Alternative/Macro data is often absolute (VIX, Sentiment) and does not
        # require cross-sectional normalization.
        kwargs.setdefault('normalize', False)
        kwargs.setdefault('winsorize', False)
        super().__init__(name, 'alternative', description, lookback_period=None, **kwargs)

    def _validate_input(self, data: pd.DataFrame):
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")


class CompositeFactor(BaseFactor):
    """Composite factors blend multiple signal types for regime-aware ranking strategies."""

    def __init__(self, name: str, description: str, **kwargs):
        kwargs.setdefault('normalize', False)
        kwargs.setdefault('winsorize', False)
        super().__init__(name, 'composite', description, lookback_period=None, **kwargs)

    def _validate_input(self, data: pd.DataFrame):
        """
        Override: Composite calculations may involve heterogeneous data sources
        (Macro + Ticker-level), requiring flexible validation.
        """
        if data.empty:
            raise ValueError(f"Empty DataFrame provided to {self.name}")