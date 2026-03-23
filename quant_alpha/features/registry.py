"""
Factor Registry & Orchestration Engine
======================================
Centralized command and control system for the discovery, configuration, and
parallel execution of alpha factors.

FIXES:
  BUG-030: select_features() now uses year-stratified sampling when the dataset
           is large (> 100,000 rows) instead of a random unstratified sample.
           An unstratified sample on a 10-year panel under-samples early years
           and oversamples recent years, producing a time-contaminated correlation
           estimate that causes the wrong features to be dropped or kept.
           Minimum 60 rows per year enforced (G6 from known project context).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Type, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config.logging_config import logger
from .base import BaseFactor


class FactorRegistry:
    """
    Registry Singleton for managing Factor lifecycles.

    Attributes:
        _registered_classes (Dict): Class-level storage for registered types.
        factors (Dict): Instance-level storage for initialized factor objects.
        factor_config (Dict): Hyperparameters injected during initialization.
    """

    _registered_classes: Dict[str, Type[BaseFactor]] = {}

    def __init__(self, factor_config: Optional[Dict[str, Any]] = None):
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_config = factor_config or {}
        self._initialize_factors()

    @classmethod
    def register(cls):
        """
        Decorator: Registers a factor class with the global registry blueprint.
        """
        def wrapper(factor_class: Type[BaseFactor]):
            key = factor_class.__name__
            cls._registered_classes[key] = factor_class
            return factor_class
        return wrapper

    def _initialize_factors(self):
        """
        Factory Method: Instantiates all registered factor classes.
        """
        for class_name, factor_cls in self._registered_classes.items():
            try:
                specific_config = self.factor_config.get(class_name, {})
                instance = factor_cls(**specific_config)
                self.factors[instance.name] = instance
            except TypeError as e:
                logger.error(f"❌ Init Error {class_name}: Missing Arguments? {e}")
            except Exception as e:
                logger.error(f"❌ Failed to load {class_name}: {e}")

        logger.info(f"✅ FactorRegistry initialized with {len(self.factors)} factors.")

    # ==================== COMPUTATION ENGINE ====================

    @staticmethod
    def _compute_single_wrapper(
        factor_instance: BaseFactor,
        df: pd.DataFrame,
        original_index: pd.Index,
    ) -> Optional[pd.Series]:
        """
        Static worker method for concurrent execution.
        """
        try:
            result = factor_instance.calculate(df)

            if isinstance(result, pd.Series):
                series = result
                series.name = factor_instance.name
            elif isinstance(result, pd.DataFrame) and factor_instance.name in result.columns:
                series = result[factor_instance.name]
            else:
                return None

            if series.isna().all():
                logger.warning(
                    f"⚠️ {factor_instance.name} skipped: 100% NaNs (Missing required data)."
                )
                return None

            if not series.index.equals(original_index):
                return series.reindex(original_index)
            return series

        except Exception as e:
            logger.error(f"❌ Error in {factor_instance.name}: {e}")
            return None

    def compute_all(self, df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
        """
        Orchestrates the parallel computation of all registered factors.
        """
        if df.empty:
            return df

        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.sort_values(['ticker', 'date'])

        logger.info(f"⚙️ Computing {len(self.factors)} factors (Parallel)...")
        start_time = time.perf_counter()

        new_features = []
        original_index = df.index

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_factor = {
                executor.submit(
                    self._compute_single_wrapper, factor, df, original_index
                ): factor
                for factor in self.factors.values()
            }

            success_count = 0

            for future in as_completed(future_to_factor):
                factor = future_to_factor[future]
                try:
                    res = future.result()
                    if res is not None:
                        new_features.append(res)
                        success_count += 1
                except Exception as e:
                    logger.error(f"❌ Worker failed for {factor.name}: {e}")

        if new_features:
            logger.info("🔗 Merging features...")
            features_df = pd.concat(new_features, axis=1)

            # Drop overlapping columns so new computations overwrite cached values
            overlap_cols = [c for c in features_df.columns if c in df.columns]
            if overlap_cols:
                df = df.drop(columns=overlap_cols)

            final_df = pd.concat([df, features_df], axis=1)
        else:
            final_df = df

        elapsed = time.perf_counter() - start_time
        logger.info(f"✅ Computed {success_count}/{len(self.factors)} factors in {elapsed:.2f}s")
        return final_df

    # ==================== FEATURE SELECTION ====================

    def select_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Multicollinearity Filter: Removes highly correlated features.

        FIX BUG-030: When len(df) > 100,000 rows, uses year-stratified sampling
        (minimum 60 rows per year) instead of a purely random sample. A random
        sample on a multi-year panel concentrates observations in years with more
        data, producing a time-contaminated correlation estimate.

        Algorithm:
        1. Compute Correlation Matrix.
        2. Identify pairs with |r| > threshold.
        3. Drop the second element of each correlated pair.
        """
        factor_cols = [f for f in self.factors.keys() if f in df.columns]
        if not factor_cols:
            return []

        logger.info("🔍 Analyzing Feature Correlations...")

        # FIX BUG-030: Year-stratified sampling preserves time structure.
        if len(df) > 100_000:
            if 'date' in df.columns:
                years = df['date'].dt.year.unique()
                n_per_year = max(60, 100_000 // len(years))
                logger.info(
                    f"[FactorRegistry] Stratified correlation sample: "
                    f"{len(years)} years × {n_per_year} rows/year"
                )
                sample = (
                    df.groupby(df['date'].dt.year, group_keys=False)[factor_cols]
                      .apply(
                          lambda g: g.sample(
                              min(len(g), n_per_year),
                              random_state=42,
                          )
                      )
                )
                corr_matrix = sample.corr().abs()
            else:
                # No date column — fall back to random sample with fixed seed
                corr_matrix = (
                    df[factor_cols]
                    .sample(100_000, random_state=42)
                    .corr()
                    .abs()
                )
        else:
            corr_matrix = df[factor_cols].corr().abs()

        # Upper triangle only to avoid double-counting
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop  = [col for col in upper.columns if any(upper[col] > threshold)]
        selected = [f for f in factor_cols if f not in to_drop]

        logger.info(
            f"✂️ Dropped {len(to_drop)} correlated features. Kept {len(selected)}."
        )
        return selected

    # ==================== UTILITIES ====================

    def get_factor(self, name: str) -> Optional[BaseFactor]:
        return self.factors.get(name)

    def list_factors(self) -> List[str]:
        return list(self.factors.keys())

    def clear(self):
        self.factors = {}
        logger.info("🗑️ Cleared all factors.")

    def __len__(self) -> int:
        return len(self.factors)

    def print_registry(self):
        """Diagnostics: Outputs the current registry state to stdout."""
        print(f"\n📚 Factor Registry Status:")
        print(f"   - Total Factors: {len(self.factors)}")

        categories: Dict[str, list] = {}
        for f in self.factors.values():
            categories.setdefault(f.category, []).append(f.name)

        for cat, names in categories.items():
            print(f"   📂 {cat.upper()} ({len(names)}):")
            for name in names[:5]:
                print(f"      - {name}")
            if len(names) > 5:
                print(f"      - ... and {len(names) - 5} more")