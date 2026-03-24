"""
Factor Registry and Orchestration Engine
========================================
Centralized command and control system for the discovery, configuration, and
parallel execution of alpha factors.

Purpose
-------
This module implements a Singleton-based registry pattern to manage the lifecycle
of alpha factors. It dynamically instantiates registered feature engineering
classes and orchestrates their parallel execution across the target universe.
Additionally, it provides robust multicollinearity filtering via year-stratified
correlation analysis.

Role in Quantitative Workflow
-----------------------------
Acts as the central integration point for the feature engineering pipeline.
Consumed by both the training (`train_models.py`) and inference 
(`generate_predictions.py`) layers to ensure deterministic and parallelized 
factor computation.

Dependencies
------------
- **Pandas/NumPy**: In-memory data manipulation and correlation matrix computation.
- **Concurrent.Futures**: Thread-pool execution for parallel factor calculations.
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
        _registered_classes (Dict[str, Type[BaseFactor]]): Class-level storage for registered types.
        factors (Dict[str, BaseFactor]): Instance-level storage for initialized factor objects.
        factor_config (Dict[str, Any]): Hyperparameters injected during initialization.
    """

    _registered_classes: Dict[str, Type[BaseFactor]] = {}

    def __init__(self, factor_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the FactorRegistry with specific hyperparameters.

        Args:
            factor_config (Optional[Dict[str, Any]]): Dictionary mapping factor class 
                names to their respective initialization arguments. Defaults to None.
        """
        self.factors: Dict[str, BaseFactor] = {}
        self.factor_config = factor_config or {}
        self._initialize_factors()

    @classmethod
    def register(cls):
        """
        Decorator: Registers a factor blueprint with the global registry.

        Args:
            None

        Returns:
            Callable: A wrapper function that adds the class to the registry.
        """
        def wrapper(factor_class: Type[BaseFactor]):
            key = factor_class.__name__
            cls._registered_classes[key] = factor_class
            return factor_class
        return wrapper

    def _initialize_factors(self):
        """
        Instantiates all registered factor blueprints mapping internal parameters.

        Iterates through the globally registered class blueprints, injects
        their respective configurations, and provisions them into the active
        instance dictionary for computation.

        Args:
            None

        Returns:
            None
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

    @staticmethod
    def _compute_single_wrapper(
        factor_instance: BaseFactor,
        df: pd.DataFrame,
        original_index: pd.Index,
    ) -> Optional[pd.Series]:
        """
        Static worker method executing a single factor's computation logic.

        Args:
            factor_instance (BaseFactor): The instantiated factor object to execute.
            df (pd.DataFrame): The target dataset containing prerequisite features.
            original_index (pd.Index): The expected index to ensure structural alignment.

        Returns:
            Optional[pd.Series]: The computed factor values aligned to the original 
                index, or None if computation fails or returns exclusively NaNs.
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

        Args:
            df (pd.DataFrame): The raw historical data warehouse slice.
            max_workers (int, optional): The maximum number of concurrent threads. 
                Defaults to 4.

        Returns:
            pd.DataFrame: A concatenated dataframe integrating the original dataset
                with all successfully computed factor series.
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

            # Identifies overlapping keys explicitly tracking prior artifacts, dropping arrays 
            # seamlessly to ensure identical parameters securely overwrite legacy extractions.
            overlap_cols = [c for c in features_df.columns if c in df.columns]
            if overlap_cols:
                df = df.drop(columns=overlap_cols)

            final_df = pd.concat([df, features_df], axis=1)
        else:
            final_df = df

        elapsed = time.perf_counter() - start_time
        logger.info(f"✅ Computed {success_count}/{len(self.factors)} factors in {elapsed:.2f}s")
        return final_df

    def select_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Filters highly correlated feature structures explicitly minimizing linear matrix redundancies.

        Implements standard mathematical year-stratified sampling routines strictly bounded to prevent 
        non-stationary temporal contamination explicitly bridging longitudinal parameters.

        Args:
            df (pd.DataFrame): The aggregated feature matrix.
            threshold (float, optional): The absolute Pearson correlation limit 
                above which features are deemed redundant. Defaults to 0.95.

        Returns:
            List[str]: A list of selected feature column names satisfying the 
                orthogonality constraints.
        """
        factor_cols = [f for f in self.factors.keys() if f in df.columns]
        if not factor_cols:
            return []

        logger.info("🔍 Analyzing Feature Correlations...")

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
                corr_matrix = (
                    df[factor_cols]
                    .sample(100_000, random_state=42)
                    .corr()
                    .abs()
                )
        else:
            corr_matrix = df[factor_cols].corr().abs()

        # Isolates diagonal structures mapped directly extracting mathematical bounds bypassing 
        # computationally repetitive overlapping arrays securely.
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop  = [col for col in upper.columns if any(upper[col] > threshold)]
        selected = [f for f in factor_cols if f not in to_drop]

        logger.info(
            f"✂️ Dropped {len(to_drop)} correlated features. Kept {len(selected)}."
        )
        return selected

    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """
        Retrieves a specific instantiated factor by name.

        Args:
            name (str): The string identifier of the target factor.

        Returns:
            Optional[BaseFactor]: The factor instance, or None if not found.
        """
        return self.factors.get(name)

    def list_factors(self) -> List[str]:
        """
        Returns a list of all actively registered factor names.

        Args:
            None

        Returns:
            List[str]: The string identifiers of all active factors.
        """
        return list(self.factors.keys())

    def clear(self):
        """
        Purges all instantiated factors from the active registry.

        Args:
            None

        Returns:
            None
        """
        self.factors = {}
        logger.info("🗑️ Cleared all factors.")

    def __len__(self) -> int:
        """
        Returns the total number of instantiated factors.

        Args:
            None

        Returns:
            int: The count of active factors in the registry.
        """
        return len(self.factors)

    def print_registry(self):
        """
        Outputs the current registry state and category distributions to standard output.

        Args:
            None

        Returns:
            None
        """
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