"""
Factor Registry & Orchestration Engine
======================================
Centralized command and control system for the discovery, configuration, and 
parallel execution of alpha factors.

Purpose
-------
The `FactorRegistry` serves as the kernel of the feature engineering pipeline. 
It decouples factor definition (in individual files) from execution logic.
It implements a **Plugin Architecture** where factors register themselves 
via decorators, allowing the pipeline to dynamically discover 100+ signals 
without manual import manifest maintenance.

Usage
-----
.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry

    # 1. Instantiate registry (auto-discovers decorated classes)
    registry = FactorRegistry()

    # 2. Compute all factors on a market data frame
    features_df = registry.compute_all(market_data_df, max_workers=8)

Importance
----------
- **Scalability**: Utilizing `ThreadPoolExecutor`, it parallelizes I/O-bound 
  and GIL-releasing NumPy operations, reducing computation time for large 
  cross-sectional datasets ($T \times N$).
- **Configuration Injection**: Separates algorithmic logic from parameter 
  tuning (e.g., lookback windows) via a centralized config dictionary.
- **Multicollinearity Reduction**: Built-in feature selection ensures the 
  downstream ML models are not fed highly correlated signals ($|r| > 0.95$).

Tools & Frameworks
------------------
- **ThreadPoolExecutor**: Manages concurrent factor execution.
- **Pandas**: Data alignment and merging strategies.
- **NumPy**: Efficient correlation matrix calculation ($X^T X$).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Type, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config.logging_config import logger
from .base import BaseFactor

class FactorRegistry:
    """
    Registry Singleton (Pattern) for managing Factor lifecycles.
    
    Attributes:
        _registered_classes (Dict): Class-level storage for registered types (Plugin pattern).
        factors (Dict): Instance-level storage for initialized factor objects.
        factor_config (Dict): Hyperparameters injected during initialization.
    """
    # Global Blueprint: Stores Class References to prevent circular imports/instantiation order issues
    _registered_classes: Dict[str, Type[BaseFactor]] = {}
    
    def __init__(self, factor_config: Optional[Dict[str, Any]] = None):
        # Active Worker Instances
        self.factors: Dict[str, BaseFactor] = {}
        # Configuration Injection
        self.factor_config = factor_config or {}
        self._initialize_factors()

    @classmethod
    def register(cls):
        """
        Decorator: Registers a factor class with the global registry blueprint.
        
        Strategy:
        Uses `cls._registered_classes` to hold a reference to the class definition
        without instantiating it. Instantiation is deferred until `__init__` is called
        with the specific runtime configuration.
        """
        def wrapper(factor_class: Type[BaseFactor]):
            # Key Strategy: Use class name as the unique identifier
            key = factor_class.__name__
            cls._registered_classes[key] = factor_class
            return factor_class
        return wrapper

    def _initialize_factors(self):
        """
        Factory Method: Instantiates all registered factor classes.
        
        Injects specific configuration parameters (e.g., lookback_window) from
        `self.factor_config` into each factor's constructor.
        """
        for class_name, factor_cls in self._registered_classes.items():
            try:
                # 1. Parameter Resolution
                specific_config = self.factor_config.get(class_name, {})
                
                # 2. Dependency Injection
                instance = factor_cls(**specific_config)
                
                self.factors[instance.name] = instance
            except TypeError as e:
                logger.error(f"❌ Init Error {class_name}: Missing Arguments? {e}")
            except Exception as e:
                logger.error(f"❌ Failed to load {class_name}: {e}")
                
        logger.info(f"✅ FactorRegistry initialized with {len(self.factors)} factors.")

    # ==================== COMPUTATION ENGINE ====================

    @staticmethod
    def _compute_single_wrapper(factor_instance: BaseFactor, df: pd.DataFrame, original_index: pd.Index) -> Optional[pd.Series]:
        """
        Static worker method for concurrent execution.
        
        Design Choice:
        Marked `@staticmethod` to ensure the callable is pickleable by process/thread pools,
        preventing the implicit capture of the entire `FactorRegistry` state (closures).
        
        Args:
            factor_instance: The factor object to execute.
            df: Market data.
            original_index: The reference index for alignment safety.
        """
        try:
            # Execution
            result = factor_instance.calculate(df)
            
            # Data Alignment & Integrity
            if factor_instance.name in result.columns:
                series = result[factor_instance.name]
                
                # Strict Index Alignment: Ensure output matches input dimensions
                if not series.index.equals(original_index):
                    return series.reindex(original_index)
                return series
                
            return None
        except Exception as e:
            logger.error(f"❌ Error in {factor_instance.name}: {e}")
            return None

    def compute_all(self, df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
        """
        Orchestrates the parallel computation of all registered factors.
        
        Concurrency Model:
        Uses `ThreadPoolExecutor`. While Python has a GIL, most heavy-lifting here
        is done by NumPy/Pandas (which release the GIL), or is I/O bound.
        
        Complexity:
        $$ O(\frac{F \cdot N}{k}) $$ where $F$ is factors, $N$ is rows, $k$ is workers.
        """
        if df.empty: return df
        
        # Data Pre-conditioning:
        # Enforce lexicographical sort order (Ticker -> Date) to guarantee 
        # correct rolling window calculation for time-series factors.
        if 'date' in df.columns and 'ticker' in df.columns:
             # Performance Cost: $O(N \log N)$
             df = df.sort_values(['ticker', 'date'])

        logger.info(f"⚙️ Computing {len(self.factors)} factors (Parallel)...")
        start_time = time.perf_counter()
        
        # Result Accumulator
        new_features = []
        
        # Snapshot index for downstream alignment
        original_index = df.index

        # Parallel Execution Block
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_factor = {
                executor.submit(self._compute_single_wrapper, factor, df, original_index): factor 
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

        # 2. Data Merging Strategy
        if new_features:
            logger.info("🔗 Merging features...")
            
            # Efficient Columnar Concatenation ($O(N)$)
            features_df = pd.concat(new_features, axis=1)
            
            # Horizontal Join with Base Data
            final_df = pd.concat([df, features_df], axis=1)
            
            # Deduplication Safety check
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        else:
            final_df = df

        elapsed = time.perf_counter() - start_time
        logger.info(f"✅ Computed {success_count}/{len(self.factors)} factors in {elapsed:.2f}s")
        return final_df

    # ==================== FEATURE SELECTION ====================

    def select_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Multicollinearity Filter: Removes highly correlated features.
        
        Identifies clusters of redundant signals and retains only one representative
        per cluster to stabilize downstream linear models and reduce overfitting.
        
        Algorithm:
        1. Compute Correlation Matrix ($X^T X$).
        2. Identify pairs with $|r| > \text{threshold}$.
        3. Drop the second element of the pair.
        """
        factor_cols = [f for f in self.factors.keys() if f in df.columns]
        if not factor_cols: return []
        
        logger.info("🔍 Analyzing Feature Correlations...")
        
        # Stochastic Approximation: 
        # Estimate correlation on a subsample ($N=100k$) to maintain constant memory 
        # complexity relative to dataset size.
        if len(df) > 100000:
             corr_matrix = df[factor_cols].sample(100000).corr().abs()
        else:
             corr_matrix = df[factor_cols].corr().abs()
        
        # Select upper triangle of correlation matrix to avoid double counting
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify columns to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        selected = [f for f in factor_cols if f not in to_drop]
        
        logger.info(f"✂️ Dropped {len(to_drop)} correlated features. Kept {len(selected)}.")
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
        """Diagnostics: Outputs the current registry state and category distribution to stdout."""
        print(f"\n📚 Factor Registry Status:")
        print(f"   - Total Factors: {len(self.factors)}")
        
        categories = {}
        for f in self.factors.values():
            categories.setdefault(f.category, []).append(f.name)
            
        for cat, names in categories.items():
            print(f"   📂 {cat.upper()} ({len(names)}):")
            for name in names[:5]:
                print(f"      - {name}")
            if len(names) > 5:
                print(f"      - ... and {len(names)-5} more")