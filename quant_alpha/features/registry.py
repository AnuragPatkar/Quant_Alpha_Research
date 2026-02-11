import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Type, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config.logging_config import logger
from .base import BaseFactor

class FactorRegistry:
    """
    The Ultimate Factor Registry (Production Hardened).
    
    Fixed Critical Issues:
    1. Alignment Safety: Forces index alignment using .reindex(df.index)
    2. Memory Safety: Removed internal copying of DataFrames
    3. Import Safety: Lazy registration (No instantiation at import time)
    4. Introspection: Added utility methods for debugging
    """
    
    # Global Blueprint (Stores Class References)
    _registered_classes: Dict[str, Type[BaseFactor]] = {}
    
    def __init__(self, factor_config: Optional[Dict[str, Any]] = None):
        # Worker Instances
        self.factors: Dict[str, BaseFactor] = {}
        # Agar config mili toh save karo, nahi toh empty dict (Fallback)
        self.factor_config = factor_config or {}
        self._initialize_factors()

    @classmethod
    def register(cls):
        """
        Decorator to register a factor class.
        Safe: Does NOT instantiate class at import time.
        """
        def wrapper(factor_class: Type[BaseFactor]):
            # Use class name as temporary key
            key = factor_class.__name__
            cls._registered_classes[key] = factor_class
            return factor_class
        return wrapper

    def _initialize_factors(self):
        """
        Instantiate all registered factor classes safely with Config Injection.
        """
        for class_name, factor_cls in self._registered_classes.items():
            try:
                # 1. Lookup Config: Check if there are specific settings for this factor.
                # Example: config = {'RSI': {'period': 21}}
                specific_config = self.factor_config.get(class_name, {})
                
                # 2. Instantiate with Config (using **kwargs unpacking)
                # If config is empty, the Factor's default values will be used (Fallback mechanism).
                instance = factor_cls(**specific_config)
                
                self.factors[instance.name] = instance
            except TypeError as e:
                logger.error(f"❌ Init Error {class_name}: Missing Arguments? {e}")
            except Exception as e:
                logger.error(f"❌ Failed to load {class_name}: {e}")
                
        logger.info(f"✅ FactorRegistry initialized with {len(self.factors)} factors.")

    # ==================== COMPUTATION ENGINE ====================

    def compute_all(self, df: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
        """
        Parallel Factor Computation Engine.
        """
        if df.empty: return df
        
        # 1. Sort once (Critical for Rolling Windows)
        if 'ticker' in df.columns and 'date' in df.columns:
            df = df.sort_values(['ticker', 'date'])

        logger.info(f"⚙️ Computing {len(self.factors)} factors (Parallel)...")
        start_time = time.time()
        
        # Container for new feature columns
        new_features = []
        
        # Capture the original index structure for alignment safety
        original_index = df.index

        def compute_single(factor):
            try:
                # PASS REFERENCE (Read-Only Concept) - Saves Memory
                result = factor.calculate(df)
                
                # CRITICAL FIX: Alignment Safety
                # If factor dropped rows (dropna), we force it back to original shape
                if factor.name in result.columns:
                    series = result[factor.name]
                    
                    # Check if realignment is needed (Performance Opt)
                    if not series.index.equals(original_index):
                        return series.reindex(original_index)
                    return series
                    
                return None
            except Exception as e:
                logger.error(f"❌ Error in {factor.name}: {e}")
                return None

        # Threading for I/O bound or GIL-releasing NumPy tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_single, f): f for f in self.factors.values()}
            
            success = 0
            for future in as_completed(futures):
                factor = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        new_features.append(res)
                        success += 1
                except Exception as e:
                    logger.error(f"❌ Worker failed for {factor.name}: {e}")

        # 2. Merge Strategy (Optimized)
        if new_features:
            logger.info("🔗 Merging features...")
            # concat along columns (axis=1). 
            # Since we forced .reindex(), alignment is guaranteed.
            features_df = pd.concat(new_features, axis=1)
            
            # Combine with original data
            final_df = pd.concat([df, features_df], axis=1)
            
            # Remove duplicate columns if any
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        else:
            final_df = df

        elapsed = time.time() - start_time
        logger.info(f"✅ Computed {success}/{len(self.factors)} factors in {elapsed:.2f}s")
        return final_df

    # ==================== FEATURE SELECTION ====================

    def select_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Removes highly correlated features.
        """
        factor_cols = [f for f in self.factors.keys() if f in df.columns]
        if not factor_cols: return []
        
        logger.info("🔍 Analyzing Feature Correlations...")
        
        # Optimization for large datasets
        if len(df) > 100000:
             corr_matrix = df[factor_cols].sample(100000).corr().abs()
        else:
             corr_matrix = df[factor_cols].corr().abs()
        
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        selected = [f for f in factor_cols if f not in to_drop]
        
        logger.info(f"✂️ Dropped {len(to_drop)} correlated features. Kept {len(selected)}.")
        return selected

    # ==================== UTILITIES (DEBUGGING) ====================

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
        """Pretty print the registry status."""
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