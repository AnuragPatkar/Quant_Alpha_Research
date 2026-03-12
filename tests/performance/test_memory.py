"""
PERFORMANCE TEST: Memory Usage
==============================
Monitors peak memory usage of critical data processing functions.
Ensures that memory usage scales linearly (or better) with data size,
and that no large intermediate copies are retained.

Focus areas:
  1. Fold generation (train_models.py) - previously O(N_folds * Data_size)
  2. Feature Engineering (registry) - should not explode RAM
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import gc
import psutil
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Virtual Filesystem Mock (Pre-import)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def memory_test_context():
    """
    Prepopulate sys.modules with stubs for heavy ML dependencies and 
    internal modules not under test, preventing ImportErrors and reducing RAM.
    
    Returns the imported modules needed for testing.
    """
    # External heavy libs
    mock_targets = [
        "lightgbm", "xgboost", "catboost",
        "sklearn.covariance", "matplotlib", "matplotlib.pyplot", "seaborn"
    ]
    
    # Internal modules we don't want to load/test here
    # We ONLY want features.registry and train_models logic.
    internal_mocks = [
        "quant_alpha.models",
        "quant_alpha.models.lightgbm_model",
        "quant_alpha.models.xgboost_model",
        "quant_alpha.models.catboost_model",
        "quant_alpha.models.trainer",
        "quant_alpha.models.feature_selector",
        "quant_alpha.backtest",
        "quant_alpha.backtest.engine",
        "quant_alpha.backtest.metrics",
        "quant_alpha.backtest.attribution",
        "quant_alpha.optimization",
        "quant_alpha.optimization.allocator",
        "quant_alpha.visualization",
        "quant_alpha.data",
        "quant_alpha.data.DataManager"
    ]
    
    with patch.dict(sys.modules):
        for mod in mock_targets + internal_mocks:
            m = MagicMock()
            m.__name__ = mod
            m.__file__ = f"mock://{mod}"
            # Ensure submodules can be imported from it if it's a package
            if "." not in mod or "quant_alpha" in mod:
                m.__path__ = []
            sys.modules[mod] = m

        # Import the modules under test dynamically inside the patch context
        try:
            import scripts.train_models as tm
            from quant_alpha.features.registry import FactorRegistry
            # Ensure feature modules are loaded to populate registry
            import quant_alpha.features.technical.volatility
            
            yield {
                "train_models": tm,
                "FactorRegistry": FactorRegistry
            }
        except ImportError as e:
            pytest.skip(f"Could not import modules under test: {e}")

# ---------------------------------------------------------------------------
# Memory Measurement
# ---------------------------------------------------------------------------
_RAM_BASELINE_MB = 0.0

def calibrate_memory_baseline():
    """Set the current RSS as the 0.0 MB baseline."""
    global _RAM_BASELINE_MB
    gc.collect()
    process = psutil.Process(os.getpid())
    _RAM_BASELINE_MB = process.memory_info().rss / 1024 / 1024

def get_process_memory_mb():
    """Return current process RSS memory in MB relative to baseline."""
    process = psutil.Process(os.getpid())
    return (process.memory_info().rss / 1024 / 1024) - _RAM_BASELINE_MB

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMemoryUsage:
    
    @pytest.fixture(autouse=True)
    def memory_warmup(self):
        """
        Warm up the memory allocator and imports to establish a stable baseline.
        This prevents one-time import costs from looking like leaks.
        """
        gc.collect()
        # Allocate and free some memory to warm up the heap
        _ = np.ones((1024, 1024))
        gc.collect()
        
        # Calibrate baseline before test runs
        calibrate_memory_baseline()
        yield
        gc.collect()

    def test_fold_boundaries_scaling(self, memory_test_context):
        """
        O(1) Check: Verify that memory usage does not scale with the number of folds.
        
        v3 implementation created N copies of the dataset (O(N)).
        v4 implementation stores only date tuples (O(1) relative to data size).
        
        We compare memory usage of generating 10 folds vs 100 folds.
        The delta should be negligible (< 1 MB).
        """
        _compute_fold_boundaries = memory_test_context["train_models"]._compute_fold_boundaries

        # Generate enough dates for ~10 folds
        # Min train (36m) + 10 * Step (3m) = 66 months ~ 1400 days
        dates_10 = pd.Series(pd.date_range("2000-01-01", periods=1500, freq="B"))
        
        gc.collect()
        mem_before_10 = get_process_memory_mb()
        folds_10 = _compute_fold_boundaries(dates_10)
        mem_after_10 = get_process_memory_mb()
        
        # Generate enough dates for ~100 folds
        # Min train (36m) + 100 * Step (3m) = 336 months ~ 7100 days
        dates_100 = pd.Series(pd.date_range("2000-01-01", periods=7500, freq="B"))
        
        gc.collect()
        mem_before_100 = get_process_memory_mb()
        folds_100 = _compute_fold_boundaries(dates_100)
        mem_after_100 = get_process_memory_mb()
        
        # Calculate net memory cost of the folds structure itself
        # (subtracting the baseline which might have shifted slightly due to the larger dates series)
        cost_10 = mem_after_10 - mem_before_10
        cost_100 = mem_after_100 - mem_before_100
        
        delta = abs(cost_100 - cost_10)
        
        print(f"\nMemory Cost (10 folds): {cost_10:.4f} MB")
        print(f"Memory Cost (100 folds): {cost_100:.4f} MB")
        print(f"Delta: {delta:.4f} MB")
        
        # Assertion: The difference in memory usage between storing 10 tuples and 100 tuples
        # should be tiny. If it were copying data, 100 folds would be massive.
        assert delta < 1.0, f"Memory usage scales with folds! Delta: {delta:.2f} MB (Expected < 1.0 MB)"

    def test_feature_generation_memory_cleanup(self, memory_test_context):
        """
        Verify that computing features does not leak memory.
        """
        FactorRegistry = memory_test_context["FactorRegistry"]

        # Create a moderately large dataset: 50 tickers, 1000 days = 50k rows
        n_tickers = 50
        n_days = 1000
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        
        index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=index).reset_index()
        rng = np.random.default_rng(42)
        df["close"] = 100.0 + rng.standard_normal(len(df)).cumsum()
        df["open"] = df["close"]
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["volume"] = 1000000.0
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        
        # Warmup baseline (reset for this test)
        calibrate_memory_baseline()
        mem_start = get_process_memory_mb()
        
        registry = FactorRegistry()
        # Just compute one complex factor to see if it cleans up
        if "volatility_21d" in registry.factors:
            f = registry.factors["volatility_21d"]
            res = f.calculate(df)
            
            # Force cleanup
            del res
            gc.collect()
        
        mem_end = get_process_memory_mb()
        
        # Allow some small growth due to internal caching or python overhead (e.g. JIT cache)
        # But it shouldn't be proportional to data size.
        diff = mem_end - mem_start
        print(f"\nFeature Calc Memory Delta: {diff:.2f} MB")
        
        # 50MB tolerance for Numba JIT overhead / module loading if this runs first
        assert diff < 50.0, f"Memory leak detected: {diff:.2f} MB retained after calculation"
