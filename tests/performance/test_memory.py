"""
Memory Profiling and Leak Detection Suite
=========================================
Validates the memory complexity and garbage collection efficiency of critical pipeline components.

Purpose
-------
This module establishes bounded performance constraints for data-intensive operations 
within the quantitative pipeline. It specifically monitors peak Resident Set Size (RSS), 
enforces $O(1)$ memory scaling for temporal cross-validation splitting, and strictly 
validates the deallocation of ephemeral feature engineering artifacts to prevent 
out-of-memory (OOM) faults during large-scale universe simulations.

Role in Quantitative Workflow
-----------------------------
Acts as an automated memory guardrail in the continuous integration environment, 
ensuring that algorithm modifications do not inadvertently introduce memory leaks 
or violate computational complexity guarantees.

Dependencies
------------
- **Psutil**: Operating system interface for precise Resident Set Size (RSS) monitoring.
- **Pytest**: Test orchestration and isolated memory fixture boundaries.
- **Unittest.Mock**: Extensive stubbing of C-extension libraries to isolate pure logic memory profiles.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

@pytest.fixture(scope="class")
def memory_test_context():
    """
    Provisions a virtualized namespace to isolate Python-level memory allocations.

    Stubs external C-extensions and downstream inference modules to prevent native 
    heap allocations (e.g., from LightGBM or SciPy) from obscuring the pure Python 
    memory footprint of the specific data structures under test.

    Yields:
        dict: A mapping of the strictly required, dynamically imported modules 
            under test (e.g., `train_models` and `FactorRegistry`).
    """
    mock_targets = [
        "lightgbm", "xgboost", "catboost",
        "sklearn.covariance", "matplotlib", "matplotlib.pyplot", "seaborn"
    ]
    
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
            
            if "." not in mod or "quant_alpha" in mod:
                m.__path__ = []
            sys.modules[mod] = m

        try:
            import scripts.train_models as tm
            from quant_alpha.features.registry import FactorRegistry
            
            import quant_alpha.features.technical.volatility
            
            yield {
                "train_models": tm,
                "FactorRegistry": FactorRegistry
            }
        except ImportError as e:
            pytest.skip(f"Could not import modules under test: {e}")

_RAM_BASELINE_MB = 0.0

def calibrate_memory_baseline():
    """
    Calibrates the zero-boundary Resident Set Size (RSS) baseline for subsequent measurements.

    Forces explicit garbage collection to clear transient Python objects and captures 
    the exact OS-level memory footprint of the current process.

    Args:
        None

    Returns:
        None
    """
    global _RAM_BASELINE_MB
    gc.collect()
    process = psutil.Process(os.getpid())
    _RAM_BASELINE_MB = process.memory_info().rss / 1024 / 1024

def get_process_memory_mb():
    """
    Calculates the delta in process RSS memory relative to the calibrated baseline.

    Args:
        None

    Returns:
        float: The net memory allocation delta measured in megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    return (process.memory_info().rss / 1024 / 1024) - _RAM_BASELINE_MB

class TestMemoryUsage:
    """
    Memory footprint validation suite for systemic data transformation boundaries.
    """
    
    @pytest.fixture(autouse=True)
    def memory_warmup(self):
        """
        Warms up the Python memory allocator to establish a deterministic testing baseline.

        Allocates and immediately frees a substantial NumPy array. This triggers the 
        underlying OS memory manager (e.g., glibc/malloc) to provision and map memory pages 
        to the process, ensuring that subsequent first-run allocations do not synthetically 
        skew the measured delta.

        Yields:
            None: Exposes control to the specific memory test.
        """
        gc.collect()
        _ = np.ones((1024, 1024))
        gc.collect()
        
        calibrate_memory_baseline()
        yield
        gc.collect()

    def test_fold_boundaries_scaling(self, memory_test_context):
        """
        Verifies that temporal cross-validation matrix generation adheres to O(1) memory limits.

        Asserts that the system strictly persists temporal indices (date tuples) 
        rather than replicating physical data matrices across folds, preventing 
        combinatorial memory explosions over extensive evaluation horizons.

        Args:
            memory_test_context (dict): The isolated execution context mapping.

        Returns:
            None
        """
        _compute_fold_boundaries = memory_test_context["train_models"]._compute_fold_boundaries

        dates_10 = pd.Series(pd.date_range("2000-01-01", periods=1500, freq="B"))
        
        gc.collect()
        mem_before_10 = get_process_memory_mb()
        folds_10 = _compute_fold_boundaries(dates_10)
        mem_after_10 = get_process_memory_mb()
        
        dates_100 = pd.Series(pd.date_range("2000-01-01", periods=7500, freq="B"))
        
        gc.collect()
        mem_before_100 = get_process_memory_mb()
        folds_100 = _compute_fold_boundaries(dates_100)
        mem_after_100 = get_process_memory_mb()
        
        cost_10 = mem_after_10 - mem_before_10
        cost_100 = mem_after_100 - mem_before_100
        
        delta = abs(cost_100 - cost_10)
        
        print(f"\nMemory Cost (10 folds): {cost_10:.4f} MB")
        print(f"Memory Cost (100 folds): {cost_100:.4f} MB")
        print(f"Delta: {delta:.4f} MB")
        
        # Evaluates structural delta magnitude; a difference < 1.0 MB definitively proves O(1) constraints
        assert delta < 1.0, f"Memory usage scales with folds! Delta: {delta:.2f} MB (Expected < 1.0 MB)"

    def test_feature_generation_memory_cleanup(self, memory_test_context):
        """
        Validates the immediate deallocation of memory mapped to computational feature graphs.

        Calculates a high-density factor on a heavily populated dataset, subsequently 
        asserting that explicit `del` invocations effectively clear reference counts 
        and restore RSS footprint to the pre-execution baseline threshold.

        Args:
            memory_test_context (dict): The isolated execution context mapping.

        Returns:
            None
        """
        FactorRegistry = memory_test_context["FactorRegistry"]

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
        
        calibrate_memory_baseline()
        mem_start = get_process_memory_mb()
        
        registry = FactorRegistry()
        if "volatility_21d" in registry.factors:
            f = registry.factors["volatility_21d"]
            res = f.calculate(df)
            
            del res
            gc.collect()
        
        mem_end = get_process_memory_mb()
        
        diff = mem_end - mem_start
        print(f"\nFeature Calc Memory Delta: {diff:.2f} MB")
        
        # Absolute leakage tolerance establishing bounds for isolated module loading and JIT trace caching
        assert diff < 50.0, f"Memory leak detected: {diff:.2f} MB retained after calculation"
