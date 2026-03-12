import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
"""
conftest.py
===========
Global pytest configuration and fixtures.

# 1. Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
This file is automatically discovered by pytest and is used to define
fixtures, hooks, and plugins that are available to all tests in the project.
"""

# 2. Environment Variables (CPU Throttling for Tests)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

# ---------------------------------------------------------------------------
# Global state tracking for test pollution prevention
# ---------------------------------------------------------------------------
_ORIGINAL_SYS_MODULES = None
_MODULES_BEFORE_INTEGRATION = None

def pytest_sessionstart(session):
    """
    Called once at the beginning of the entire test session.
    This is the ideal place to handle global state initialization for
    C-extension libraries that can cause "duplicate registration" errors
    when modules are re-imported across different test files.
    
    Also captures initial sys.modules state for test pollution prevention.
    """
    global _ORIGINAL_SYS_MODULES
    
    # 1. Initialize Numba and PyArrow to prevent duplicate registration errors
    try:
        # For PyArrow: Prevents `ArrowKeyError: pandas.period already defined`
        import pandas.core.arrays.arrow.extension_types
    except ImportError:
        pass
    try:
        # For Numba: Prevents `duplicate registration for <class ...>` errors
        import numba
        # By also importing numpy.polynomial, we encourage Numba to register
        # its corresponding types once, globally, at the start of the session.
        import numpy.polynomial
    except ImportError:
        pass
    
    # 2. Capture sys.modules state for test pollution prevention
    if _ORIGINAL_SYS_MODULES is None:
        _ORIGINAL_SYS_MODULES = set(sys.modules.keys())

# 3. Shared Fixtures
@pytest.fixture(scope="session")
def synthetic_data():
    """
    Creates a synthetic panel dataset for testing models.
    Returns: (df, features, target_col)
    """
    np.random.seed(42)
    n_tickers = 10
    n_days = 100
    n_features = 10
    
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    rows = []
    for ticker in tickers:
        sector = np.random.choice(["Tech", "Finance", "Health"])
        industry = np.random.choice(["Soft", "Bank", "Pharma"])
        
        for date in dates:
            row = {
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "target": np.random.normal(0, 0.02)
            }
            # Add numeric features
            for i in range(n_features):
                row[f"f_{i:03d}"] = np.random.normal()
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # Set dtypes
    df["sector"] = df["sector"].astype("category")
    df["industry"] = df["industry"].astype("category")
    
    features = [c for c in df.columns if c.startswith("f_")] + ["sector", "industry"]
    return df, features, "target"

@pytest.fixture(scope="session")
def sample_covariance_matrix():
    """Creates a dummy covariance matrix for optimization tests."""
    tickers = [f"TICK{i:03d}" for i in range(5)]
    cov = pd.DataFrame(
        np.identity(5) * 0.0004, # Low variance
        index=tickers,
        columns=tickers
    )
    return cov, tickers

@pytest.fixture(scope="session")
def sample_expected_returns(sample_covariance_matrix):
    """Creates dummy expected returns."""
    _, tickers = sample_covariance_matrix
    return {t: 0.01 * (i + 1) for i, t in enumerate(tickers)}

# Custom Objective for Models (Shared)
def weighted_symmetric_mae(y_true, y_pred):
    residuals = y_true - y_pred
    weights = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad = -weights * np.tanh(residuals)
    hess = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess

# Inject into main for pickle compatibility during tests
try:
    sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae
except Exception:
    pass

# ---------------------------------------------------------------------------
# 4. Test Pollution Prevention (autouse fixture)
# ---------------------------------------------------------------------------
# This prevents sys.modules pollution from integration tests (especially test_production.py)
# from affecting subsequent unit tests.

def pytest_runtest_setup(item):
    """Hook that captures sys.modules state before running ANY integration test."""
    global _MODULES_BEFORE_INTEGRATION
    # Capture state before ANY integration test (not just production and deployment)
    if "tests/integration" in str(item.fspath) or "tests\\integration" in str(item.fspath):
        _MODULES_BEFORE_INTEGRATION = set(sys.modules.keys())

def pytest_runtest_teardown(item):
    """
    Hook that aggressively restores sys.modules after ANY integration test.
    This ensures unit tests are not polluted by patched/mocked modules.
    
    CRITICAL: Removes scripts.* modules because they cache references to MagicMock
    objects that were loaded while sys.modules was patched.
    
    Strategy:
      1. Remove modules added by the integration test
      2. Force delete quant_alpha.* and scripts.* modules
      3. Reset Numba's type registry
      4. Invalidate all caches
      5. Garbage collect
    """
    global _MODULES_BEFORE_INTEGRATION
    
    # Clean up after ANY integration test, not just specific ones
    if "tests/integration" not in str(item.fspath) and "tests\\integration" not in str(item.fspath):
        return
    
    # 1. Remove any modules added by this integration test
    if _MODULES_BEFORE_INTEGRATION is not None:
        current_modules = set(sys.modules.keys())
        modules_to_remove = current_modules - _MODULES_BEFORE_INTEGRATION
        
        for module_name in list(modules_to_remove):  # list() to avoid "dict changed size" error
            try:
                del sys.modules[module_name]
            except (KeyError, RuntimeError):
                pass
    
    # 2. Force removal of quant_alpha* and scripts* modules
    modules_to_reload = [name for name in list(sys.modules.keys()) if name.startswith(("quant_alpha", "scripts"))]
    for module_name in modules_to_reload:
        try:
            del sys.modules[module_name]
        except (KeyError, RuntimeError):
            pass
    
    # 3. Forcefully clear Numba's type registry to prevent "duplicate registration" errors
    try:
        import numba
        from numba.core import types
        # Clear Numba's registry of registered types
        if hasattr(types, '_registry'):
            types._registry.clear()
        # Also try the alternate path where Numba caches JIT functions
        if hasattr(numba, '_internal') and hasattr(numba._internal, 'registry'):
            numba._internal.registry.clear()
    except (ImportError, AttributeError):
        pass
    
    # 4. Clear importlib caches and metafinder-cached modules
    try:
        from importlib import invalidate_caches
        invalidate_caches()
    except ImportError:
        pass
    
    try:
        import importlib
        if hasattr(importlib, "_bootstrap_external"):
            # Clear the find_spec cache
            if hasattr(importlib._bootstrap_external, '_get_cached'):
                try:
                    importlib._bootstrap_external._get_cached.cache_clear()
                except AttributeError:
                    pass
    except (ImportError, AttributeError):
        pass
    
    # 5. Force garbage collection to remove references to mocked objects
    import gc
    gc.collect()

@pytest.fixture(autouse=True)
def cleanup_modules_after_test():
    """
    Autouse fixture that cleans up sys.modules after each test.
    
    Prevents test pollution where mocked/patched modules from integration tests
    (especially test_production.py which uses patch.dict(sys.modules)) persist
    and corrupt subsequent unit tests with:
      - KeyError: missing DataFrame columns ('date', 'ticker', 'reason')
      - UnboundLocalError: '_pickle' (model classes replaced with MagicMock)
      - Import errors from orphaned stub modules
    
    This fixture works alongside the pytest_runtest_teardown hook to provide
    defense-in-depth: the hook handles test_production.py specifically, and this
    fixture provides a fallback for any test that leaves sys.modules dirty.
    """
    yield  # Run the test first
    
    # Post-test cleanup: Clear any mocked model classes that might interfere with imports
    # This prevents UnboundLocalError: '_pickle' issues in test_models.py
    try:
        # If any model modules are MagicMock, reload the real ones
        for model_cls_name in ["LightGBMModel", "XGBoostModel", "CatBoostModel"]:
            for module_path in [
                "quant_alpha.models.lightgbm_model",
                "quant_alpha.models.xgboost_model", 
                "quant_alpha.models.catboost_model"
            ]:
                if module_path in sys.modules:
                    mod = sys.modules[module_path]
                    # Check if the model class was replaced with a MagicMock
                    if hasattr(mod, model_cls_name):
                        attr = getattr(mod, model_cls_name)
                        # If it's a MagicMock (not a real class), remove the module to force reimport
                        if type(attr).__name__ == "MagicMock":
                            try:
                                del sys.modules[module_path]
                            except KeyError:
                                pass
    except Exception:
        pass  # Silently ignore errors during cleanup