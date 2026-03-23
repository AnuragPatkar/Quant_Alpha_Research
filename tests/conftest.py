"""
Global Pytest Configuration and Fixtures
========================================
Centralized test suite orchestration and mock data provisioning.

Purpose
-------
This module serves as the foundational testing configuration for the Quant Alpha
platform. It provisions deterministic synthetic datasets, manages C-extension 
memory states (e.g., Numba, PyArrow), and rigorously enforces test isolation 
to prevent namespace pollution across integration and unit tests.

Role in Quantitative Workflow
-----------------------------
Automatically discovered by pytest. Ensures that all localized tests execute 
within a stable, reproducible environment without requiring redundant setup logic.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Binds deterministic thread limits for numerical backends to prevent test-suite CPU thrashing
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

# Global state tracking for strict test isolation and namespace pollution prevention
_ORIGINAL_SYS_MODULES = None
_MODULES_BEFORE_INTEGRATION = None

def pytest_sessionstart(session):
    """
    Executes global initialization prior to test suite execution.

    Pre-loads specific C-extension libraries to prevent 'duplicate registration' 
    faults during parallel test discovery, and captures the baseline module state 
    to enforce strict isolation.

    Args:
        session (pytest.Session): The pytest session object.

    Returns:
        None
    """
    global _ORIGINAL_SYS_MODULES
    
    try:
        # Pre-allocates PyArrow extension types to prevent ArrowKeyError on pandas.period
        import pandas.core.arrays.arrow.extension_types
    except ImportError:
        pass
    try:
        # Pre-initializes Numba and numpy.polynomial to prevent duplicate registration faults
        import numba
        import numpy.polynomial
    except ImportError:
        pass
    
    if _ORIGINAL_SYS_MODULES is None:
        _ORIGINAL_SYS_MODULES = set(sys.modules.keys())

@pytest.fixture(scope="session")
def synthetic_data():
    """
    Generates a deterministic, synthetic panel dataset for model validation.

    Args:
        None

    Returns:
        tuple: A 3-element tuple containing:
            - pd.DataFrame: The generated synthetic OHLCV and feature dataset.
            - list[str]: The list of feature column names.
            - str: The target column identifier.
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
            for i in range(n_features):
                row[f"f_{i:03d}"] = np.random.normal()
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    df["sector"] = df["sector"].astype("category")
    df["industry"] = df["industry"].astype("category")
    
    features = [c for c in df.columns if c.startswith("f_")] + ["sector", "industry"]
    return df, features, "target"

@pytest.fixture(scope="session")
def sample_covariance_matrix():
    """
    Provisions a synthetic, positive-definite covariance matrix.

    Args:
        None

    Returns:
        tuple: A 2-element tuple containing:
            - pd.DataFrame: The N x N identity covariance matrix scaled by low variance.
            - list[str]: The ordered list of ticker symbols.
    """
    tickers = [f"TICK{i:03d}" for i in range(5)]
    cov = pd.DataFrame(
        np.identity(5) * 0.0004,
        index=tickers,
        columns=tickers
    )
    return cov, tickers

@pytest.fixture(scope="session")
def sample_expected_returns(sample_covariance_matrix):
    """
    Generates a deterministic expected returns vector aligned with the covariance matrix.

    Args:
        sample_covariance_matrix (tuple): The covariance fixture output.

    Returns:
        dict[str, float]: A mapping of ticker symbols to expected annualized returns.
    """
    _, tickers = sample_covariance_matrix
    return {t: 0.01 * (i + 1) for i, t in enumerate(tickers)}

def weighted_symmetric_mae(y_true, y_pred):
    """
    Calculates an asymmetric penalty for directional errors during optimization.

    Exposed globally to ensure deterministic resolution during joblib unpickling 
    in isolated test environments.

    Args:
        y_true (np.ndarray): Empirical target variables mappings.
        y_pred (np.ndarray): Current predictive node output sequences.

    Returns:
        tuple[np.ndarray, np.ndarray]: Derived Gradient and Hessian bounds required by GBDTs.
    """
    residuals = y_true - y_pred
    weights = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad = -weights * np.tanh(residuals)
    hess = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess

# Injects into main namespace to guarantee pickle compatibility during unpickling procedures
try:
    sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae
except Exception:
    pass

def pytest_runtest_setup(item):
    """
    Captures the global module namespace state prior to executing integration tests.

    Args:
        item (pytest.Item): The current test item being executed.

    Returns:
        None
    """
    global _MODULES_BEFORE_INTEGRATION
    if "tests/integration" in str(item.fspath) or "tests\\integration" in str(item.fspath):
        _MODULES_BEFORE_INTEGRATION = set(sys.modules.keys())

def pytest_runtest_teardown(item):
    """
    Aggressively restores the global module namespace after integration tests.

    Forces garbage collection, invalidates import caches, and resets Numba's 
    type registry to guarantee subsequent unit tests execute in a pristine, 
    unpolluted state.

    Args:
        item (pytest.Item): The current test item that finished execution.

    Returns:
        None
    """
    global _MODULES_BEFORE_INTEGRATION
    
    if "tests/integration" not in str(item.fspath) and "tests\\integration" not in str(item.fspath):
        return
    
    if _MODULES_BEFORE_INTEGRATION is not None:
        current_modules = set(sys.modules.keys())
        modules_to_remove = current_modules - _MODULES_BEFORE_INTEGRATION
        
        # Coerces to list to avoid 'dictionary changed size during iteration' exceptions
        for module_name in list(modules_to_remove):
            try:
                del sys.modules[module_name]
            except (KeyError, RuntimeError):
                pass
    
    modules_to_reload = [name for name in list(sys.modules.keys()) if name.startswith(("quant_alpha", "scripts"))]
    for module_name in modules_to_reload:
        try:
            del sys.modules[module_name]
        except (KeyError, RuntimeError):
            pass
    
    try:
        import numba
        from numba.core import types
        if hasattr(types, '_registry'):
            types._registry.clear()
        if hasattr(numba, '_internal') and hasattr(numba._internal, 'registry'):
            numba._internal.registry.clear()
    except (ImportError, AttributeError):
        pass
    
    try:
        from importlib import invalidate_caches
        invalidate_caches()
    except ImportError:
        pass
    
    try:
        import importlib
        if hasattr(importlib, "_bootstrap_external"):
            if hasattr(importlib._bootstrap_external, '_get_cached'):
                try:
                    importlib._bootstrap_external._get_cached.cache_clear()
                except AttributeError:
                    pass
    except (ImportError, AttributeError):
        pass
    
    import gc
    gc.collect()

@pytest.fixture(autouse=True)
def cleanup_modules_after_test():
    """
    Autouse fixture executing defensive namespace cleanup after every test.

    Acts as a fail-safe mechanism against persistent mocked artifacts (e.g., MagicMock)
    corrupting the sys.modules cache, specifically targeting model class wrappers.

    Args:
        None

    Returns:
        Generator: Yields control to the test execution, then performs teardown.
    """
    yield
    
    try:
        for model_cls_name in ["LightGBMModel", "XGBoostModel", "CatBoostModel"]:
            for module_path in [
                "quant_alpha.models.lightgbm_model",
                "quant_alpha.models.xgboost_model", 
                "quant_alpha.models.catboost_model"
            ]:
                if module_path in sys.modules:
                    mod = sys.modules[module_path]
                    if hasattr(mod, model_cls_name):
                        attr = getattr(mod, model_cls_name)
                        if type(attr).__name__ == "MagicMock":
                            try:
                                del sys.modules[module_path]
                            except KeyError:
                                pass
    except Exception:
        pass