"""
Integration Test Isolation and Fixture Configuration
====================================================
Provides strict namespace isolation for integration tests modifying sys.modules.

Purpose
-------
This module establishes a localized pytest configuration scoped specifically 
to the integration test suite. It enforces rigorous teardown and cache 
invalidation protocols to prevent namespace pollution and MagicMock leakage 
from corrupting subsequent unit tests.

Role in Quantitative Workflow
-----------------------------
Crucial for maintaining test suite stability across complex multi-module 
execution graphs, ensuring that heavy mock objects used in production 
simulations do not persist in the global Python import state.
"""

import sys
import pytest
import importlib
import gc

_MODULES_AT_INTEGRATION_START = None

@pytest.fixture(scope="function", autouse=True)
def isolate_integration_test():
    """
    Autouse fixture enforcing strict module state isolation for integration tests.

    Captures the global `sys.modules` state prior to test execution and 
    aggressively purges target modules during teardown. This prevents 
    mocked artifacts (e.g., MagicMock representations of ML models) from 
    persisting in the namespace and causing fatal deserialization or 
    'Can't pickle' errors in parallel or subsequent test suites.

    Yields:
        None: Yields control to the test function execution.
    """
    global _MODULES_AT_INTEGRATION_START
    _MODULES_AT_INTEGRATION_START = set(sys.modules.keys())
    
    yield
    
    # Teardown phase: Aggressively purge application-specific namespaces to force pristine imports
    modules_to_remove = [
        name for name in list(sys.modules.keys()) 
        if name.startswith("quant_alpha") or name.startswith("scripts") or name == "train_models"
    ]
    
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
        except (KeyError, RuntimeError):
            pass
    
    # Clear Numba's global type registry to prevent 'duplicate registration' collisions 
    # caused by stub module reload loops
    try:
        import numba
        if hasattr(numba, "_internal"):
            try:
                if hasattr(numba._internal, "_dispatcher_cache"):
                    numba._internal._dispatcher_cache.clear()
            except Exception:
                pass
        try:
            from numba.core import types
            if hasattr(types, "_registry"):
                types._registry.clear()
        except Exception:
            pass
    except ImportError:
        pass
    
    # Flush the Python import resolution cache to guarantee subsequent imports 
    # fetch the physical files rather than stale mock references
    try:
        importlib.invalidate_caches()
    except ImportError:
        pass
    
    # Force an immediate garbage collection pass to purge unreferenced MagicMock objects from memory
    gc.collect()
