"""
conftest.py for integration tests
==================================
Provides isolation for integration tests that modify sys.modules.
This file is scoped to tests/integration/ and takes precedence over parent conftest.py
for fixtures defined in this directory.
"""

import sys
import pytest
import importlib
import gc

# Track module state at integration test start
_MODULES_AT_INTEGRATION_START = None

@pytest.fixture(scope="function", autouse=True)
def isolate_integration_test():
    """
    Autouse fixture for integration tests that:
    1. Captures sys.modules before test
    2. Cleans up quant_alpha AND scripts modules after test
    3. Resets Python's import caches
    
    This prevents integration test pollution from affecting unit tests.
    
    CRITICAL: We must remove both quant_alpha.* and scripts.* modules because:
      - Integration tests import scripts inside patched sys.modules
      - scripts modules cache references to MagicMock model classes
      - Even after sys.modules restoration, scripts modules persist with stale references
      - This causes "Can't pickle - it's not the same object" errors in unit tests
    """
    global _MODULES_AT_INTEGRATION_START
    _MODULES_AT_INTEGRATION_START = set(sys.modules.keys())
    
    yield  # Run integration test
    
    # POST-TEST CLEANUP:
    # Remove quant_alpha* and scripts* modules to force fresh imports with real modules
    modules_to_remove = [
        name for name in list(sys.modules.keys()) 
        if name.startswith("quant_alpha") or name.startswith("scripts") or name == "train_models"
    ]
    
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
        except (KeyError, RuntimeError):
            pass  # Module already removed or locked
    
    # Clear Numba's type cache (prevents "duplicate registration" errors)
    # Numba caches types globally, and stub modules can corrupt this cache
    try:
        import numba
        # Try to reset Numba's internal state
        if hasattr(numba, "_internal"):
            try:
                # Clear any cached compilations
                if hasattr(numba._internal, "_dispatcher_cache"):
                    numba._internal._dispatcher_cache.clear()
            except Exception:
                pass
        # Also try types reset
        try:
            from numba.core import types
            if hasattr(types, "_registry"):
                types._registry.clear()
        except Exception:
            pass
    except ImportError:
        pass
    
    # Invalidate importlib caches to ensure next import uses fresh modules
    try:
        importlib.invalidate_caches()
    except ImportError:
        pass
    
    # Force garbage collection to clean up references to mocked modules
    gc.collect()

