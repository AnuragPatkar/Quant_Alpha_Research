"""
Quant Alpha Research - Test Suite
==================================
Comprehensive tests for the quant_alpha library.

Usage:
    python tests/test_data_loading.py
    python tests/test_features.py
    python tests/test_models.py
    python tests/test_backtest.py
    python tests/test_visualization.py
    python tests/test_integration.py
    python tests/run_all_tests.py

Quick Mode (synthetic data):
    python tests/run_all_tests.py --quick
"""

import sys
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Test configuration
QUICK_MODE = True
VERBOSE = True
RANDOM_SEED = 42

__version__ = "1.0.0"