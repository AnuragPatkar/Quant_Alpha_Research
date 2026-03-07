"""
UNIT TEST: Utilities
====================
Tests for math_utils and other utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.utils.math_utils import (
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_sortino
)

class TestMathUtils:

    def test_calculate_sharpe_zero_volatility(self):
        """
        Sharpe ratio should handle zero volatility (flat returns) gracefully.
        """
        # Flat returns -> std_dev = 0
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        
        # Should return 0.0 or handle division by zero without crash
        assert sharpe == 0.0, f"Expected 0.0 for zero vol, got {sharpe}"

    def test_calculate_sharpe_normal(self):
        """Standard Sharpe calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sharpe = calculate_sharpe(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_max_drawdown_positive_curve(self):
        """
        If equity curve is strictly increasing, max drawdown should be 0.
        """
        equity = pd.Series(np.linspace(100, 200, 50))
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_calculate_max_drawdown_crash(self):
        """
        Verify drawdown calculation on a crash.
        100 -> 50 (50% drop) -> 75. Max DD should be 0.5.
        """
        equity = pd.Series([100, 80, 50, 60, 75])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.5)

    def test_calculate_sortino_no_downside(self):
        """
        Sortino ratio with no negative returns (infinite theoretically).
        Implementation usually returns Inf or a large number.
        """
        returns = pd.Series([0.01, 0.02, 0.01, 0.03]) # All positive
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        # math_utils.py returns np.inf if downside_deviation == 0
        assert sortino == np.inf or sortino > 1000