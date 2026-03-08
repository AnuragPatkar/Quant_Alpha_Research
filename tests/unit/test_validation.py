"""
UNIT TEST: Factor Validation Logic
==================================
Tests the FactorValidator class in scripts/validate_factors.py.
Verifies IC calculation, t-stats, and autocorrelation logic.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import types
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Mock config.settings to prevent Config() validation error during import
if "config.settings" not in sys.modules:
    mock_settings = types.ModuleType("config.settings")
    mock_config = MagicMock()
    mock_config.RESULTS_DIR = Path("results")
    mock_settings.config = mock_config
    sys.modules["config.settings"] = mock_settings

# Mock dependencies for validate_factors
if "quant_alpha.utils" not in sys.modules:
    mock_utils = MagicMock()
    # Fix: decorators must return the function, not a Mock
    mock_utils.time_execution = lambda f: f
    # Handle @retry(params) -> returns decorator -> returns function
    mock_utils.retry = lambda *args, **kwargs: (lambda f: f)
    sys.modules["quant_alpha.utils"] = mock_utils
if "scripts.train_models" not in sys.modules:
    sys.modules["scripts.train_models"] = MagicMock()

from scripts import validate_factors

class TestFactorValidator:
    
    @pytest.fixture
    def dummy_data(self):
        """Create synthetic data with a known good factor."""
        # Need multiple tickers for cross-sectional IC calculation
        dates = pd.date_range("2023-01-01", periods=50)
        
        # Ticker A: Positive signal, Positive return
        df_a = pd.DataFrame({
            "date": dates,
            "ticker": "A",
            "good_factor": 1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": 0.05
        })
        # Ticker B: Negative signal, Negative return (Perfect Rank Correlation with A)
        df_b = pd.DataFrame({
            "date": dates,
            "ticker": "B",
            "good_factor": -1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": -0.05
        })
        
        return pd.concat([df_a, df_b]).sort_values("date").reset_index(drop=True)

    def test_identify_factors(self, dummy_data):
        """Should ignore metadata columns and pick numeric factors."""
        validator = validate_factors.FactorValidator(dummy_data)
        assert "good_factor" in validator.factors
        assert "noise_factor" in validator.factors
        assert "date" not in validator.factors
        assert "ticker" not in validator.factors
        assert "raw_ret_5d" not in validator.factors

    def test_ic_calculation(self, dummy_data):
        """Verify IC calculation for a perfect factor."""
        validator = validate_factors.FactorValidator(dummy_data)
        validator.target = "raw_ret_5d"  # Explicitly set target
        stats = validator.compute_ic_stats()
        
        good = stats.loc["good_factor"]
        assert good["ic_mean"] == pytest.approx(1.0, abs=0.05)
        assert good["t_stat"] > 10.0 # Extremely significant
        
        noise = stats.loc["noise_factor"]
        assert abs(noise["ic_mean"]) < 0.3 # Should be low

    def test_autocorrelation(self, dummy_data):
        """Verify AR(1) calculation."""
        # Create a factor with high autocorrelation (trend)
        dummy_data["trend"] = np.linspace(0, 10, len(dummy_data))
        validator = validate_factors.FactorValidator(dummy_data)
        ac = validator.compute_autocorrelation()
        
        assert ac.loc["trend", "autocorr"] > 0.9
        assert abs(ac.loc["noise_factor", "autocorr"]) < 0.5