"""
INTEGRATION TEST: Data Flow & Feature Engineering Pipeline
==========================================================
Verifies the end-to-end data journey in scripts/train_models.py:
1. Ingestion: Loading raw data via DataManager (mocked).
2. Construction: Building the master dataset and computing factors.
3. Enrichment: Adding macro/fundamental features.
4. Caching: Verifying parquet cache creation, hashing, and reloading.

This test uses aggressive mocking to isolate the data logic from 
ML dependencies and production configuration.
"""

import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import types
import importlib
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
# Mocking Infrastructure (Pre-import)
# ---------------------------------------------------------------------------
# We must mock the configuration and heavy dependencies BEFORE importing 
# train_models.py. We use a fixture to contain this side-effect.

@pytest.fixture
def data_flow_context():
    """
    Sets up a mocked environment and imports train_models dynamically.
    Yields (train_models_module, mock_config_instance).
    """
    # 1. Create a temporary directory for this test session
    test_dir = Path(tempfile.mkdtemp())
    
    # 2. Define a Mock Config that points to the temp dir
    mock_settings = types.ModuleType("config.settings")
    
    class MockConfig:
        def __init__(self):
            self.DATA_DIR = test_dir / "data"
            self.CACHE_DIR = self.DATA_DIR / "cache"
            self.PRICES_DIR = self.DATA_DIR / "prices"
            self.LOG_DIR = self.DATA_DIR / "logs"
            self.MODELS_DIR = self.DATA_DIR / "models"
            self.RESULTS_DIR = self.DATA_DIR / "results"
            self.ALTERNATIVE_DIR = self.DATA_DIR / "alternative"
            
            # Create dirs
            for p in [self.DATA_DIR, self.CACHE_DIR, self.PRICES_DIR, 
                      self.LOG_DIR, self.MODELS_DIR, self.RESULTS_DIR, 
                      self.ALTERNATIVE_DIR]:
                p.mkdir(parents=True, exist_ok=True)

            # Settings
            self.BACKTEST_START_DATE = "2020-01-01"
            self.BACKTEST_END_DATE = "2021-01-01"
            self.LOG_FILE = self.LOG_DIR / "test.log"
            self.LOG_LEVEL = "INFO"
            self.LOG_FORMAT = "%(message)s"
            self.LOG_DATE_FORMAT = "%Y-%m-%d"
            self.INITIAL_CAPITAL = 100000
            self.TRANSACTION_COST_BPS = 10
            self.RISK_FREE_RATE = 0.0
            self.OPT_LOOKBACK_DAYS = 20
            self.OPT_RISK_AVERSION = 1.0
            self.OPT_KELLY_FRACTION = 1.0
            self.TRAILING_STOP_PCT = 0.1
            self.PROD_IC_THRESHOLD = 0.01
            self.PROD_IC_TSTAT = 2.0
            self.MIN_OOS_IC_THRESHOLD = 0.005
            self.MIN_OOS_IC_TSTAT = 1.5
            self.PROD_MODEL_MIN_DATE = "2020-01-01"
            self.ENV = "test"

    mock_cfg_instance = MockConfig()
    mock_settings.Config = MockConfig
    mock_settings.config = mock_cfg_instance
    
    # 3. Mock quant_alpha.utils
    mock_utils = types.ModuleType("quant_alpha.utils")
    mock_utils.setup_logging = MagicMock()
    mock_utils.time_execution = lambda f: f  # Decorator pass-through
    mock_utils.load_parquet = pd.read_parquet
    mock_utils.save_parquet = lambda df, path: df.to_parquet(path)
    mock_utils.calculate_returns = MagicMock()
    
    # Mock Preprocessing
    mock_preproc = types.ModuleType("quant_alpha.utils.preprocessing")
    # Use REAL classes for logic testing
    from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb
    
    mock_preproc.WinsorisationScaler = WinsorisationScaler
    mock_preproc.SectorNeutralScaler = SectorNeutralScaler
    mock_preproc.winsorize_clip_nb = winsorize_clip_nb

    # 4. Mock DataManager (The Source of Truth)
    mock_dm_mod = types.ModuleType("quant_alpha.data.DataManager")
    class MockDataManager:
        def get_master_data(self):
            # Return dummy OHLCV data
            dates = pd.date_range("2020-01-01", periods=20)
            df = pd.DataFrame({
                "date": dates,
                "ticker": ["AAPL"] * 20,
                "close": np.linspace(100, 120, 20),
                "open": np.linspace(100, 120, 20),
                "high": np.linspace(101, 121, 20),
                "low": np.linspace(99, 119, 20),
                "volume": [1000] * 20
            })
            return df
    mock_dm_mod.DataManager = MockDataManager
    

    # 5. Mock FactorRegistry (The Feature Engineer)
    mock_reg_mod = types.ModuleType("quant_alpha.features.registry")
    class MockFactorRegistry:
        def compute_all(self, df):
            # Simulate computing a factor
            df["factor_momentum"] = df["close"] * 0.5
            return df
    mock_reg_mod.FactorRegistry = MockFactorRegistry

    # 6. Mock ML Models & Heavy Libs (Prevent ImportErrors)
    modules_to_mock = [
        "quant_alpha.models.trainer",
        "quant_alpha.models.lightgbm_model",
        "quant_alpha.models.xgboost_model",
        "quant_alpha.models.catboost_model",
        "quant_alpha.models.feature_selector",
        "quant_alpha.backtest.engine",
        "quant_alpha.backtest.metrics",
        "quant_alpha.backtest.attribution",
        "quant_alpha.optimization.allocator",
        "quant_alpha.visualization",
        # Feature submodules
        "quant_alpha.features.technical.momentum",
        "quant_alpha.features.technical.volatility",
        "quant_alpha.features.technical.volume",
        "quant_alpha.features.technical.mean_reversion",
        "quant_alpha.features.fundamental.value",
        "quant_alpha.features.fundamental.quality",
        "quant_alpha.features.fundamental.growth",
        "quant_alpha.features.fundamental.financial_health",
        "quant_alpha.features.earnings.surprises",
        "quant_alpha.features.earnings.estimates",
        "quant_alpha.features.earnings.revisions",
        "quant_alpha.features.alternative.macro",
        "quant_alpha.features.alternative.sentiment",
        "quant_alpha.features.alternative.inflation",
        "quant_alpha.features.composite.macro_adjusted",
        "quant_alpha.features.composite.system_health",
        "quant_alpha.features.composite.smart_signals",
    ]

    mocks = {}
    for mod_name in modules_to_mock:
        m = types.ModuleType(mod_name)
        # Add dummy classes to satisfy imports
        if "lightgbm_model" in mod_name: m.LightGBMModel = MagicMock()
        if "xgboost_model" in mod_name: m.XGBoostModel = MagicMock()
        if "catboost_model" in mod_name: m.CatBoostModel = MagicMock()
        if "trainer" in mod_name: m.WalkForwardTrainer = MagicMock()
        if "feature_selector" in mod_name: m.FeatureSelector = MagicMock()
        if "engine" in mod_name: m.BacktestEngine = MagicMock()
        if "metrics" in mod_name: m.print_metrics_report = MagicMock()
        if "attribution" in mod_name: 
            m.SimpleAttribution = MagicMock()
            m.FactorAttribution = MagicMock()
        if "allocator" in mod_name: m.PortfolioAllocator = MagicMock()
        if "visualization" in mod_name:
            m.plot_equity_curve = MagicMock()
            m.plot_drawdown = MagicMock()
            m.plot_monthly_heatmap = MagicMock()
            m.plot_ic_time_series = MagicMock()
            m.generate_tearsheet = MagicMock()
            
        mocks[mod_name] = m

    # Add core mocks
    mocks["config.settings"] = mock_settings
    mocks["config"] = mock_settings
    mocks["quant_alpha.utils"] = mock_utils
    mocks["quant_alpha.utils.preprocessing"] = mock_preproc
    mocks["quant_alpha.data.DataManager"] = mock_dm_mod
    mocks["quant_alpha.features.registry"] = mock_reg_mod

    # Ensure parent packages exist in mocks to prevent import errors
    # This fixes the "quant_alpha is not a package" issue by ensuring it has __path__
    qa = types.ModuleType("quant_alpha")
    qa.__path__ = []
    mocks["quant_alpha"] = qa
    
    qa_data = types.ModuleType("quant_alpha.data")
    qa_data.__path__ = []
    qa_data.DataManager = mock_dm_mod # Link for attribute lookup
    mocks["quant_alpha.data"] = qa_data

    # Apply patches
    with patch.dict(sys.modules, mocks):
        # Import script under test
        if "scripts.train_models" in sys.modules:
            tm = importlib.reload(sys.modules["scripts.train_models"])
        else:
            tm = importlib.import_module("scripts.train_models")
        
        yield tm, mock_cfg_instance

    # Cleanup temp dir
    shutil.rmtree(test_dir)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestDataFlow:
    
    def test_dataset_construction_and_caching(self, data_flow_context):
        """
        Verify load_and_build_full_dataset:
        1. Calls DataManager (mocked) to get raw data.
        2. Computes factors via Registry (mocked).
        3. Saves the result to parquet cache.
        4. Loads from cache on subsequent calls.
        """
        train_models, mock_config = data_flow_context

        # --- Run 1: Build from "Raw" ---
        # Force rebuild to ignore any existing cache logic
        df_built = train_models.load_and_build_full_dataset(force_rebuild=True)
        
        # Assertions on built data
        assert not df_built.empty
        assert len(df_built) == 20
        assert "factor_momentum" in df_built.columns, "Factors should be computed"
        assert "raw_ret_5d" in df_built.columns, "Target column should be added"
        
        # Verify Cache File Created
        cache_file = mock_config.CACHE_DIR / "master_data_with_factors.parquet"
        assert cache_file.exists(), "Cache file was not created"
        
        # --- Run 2: Load from Cache ---
        # We mock DataManager to raise an error. If the code tries to rebuild
        # instead of loading from cache, the test will fail.
        with patch("quant_alpha.data.DataManager.DataManager") as mock_dm_class:
            mock_dm_class.side_effect = Exception("Should not be called! Data should come from cache.")
            
            df_cached = train_models.load_and_build_full_dataset(force_rebuild=False)
            
            # Should be identical to built data
            pd.testing.assert_frame_equal(df_built, df_cached)

    def test_macro_feature_enrichment(self, data_flow_context):
        """
        Verify add_macro_features correctly calculates rolling macro stats.
        """
        train_models, _ = data_flow_context

        # Create dummy data with price history
        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({
            "date": dates,
            "ticker": ["A"] * 30,
            "close": np.linspace(100, 130, 30), # Uptrend
            "open": np.linspace(100, 130, 30)
        })
        
        # Run enrichment
        df_enriched = train_models.add_macro_features(df)
        
        # Check for expected columns
        expected_cols = ["macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d"]
        for col in expected_cols:
            assert col in df_enriched.columns, f"Missing macro column: {col}"
            
        # Check logic: macro_mom_5d should be positive (uptrend)
        # Note: first few rows will be NaN due to rolling window
        last_val = df_enriched["macro_mom_5d"].iloc[-1]
        assert last_val > 0, "Macro momentum should be positive for uptrend"

    def test_target_construction(self, data_flow_context):
        """
        Verify build_target correctly computes sector-neutral targets.
        """
        train_models, _ = data_flow_context

        # Create 2 tickers in same sector, one outperforming the other
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "ticker": ["A", "B", "A", "B"],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "raw_ret_5d": [0.05, 0.03, 0.02, 0.04] 
            # Day 1: Mean=0.04. A=0.05(+0.01), B=0.03(-0.01)
        })
        
        res = train_models.build_target(df)
        
        assert "target" in res.columns
        
        # Check Day 1 (Index 0 and 1)
        # A (0.05) vs Mean (0.04) -> Target should be 0.01
        assert res.loc[0, "target"] == pytest.approx(0.01)
        
        # B (0.03) vs Mean (0.04) -> Target should be -0.01
        assert res.loc[1, "target"] == pytest.approx(-0.01)

    def test_winsorization_scaler(self, data_flow_context):
        """
        Verify WinsorisationScaler clips outliers correctly.
        """
        train_models, _ = data_flow_context

        # Create data with outliers
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"] * 100),
            "val": np.random.normal(0, 1, 100)
        })
        # Add extreme outliers
        df.loc[0, "val"] = 1000.0
        df.loc[1, "val"] = -1000.0
        
        scaler = train_models.WinsorisationScaler(clip_pct=0.05)
        scaler.fit(df, ["val"])
        res = scaler.transform(df, ["val"])
        
        # Max value should be much less than 1000
        assert res["val"].max() < 100.0
        assert res["val"].min() > -100.0
        
        # Check that it actually clipped to the quantile
        upper_bound = df["val"].quantile(0.95)
        assert res["val"].max() == pytest.approx(upper_bound)
