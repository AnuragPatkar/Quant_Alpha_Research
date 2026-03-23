"""
Data Ingestion and Feature Engineering Integration Tests
========================================================
Verifies the end-to-end data processing DAG within the training pipeline.

Purpose
-------
This module executes integration tests against the `train_models` data
orchestration layer. It validates the chronological sequence of raw data
ingestion, feature computation, macro-economic enrichment, and the integrity
of the persistent Parquet caching mechanism.

Role in Quantitative Workflow
-----------------------------
Ensures that data transformations applied before machine learning inference
(e.g., sector neutralization, winsorization) are structurally sound and
deterministically cached, preventing data leakage and pipeline regressions.

Dependencies
------------
- **Pytest**: Test execution and fixture management.
- **Unittest.Mock**: Aggressive dependency isolation for deterministic execution.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

@pytest.fixture
def data_flow_context():
    """
    Provisions an aggressively mocked environment for testing the data layer.

    Stubs external C-extension dependencies, ML frameworks, and global configuration 
    state BEFORE importing the target script, thereby isolating I/O logic and 
    preventing test environment pollution.

    Yields:
        tuple: A 2-element tuple containing:
            - module: The dynamically imported `train_models` module.
            - MockConfig: The deterministic mock configuration instance.
    """
    # Provisions a transient filesystem to isolate cache artifacts and prevent race conditions
    test_dir = Path(tempfile.mkdtemp())
    
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
    
    mock_utils = types.ModuleType("quant_alpha.utils")
    mock_utils.setup_logging = MagicMock()
    mock_utils.time_execution = lambda f: f
    mock_utils.load_parquet = pd.read_parquet
    mock_utils.save_parquet = lambda df, path: df.to_parquet(path)
    mock_utils.calculate_returns = MagicMock()
    
    mock_preproc = types.ModuleType("quant_alpha.utils.preprocessing")

    from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb
    
    mock_preproc.WinsorisationScaler = WinsorisationScaler
    mock_preproc.SectorNeutralScaler = SectorNeutralScaler
    mock_preproc.winsorize_clip_nb = winsorize_clip_nb

    # Substitutes the central data layer with deterministic OHLCV arrays
    mock_dm_mod = types.ModuleType("quant_alpha.data.DataManager")
    class MockDataManager:
        def get_master_data(self):
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
    
    mock_reg_mod = types.ModuleType("quant_alpha.features.registry")
    class MockFactorRegistry:
        def compute_all(self, df):
            df["factor_momentum"] = df["close"] * 0.5
            return df
    mock_reg_mod.FactorRegistry = MockFactorRegistry

    # Stubs heavy machine learning dependencies to accelerate execution and prevent C-extension faults
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

    mocks["config.settings"] = mock_settings
    mocks["config"] = mock_settings
    mocks["quant_alpha.utils"] = mock_utils
    mocks["quant_alpha.utils.preprocessing"] = mock_preproc
    mocks["quant_alpha.data.DataManager"] = mock_dm_mod
    mocks["quant_alpha.features.registry"] = mock_reg_mod

    # Enforces strict package hierarchy resolution to bypass ImportErrors on relative calls
    qa = types.ModuleType("quant_alpha")
    qa.__path__ = []
    mocks["quant_alpha"] = qa
    
    qa_data = types.ModuleType("quant_alpha.data")
    qa_data.__path__ = []
    qa_data.DataManager = mock_dm_mod 
    mocks["quant_alpha.data"] = qa_data

    with patch.dict(sys.modules, mocks):
        if "scripts.train_models" in sys.modules:
            tm = importlib.reload(sys.modules["scripts.train_models"])
        else:
            tm = importlib.import_module("scripts.train_models")
        
        yield tm, mock_cfg_instance

    shutil.rmtree(test_dir)


class TestDataFlow:
    
    def test_dataset_construction_and_caching(self, data_flow_context):
        """
        Validates the complete dataset hydration and caching lifecycle.

        Verifies that the `load_and_build_full_dataset` pipeline accurately requests 
        raw data, computes target features, and securely serializes the result to 
        a Parquet cache. Subsequently confirms that repeated requests bypass computation 
        and load directly from disk.

        Args:
            data_flow_context (tuple): Injected isolated execution context.

        Returns:
            None
        """
        train_models, mock_config = data_flow_context

        # Initiates a cold start to bypass internal cache logic and force computation
        df_built = train_models.load_and_build_full_dataset(force_rebuild=True)
        
        assert not df_built.empty
        assert len(df_built) == 20
        assert "factor_momentum" in df_built.columns, "Factors should be computed"
        assert "raw_ret_5d" in df_built.columns, "Target column should be added"
        
        cache_file = mock_config.CACHE_DIR / "master_data_with_factors.parquet"
        assert cache_file.exists(), "Cache file was not created"
        
        # Overrides the data layer to trigger an exception, ensuring the cache intercepts the data request
        with patch("quant_alpha.data.DataManager.DataManager") as mock_dm_class:
            mock_dm_class.side_effect = Exception("Should not be called! Data should come from cache.")
            
            df_cached = train_models.load_and_build_full_dataset(force_rebuild=False)
            
            pd.testing.assert_frame_equal(df_built, df_cached)

    def test_macro_feature_enrichment(self, data_flow_context):
        """
        Verifies the structural alignment and calculation of macroeconomic indicators.

        Ensures that rolling momentum sequences and structural volatility proxies 
        are correctly hydrated into the panel dataset.

        Args:
            data_flow_context (tuple): Injected isolated execution context.

        Returns:
            None
        """
        train_models, _ = data_flow_context

        dates = pd.date_range("2020-01-01", periods=30)
        df = pd.DataFrame({
            "date": dates,
            "ticker": ["A"] * 30,
            "close": np.linspace(100, 130, 30),
            "open": np.linspace(100, 130, 30)
        })
        
        df_enriched = train_models.add_macro_features(df)
        
        expected_cols = ["macro_mom_5d", "macro_mom_21d", "macro_vix_proxy", "macro_trend_200d"]
        for col in expected_cols:
            assert col in df_enriched.columns, f"Missing macro column: {col}"
            
        last_val = df_enriched["macro_mom_5d"].iloc[-1]
        assert last_val > 0, "Macro momentum should be positive for uptrend"

    def test_target_construction(self, data_flow_context):
        """
        Validates the accurate derivation of sector-neutral forward return targets.

        Args:
            data_flow_context (tuple): Injected isolated execution context.

        Returns:
            None
        """
        train_models, _ = data_flow_context

        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "ticker": ["A", "B", "A", "B"],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "raw_ret_5d": [0.05, 0.03, 0.02, 0.04]
        })
        
        res = train_models.build_target(df)
        
        assert "target" in res.columns
        
        assert res.loc[0, "target"] == pytest.approx(0.01)
        
        assert res.loc[1, "target"] == pytest.approx(-0.01)

    def test_winsorization_scaler(self, data_flow_context):
        """
        Evaluates the statistical boundaries of the WinsorisationScaler logic.

        Verifies that extreme outliers are effectively clamped to their respective 
        quantile thresholds to prevent gradient explosion during model training.

        Args:
            data_flow_context (tuple): Injected isolated execution context.

        Returns:
            None
        """
        train_models, _ = data_flow_context

        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01"] * 100),
            "val": np.random.normal(0, 1, 100)
        })
        df.loc[0, "val"] = 1000.0
        df.loc[1, "val"] = -1000.0
        
        scaler = train_models.WinsorisationScaler(clip_pct=0.05)
        scaler.fit(df, ["val"])
        res = scaler.transform(df, ["val"])
        
        assert res["val"].max() < 100.0
        assert res["val"].min() > -100.0
        
        upper_bound = df["val"].quantile(0.95)
        assert res["val"].max() == pytest.approx(upper_bound)
