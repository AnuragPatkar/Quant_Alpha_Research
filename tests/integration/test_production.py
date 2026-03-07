## 📁 tests/integration/test_production.py


"""
INTEGRATION TEST: Production Readiness
======================================
Verifies the end-to-end production workflow:
1. Health Check (deploy_model.py)
2. Inference (generate_predictions.py)
3. Portfolio Optimization (optimize_portfolio.py)

Ensures that the scripts interact correctly with the filesystem and configuration,
and that the critical path for daily trading is functional.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import types
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, date

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
# the scripts to prevent them from crashing on missing paths or libs.

def _stub_module(module_name, **attrs):
    """
    Robustly stub a module in sys.modules, ensuring parent packages exist.
    Handles 'from X.Y import Z' and 'import X.Y' patterns.
    """
    parts = module_name.split(".")
    
    # Ensure all parent packages exist
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            m = types.ModuleType(parent_name)
            m.__path__ = [] # Mark as package
            sys.modules[parent_name] = m
            
    # Create or update the target module
    if module_name in sys.modules:
        m = sys.modules[module_name]
    else:
        m = types.ModuleType(module_name)
        # If it's likely a package (has submodules implied), give it a path
        if "." not in module_name or "quant_alpha" in module_name:
             m.__path__ = []
        sys.modules[module_name] = m
    
    # Set attributes
    for k, v in attrs.items():
        setattr(m, k, v)
        
    # Link to parent
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], m)
        
    return m

# 1. Stub Config
_stub_module("config")

# Create a dummy config object to satisfy import-time requirements
# optimize_portfolio.py accesses config.RESULTS_DIR and config.CACHE_DIR at module level
class DummyConfig:
    RESULTS_DIR = Path("results")
    CACHE_DIR = Path("cache")
    DATA_DIR = Path("data")
    PRICES_DIR = Path("prices")
    MODELS_DIR = Path("models")
    LOG_DIR = Path("logs")
    BACKTEST_START_DATE = "2020-01-01"
    INITIAL_CAPITAL = 100000

_stub_module("config.settings", config=DummyConfig())

# 2. Stub Heavy Libs
_stub_module("psutil", virtual_memory=lambda: MagicMock(total=16*1024**3, used=1024**3))
_stub_module("numba", njit=lambda *args, **kwargs: (lambda f: f), prange=range)
_stub_module("tqdm", tqdm=lambda x, **k: x)
_stub_module("lightgbm")
_stub_module("xgboost")
_stub_module("catboost")
_stub_module("sklearn.covariance", LedoitWolf=MagicMock())

# 3. Stub Quant Alpha Internals
# Data
_stub_module("quant_alpha.data")
_stub_module("quant_alpha.data.DataManager", DataManager=MagicMock())

# Utils
_stub_module("quant_alpha.utils", 
             setup_logging=MagicMock(), 
             load_parquet=pd.read_parquet, 
             save_parquet=lambda df, path: df.to_parquet(path), 
             time_execution=lambda f: f,
             calculate_returns=lambda df: df.pct_change())

# Models
_stub_module("quant_alpha.models.lightgbm_model", LightGBMModel=MagicMock())
_stub_module("quant_alpha.models.xgboost_model", XGBoostModel=MagicMock())
_stub_module("quant_alpha.models.catboost_model", CatBoostModel=MagicMock())
_stub_module("quant_alpha.models.trainer", WalkForwardTrainer=MagicMock())
_stub_module("quant_alpha.models.feature_selector", FeatureSelector=MagicMock())

# Backtest & Opt
_stub_module("quant_alpha.backtest.engine", BacktestEngine=MagicMock())
_stub_module("quant_alpha.backtest.metrics", print_metrics_report=MagicMock())
_stub_module("quant_alpha.backtest.attribution", SimpleAttribution=MagicMock(), FactorAttribution=MagicMock())
_stub_module("quant_alpha.optimization.allocator", PortfolioAllocator=MagicMock())

# Visualization
_stub_module("quant_alpha.visualization", 
             plot_equity_curve=MagicMock(), plot_drawdown=MagicMock(),
             plot_monthly_heatmap=MagicMock(), plot_ic_time_series=MagicMock(),
             generate_tearsheet=MagicMock())

# Features (Imported by train_models.py)
_stub_module("quant_alpha.features.registry", FactorRegistry=MagicMock())

feature_modules = [
    "technical.momentum", "technical.volatility", "technical.volume", "technical.mean_reversion",
    "fundamental.value", "fundamental.quality", "fundamental.growth", "fundamental.financial_health",
    "earnings.surprises", "earnings.estimates", "earnings.revisions",
    "alternative.macro", "alternative.sentiment", "alternative.inflation",
    "composite.macro_adjusted", "composite.system_health", "composite.smart_signals"
]
for fm in feature_modules:
    _stub_module(f"quant_alpha.features.{fm}")

# ---------------------------------------------------------------------------
# Import Scripts (Now safe)
# ---------------------------------------------------------------------------
try:
    import scripts.deploy_model as deploy_model
    import scripts.generate_predictions as gen_pred
    import scripts.optimize_portfolio as opt_port
    # Also need train_models for shared utils used in generate_predictions
    import scripts.train_models as train_models
except ImportError as e:
    # If this fails, the test cannot run. We let it fail loudly.
    raise ImportError(f"Critical failure importing scripts: {e}")

# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestProductionCycle:
    
    @pytest.fixture
    def mock_env(self):
        """
        Sets up a temporary production environment (dirs, config) 
        and restores original state after test.
        """
        # 1. Create Temp Dirs
        temp_dir = tempfile.mkdtemp()
        root = Path(temp_dir)
        
        dirs = {
            "data": root / "data",
            "cache": root / "data" / "cache",
            "prices": root / "data" / "prices",
            "models": root / "models" / "production",
            "archive": root / "models" / "archive",
            "preds": root / "results" / "predictions",
            "orders": root / "results" / "orders",
            "logs": root / "logs",
            "results": root / "results",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # 2. Define Mock Config
        class MockConfig:
            def __init__(self):
                self.DATA_DIR = dirs["data"]
                self.CACHE_DIR = dirs["cache"]
                self.PRICES_DIR = dirs["prices"]
                self.MODELS_DIR = root / "models"
                self.RESULTS_DIR = dirs["results"]
                self.LOG_DIR = dirs["logs"]
                self.BACKTEST_START_DATE = "2020-01-01"
                self.INITIAL_CAPITAL = 100_000
                self.OPT_RISK_AVERSION = 2.5
                self.OPT_KELLY_FRACTION = 1.0
                self.RISK_FREE_RATE = 0.0
                self.OPT_LOOKBACK_DAYS = 20
                self.MAX_LEVERAGE = 1.0
                self.PROD_IC_THRESHOLD = 0.01
                self.PROD_IC_TSTAT = 2.0
                self.MIN_OOS_IC_THRESHOLD = 0.005
                self.MIN_OOS_IC_TSTAT = 1.5
                self.INFERENCE_SCALER_LOOKBACK_YEARS = 1
                
        mock_cfg = MockConfig()
        
        # 3. Patch Config in all loaded scripts
        # We need to patch the 'config' object imported in each script
        patches = []
        for module in [deploy_model, gen_pred, opt_port, train_models]:
            p = patch.object(module, 'config', mock_cfg)
            p.start()
            patches.append(p)
            
        yield dirs
        
        # 4. Cleanup
        for p in patches:
            p.stop()
        shutil.rmtree(temp_dir)

    def test_health_check_detects_stale_cache(self, mock_env):
        """
        deploy_model.verify_deployment should warn/fail if signal cache is missing or stale.
        """
        # Case 1: No cache -> Warning (but returns False for "all ok" if no models)
        # We need at least one model to trigger the check logic fully
        (mock_env["models"] / "TestModel_latest.pkl").touch()
        
        # Mock model that returns valid predictions for the smoke test
        mock_model = MagicMock()
        # deploy_model smoke test uses 50 rows
        mock_model.predict.return_value = np.array([0.5] * 50)

        # Mock joblib to avoid loading the empty file
        with patch("joblib.load", return_value={"model": mock_model, "feature_names": ["f1"]}):
            manager = deploy_model.DeploymentManager()
            
            # Run check - should log warning about missing cache
            with patch.object(deploy_model.logger, "warning") as mock_warn:
                manager.verify_deployment()
                # Should warn about missing ensemble_predictions.parquet
                # Note: verify_deployment logs warnings but might return False if no models or issues found
                # We check if logger.warning was called with specific text
                assert any("not found" in str(c) for c in mock_warn.call_args_list)

    def test_inference_generates_signals(self, mock_env):
        """
        generate_predictions should load models, process data, and save parquet.
        """
        # 1. Setup Dummy Data
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        master_df = pd.DataFrame({
            "date": dates,
            "ticker": "AAPL",
            "open": 100.0, "close": 101.0, "volume": 1000,
            "sector": "Tech", "industry": "Soft",
            "raw_ret_5d": 0.01, "target": 0.01,
            "f1": np.random.rand(5)
        })
        
        # 2. Mock Dependencies
        mock_model = MagicMock()
        # FIX: Return predictions matching input length (1 row when last_day_only=True)
        mock_model.predict.side_effect = lambda x: np.array([0.5] * len(x))
        payload = {"model": mock_model, "feature_names": ["f1"], "trained_to": "2022"}
        
        # We need to patch load_production_models and load_and_build_full_dataset
        # inside generate_predictions module
        with patch("scripts.generate_predictions.load_production_models", return_value={"M1": payload}), \
             patch("scripts.generate_predictions.load_and_build_full_dataset", return_value=master_df), \
             patch("scripts.generate_predictions.add_macro_features", side_effect=lambda x: x):
            
            # 3. Run
            gen_pred.generate_predictions(last_day_only=True)
            
        # 4. Verify Output
        out_files = list(mock_env["preds"].glob("*.parquet"))
        assert len(out_files) == 1
        df = pd.read_parquet(out_files[0])
        assert "ensemble_alpha" in df.columns
        assert len(df) == 1 # last day only

    def test_optimization_generates_orders(self, mock_env):
        """
        optimize_portfolio should read signals and generate orders.csv.
        """
        # 1. Setup Cache Files
        # Master Data
        master_df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-05", "2023-01-05"]),
            "ticker": ["AAPL", "MSFT"],
            "close": [150.0, 300.0],
            "market_cap": [2e12, 2.5e12]
        })
        master_df.to_parquet(mock_env["cache"] / "master_data_with_factors.parquet")
        
        # Predictions
        pred_df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-05", "2023-01-05"]),
            "ticker": ["AAPL", "MSFT"],
            "ensemble_alpha": [0.9, 0.8] # High scores
        })
        pred_df.to_parquet(mock_env["preds"] / "alpha_signals_2023-01-05.parquet")
        
        # 2. Run Optimizer
        # Mock risk model to avoid needing 60 days of history
        # FIX: Patch global path variables in optimize_portfolio which are set at import time
        # and don't automatically update when config is patched.
        with patch("scripts.optimize_portfolio.PRICES_DIR", mock_env["cache"] / "master_data_with_factors.parquet"), \
             patch("scripts.optimize_portfolio.PREDICTIONS_DIR", mock_env["preds"]), \
             patch("scripts.optimize_portfolio.OUTPUT_DIR", mock_env["orders"]), \
             patch.object(opt_port.ProductionOptimizer, "estimate_risk_model") as mock_risk:
            
            mock_risk.return_value = pd.DataFrame(
                [[0.04, 0.01], [0.01, 0.04]], 
                index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
            )
            
            # We also need to ensure the Allocator is mocked or works
            # Since we stubbed PortfolioAllocator in sys.modules, it returns a MagicMock
            # We need to configure that mock to return weights
            mock_allocator_instance = opt_port.PortfolioAllocator.return_value
            mock_allocator_instance.allocate.return_value = {"AAPL": 0.6, "MSFT": 0.4}
            
            optimizer = opt_port.ProductionOptimizer(capital=100_000, method="mean_variance")
            optimizer.run()
            
        # 3. Verify Orders
        latest = mock_env["orders"] / "orders_latest.csv"
        assert latest.exists()
        orders = pd.read_csv(latest)
        assert len(orders) == 2
        assert orders.iloc[0]["ticker"] == "AAPL"
        assert orders.iloc[0]["side"] == "LONG"
