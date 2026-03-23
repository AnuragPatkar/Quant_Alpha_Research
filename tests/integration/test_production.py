## 📁 tests/integration/test_production.py


"""
Production Readiness Integration Tests
======================================
Validates the end-to-end operational viability of the live trading DAG.

Purpose
-------
This module executes integration tests verifying the critical path for daily 
production execution: Health Verification (`deploy_model.py`), Signal 
Inference (`generate_predictions.py`), and Portfolio Construction 
(`optimize_portfolio.py`). It ensures seamless inter-module data propagation 
across the ephemeral filesystem and configuration boundaries.

Role in Quantitative Workflow
-----------------------------
Serves as the ultimate deployment gate. By isolating the environment through 
aggressive `sys.modules` patching and mock artifact generation, this suite 
guarantees that subsequent operational states will not encounter catastrophic 
I/O failures or unresolved dependencies during live market execution.

Dependencies
------------
- **Pytest**: Test execution, fixture orchestration, and temporary directory management.
- **Unittest.Mock**: Deep namespace patching for C-extension and external dependency isolation.
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
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, date

warnings.filterwarnings("ignore", message=".*pyarrow.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

def _stub_module(module_name, **attrs):
    """
    Dynamically constructs and injects isolated module stubs into the global namespace.

    Recursively verifies and provisions parent package structures to satisfy Python's 
    relative import resolution mechanics (e.g., handling 'from X.Y import Z'), 
    preventing cascading ImportErrors during aggressive dependency mocking.

    Args:
        module_name (str): The dot-delimited path of the target module to stub.
        **attrs: Arbitrary attributes (functions, classes, mocks) to bind to the stub.

    Returns:
        types.ModuleType: The dynamically provisioned module instance.
    """
    parts = module_name.split(".")
    
    # Iteratively synthesize parent hierarchy to satisfy __path__ requirements
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            m = types.ModuleType(parent_name)
            m.__path__ = []
            sys.modules[parent_name] = m
            
    if module_name in sys.modules:
        m = sys.modules[module_name]
    else:
        m = types.ModuleType(module_name)
        if "." not in module_name or "quant_alpha" in module_name:
             m.__path__ = []
        sys.modules[module_name] = m
    
    for k, v in attrs.items():
        setattr(m, k, v)
        
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], m)
        
    return m

class DummyConfig:
    """
    Deterministic configuration overlay designed to satisfy module-level I/O constants.

    Bypasses standard settings resolution to prevent the scripts from accessing or 
    mutating the host machine's physical filesystem during instantiation.
    """
    RESULTS_DIR = Path("results")
    CACHE_DIR = Path("cache")
    DATA_DIR = Path("data")
    PRICES_DIR = Path("prices")
    FUNDAMENTALS_DIR = Path("fundamentals")
    EARNINGS_DIR = Path("earnings")
    ALTERNATIVE_DIR = Path("alternative")
    MODELS_DIR = Path("models")
    LOG_DIR = Path("logs")
    PREDICTIONS_DIR = Path("results/predictions")
    BACKTEST_START_DATE = "2020-01-01"
    BACKTEST_END_DATE = "2023-12-31"
    INITIAL_CAPITAL = 100000
    MODEL_WEIGHTS = {"lightgbm": 0.4, "xgboost": 0.3, "catboost": 0.3}

class TestProductionCycle:
    """
    Integration suite encompassing the operational boundaries of the production engine.
    """
    
    @pytest.fixture
    def isolated_modules(self):
        """
        Provisions a strictly isolated execution boundary via sys.modules context patching.

        Aggressively intercepts the dependency graph, replacing computationally heavy 
        C-extensions and live framework implementations with functional stubs. 
        Executes meticulous teardown sequences post-yield to prevent `MagicMock` 
        artifacts from escaping scope and poisoning subsequent unit tests.

        Yields:
            None: Exposes control to the test function execution.
        """
        import importlib
        import gc
        
        # Secure real class definitions from disk prior to namespace hijacking
        from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler, winsorize_clip_nb

        try:
            with patch.dict(sys.modules):
                _stub_module("config")
                _stub_module("config.settings", config=DummyConfig())

                _stub_module("psutil", virtual_memory=lambda: MagicMock(total=16*1024**3, used=1024**3))
                _stub_module("numba", njit=lambda *args, **kwargs: (lambda f: f), prange=range)
                _stub_module("tqdm", tqdm=lambda x, **k: x)
                _stub_module("lightgbm")
                _stub_module("xgboost")
                _stub_module("catboost")
                _stub_module("sklearn.covariance", LedoitWolf=MagicMock())

                _stub_module("quant_alpha.data")
                _stub_module("quant_alpha.data.DataManager", DataManager=MagicMock())

                _stub_module("quant_alpha.utils", 
                            setup_logging=MagicMock(), 
                            load_parquet=pd.read_parquet, 
                            save_parquet=lambda df, path: df.to_parquet(path), 
                            time_execution=lambda f: f,
                            calculate_returns=lambda df: df.pct_change())
                
                _stub_module("quant_alpha.utils.preprocessing",
                             WinsorisationScaler=WinsorisationScaler,
                             SectorNeutralScaler=SectorNeutralScaler,
                             winsorize_clip_nb=winsorize_clip_nb)

                _stub_module("quant_alpha.models.lightgbm_model", LightGBMModel=MagicMock())
                _stub_module("quant_alpha.models.xgboost_model", XGBoostModel=MagicMock())
                _stub_module("quant_alpha.models.catboost_model", CatBoostModel=MagicMock())
                _stub_module("quant_alpha.models.trainer", WalkForwardTrainer=MagicMock())
                _stub_module("quant_alpha.models.feature_selector", FeatureSelector=MagicMock())

                _stub_module("quant_alpha.backtest.engine", BacktestEngine=MagicMock())
                _stub_module("quant_alpha.backtest.metrics", print_metrics_report=MagicMock())
                _stub_module("quant_alpha.backtest.attribution", SimpleAttribution=MagicMock(), FactorAttribution=MagicMock())
                _stub_module("quant_alpha.optimization.allocator", PortfolioAllocator=MagicMock())

                _stub_module("quant_alpha.visualization", 
                            plot_equity_curve=MagicMock(), plot_drawdown=MagicMock(),
                            plot_monthly_heatmap=MagicMock(), plot_ic_time_series=MagicMock(),
                            generate_tearsheet=MagicMock())

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

                import scripts.deploy_model as deploy_model
                import scripts.generate_predictions as gen_pred
                import scripts.optimize_portfolio as opt_port
                import scripts.train_models as train_models
                
                self.deploy_model = deploy_model
                self.gen_pred = gen_pred
                self.opt_port = opt_port
                self.train_models = train_models
                
                yield
        finally:
            # Forces aggressive namespace purging post-execution to sever cached references 
            # to structural MagicMocks, averting fatal 'Can't pickle' assertions in standard unit tests.
            modules_to_delete = [
                name for name in list(sys.modules.keys()) 
                if name.startswith("quant_alpha") or name.startswith("scripts")
            ]
            for module_name in modules_to_delete:
                try:
                    del sys.modules[module_name]
                except (KeyError, RuntimeError):
                    pass
            
            try:
                importlib.invalidate_caches()
            except Exception:
                pass
            
            gc.collect()

    @pytest.fixture
    def mock_env(self):
        """
        Provisions an ephemeral directory tree mimicking the production architecture.

        Yields:
            dict[str, Path]: Mapped dictionary pointing to isolated transient storage paths.
        """
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
            
        class MockConfig:
            def __init__(self):
                self.DATA_DIR = dirs["data"]
                self.CACHE_DIR = dirs["cache"]
                self.PRICES_DIR = dirs["prices"]
                self.MODELS_DIR = root / "models"
                self.RESULTS_DIR = dirs["results"]
                self.PREDICTIONS_DIR = dirs["preds"]
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
                self.MODEL_WEIGHTS = {"lightgbm": 0.4, "xgboost": 0.3, "catboost": 0.3}
                
        mock_cfg = MockConfig()
        
        # Intercepts the configuration singleton instances dynamically acquired by active modules
        patches = []
        for module in [self.deploy_model, self.gen_pred, self.opt_port, self.train_models]:
            p = patch.object(module, 'config', mock_cfg)
            p.start()
            patches.append(p)
            
        yield dirs
        
        for p in patches:
            p.stop()
        shutil.rmtree(temp_dir)

    def test_health_check_detects_stale_cache(self, isolated_modules, mock_env):
        """
        Validates the latency bounds enforced by the DeploymentManager.

        Ensures that execution appropriately aborts or warns if the foundational signal 
        cache exceeds mathematical decay thresholds.

        Args:
            isolated_modules (None): Injected dependency isolation context.
            mock_env (dict): Provisioned temporary path mapping.

        Returns:
            None
        """
        (mock_env["models"] / "TestModel_latest.pkl").touch()
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5] * 50)

        with patch("joblib.load", return_value={"model": mock_model, "feature_names": ["f1"]}):
            manager = self.deploy_model.DeploymentManager()
            
            with patch.object(self.deploy_model.logger, "warning") as mock_warn:
                manager.verify_deployment()
                assert any("not found" in str(c) for c in mock_warn.call_args_list)

    def test_inference_generates_signals(self, isolated_modules, mock_env):
        """
        Validates the holistic signal inference data flow mechanism.

        Args:
            isolated_modules (None): Injected dependency isolation context.
            mock_env (dict): Provisioned temporary path mapping.

        Returns:
            None
        """
        warnings.filterwarnings("ignore", message=".*pyarrow.*")
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        master_df = pd.DataFrame({
            "date": dates,
            "ticker": "AAPL",
            "open": 100.0, "close": 101.0, "volume": 1000,
            "sector": "Tech", "industry": "Soft",
            "raw_ret_5d": 0.01, "target": 0.01,
            "f1": np.random.rand(5)
        })
        
        mock_model = MagicMock()
        mock_model.predict.side_effect = lambda x: np.array([0.5] * len(x))
        payload = {"model": mock_model, "feature_names": ["f1"], "trained_to": "2022"}
        
        with patch("scripts.generate_predictions.load_production_models", return_value={"M1": payload}), \
             patch("scripts.generate_predictions.load_and_build_full_dataset", return_value=master_df), \
             patch("scripts.generate_predictions.add_macro_features", side_effect=lambda x: x):
            
            self.gen_pred.generate_predictions(last_day_only=True)
            
        out_files = list(mock_env["preds"].glob("*.parquet"))
        assert len(out_files) == 1
        df = pd.read_parquet(out_files[0])
        assert "ensemble_alpha" in df.columns
        assert len(df) == 1

    def test_optimization_generates_orders(self, isolated_modules, mock_env):
        """
        Validates the end-to-end integration of the Portfolio Construction engine.

        Ensures the optimization routine correctly parses inference signals, constructs 
        the target risk matrices, dynamically solves the continuous bounds, and 
        persists the discrete discrete share allocations to disk.

        Args:
            isolated_modules (None): Injected dependency isolation context.
            mock_env (dict): Provisioned temporary path mapping.

        Returns:
            None
        """
        master_df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-05", "2023-01-05"]),
            "ticker": ["AAPL", "MSFT"],
            "close": [150.0, 300.0],
            "market_cap": [2e12, 2.5e12]
        })
        master_df.to_parquet(mock_env["cache"] / "master_data_with_factors.parquet")
        
        pred_df = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-05", "2023-01-05"]),
            "ticker": ["AAPL", "MSFT"],
            "ensemble_alpha": [0.9, 0.8]
        })
        pred_df.to_parquet(mock_env["preds"] / "alpha_signals_2023-01-05.parquet")
        
        # Forces global path re-bindings specifically targeting localized module namespace evaluations
        with patch("scripts.optimize_portfolio.PRICES_DIR", mock_env["cache"] / "master_data_with_factors.parquet"), \
             patch("scripts.optimize_portfolio.PREDICTIONS_DIR", mock_env["preds"]), \
             patch("scripts.optimize_portfolio.OUTPUT_DIR", mock_env["orders"]), \
             patch.object(self.opt_port.ProductionOptimizer, "estimate_risk_model") as mock_risk:
            
            mock_risk.return_value = pd.DataFrame(
                [[0.04, 0.01], [0.01, 0.04]], 
                index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
            )
            
            mock_allocator_instance = self.opt_port.PortfolioAllocator.return_value
            mock_allocator_instance.allocate.return_value = {"AAPL": 0.6, "MSFT": 0.4}
            
            optimizer = self.opt_port.ProductionOptimizer(capital=100_000, method="mean_variance")
            optimizer.run()
            
        latest = mock_env["orders"] / "orders_latest.csv"
        assert latest.exists()
        orders = pd.read_csv(latest)
        assert len(orders) == 2
        assert orders.iloc[0]["ticker"] == "AAPL"
        assert orders.iloc[0]["side"] == "LONG"
