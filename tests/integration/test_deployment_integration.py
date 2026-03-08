"""
INTEGRATION TEST: Deployment Lifecycle
======================================
Tests the DeploymentManager script to ensure the check -> archive -> prune
lifecycle works as expected in a simulated production environment.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import components to be tested
from scripts.deploy_model import DeploymentManager
from config.settings import config

# Define a picklable dummy model class at module level
class DummyModel:
    def predict(self, x):
        # Return random predictions to pass variance check (std > 1e-6)
        return np.random.default_rng(42).random(len(x))


@pytest.fixture
def mock_prod_env(tmp_path, monkeypatch):
    """
    Creates a temporary production environment with a dummy model,
    and patches the config to use it.
    """
    prod_dir = tmp_path / "models" / "production"
    archive_dir = tmp_path / "models" / "archive"
    cache_dir = tmp_path / "data" / "cache"
    
    prod_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch config to use temp dirs
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)

    # Create dummy signal cache to avoid "RETRAIN URGENTLY" logs
    pd.DataFrame({"date": [pd.Timestamp.now()]}).to_parquet(
        cache_dir / "ensemble_predictions.parquet"
    )

    # Create a dummy model payload
    dummy_model = DummyModel()
    payload = {
        "model": dummy_model,
        "feature_names": ["f_001", "f_002"],
        "trained_to": "2023-12-31",
        "oos_metrics": {"ic_mean": 0.02, "ic_std": 0.01, "n_dates": 252}
    }
    
    # Save dummy model
    model_path = prod_dir / "LGBM_latest.pkl"
    joblib.dump(payload, model_path)
    
    return prod_dir, archive_dir

@pytest.mark.integration
class TestDeploymentIntegration:
    """
    Integration tests for the DeploymentManager script.
    Verifies the check -> archive -> prune lifecycle.
    """

    def test_verify_deployment(self, mock_prod_env, caplog):
        """
        Test that verify_deployment can load and check a dummy model.
        """
        manager = DeploymentManager()
        healthy = manager.verify_deployment()

        assert healthy is True
        assert "Verifying 1 production model(s)" in caplog.text
        assert "Lgbm" in caplog.text
        assert "IC=+0.0200" in caplog.text
        assert "✅ PROD" in caplog.text # Check gate tier logic

    def test_archive_and_prune_lifecycle(self, mock_prod_env):
        """
        Test the full archive and prune lifecycle.
        """
        prod_dir, archive_dir = mock_prod_env
        manager = DeploymentManager()

        # 1. Archive
        archive_path = manager.archive_current_models()
        assert archive_path is not None
        assert archive_path.is_dir()
        assert (archive_path / "LGBM_latest.pkl").exists()
        assert (archive_path / "manifest.json").exists()

        # 2. Prune (should keep the one we just made)
        manager.prune_archives(keep_last=1)
        archives = list(archive_dir.iterdir())
        assert len(archives) == 1
        assert archives[0].name == archive_path.name

        # 3. Archive again to have something to prune
        manager.archive_current_models()
        
        # 4. Prune again (keep 1, delete 1)
        manager.prune_archives(keep_last=1)
        archives = list(archive_dir.iterdir())
        assert len(archives) == 1
