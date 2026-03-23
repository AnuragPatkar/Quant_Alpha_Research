"""
Model Deployment Lifecycle Integration Tests
============================================
Validates the promotion, archival, and pruning mechanisms of the production deployment manager.

Purpose
-------
This module executes integration tests against the `DeploymentManager` to
verify that the strict `check -> archive -> prune` state machine operates 
correctly within a simulated production filesystem. It ensures that model
artifacts and manifests are durably persisted and that legacy models are 
safely garbage collected without violating retention policies.

Role in Quantitative Workflow
-----------------------------
Provides automated assurance that the deployment boundary securely manages 
the transition of Gate-passing models into live production trading, preventing
accidental deletion or corrupted artifact loads.

Dependencies
------------
- **Pytest**: Test execution, log capturing (`caplog`), and temporary directory (`tmp_path`) management.
- **Joblib**: Mock artifact serialization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import warnings

warnings.filterwarnings("ignore", message=".*pyarrow.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.deploy_model import DeploymentManager
from config.settings import config

class DummyModel:
    """
    Synthetic model representation for artifact serialization testing.
    """
    def predict(self, x):
        r"""
        Generates continuous predictions guaranteeing specific statistical moments.

        Args:
            x (pd.DataFrame | np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: Random uniform predictions to strictly satisfy the systemic 
                variance constraints ($\sigma > 1e-6$) evaluated during smoke testing.
        """
        return np.random.default_rng(42).random(len(x))


@pytest.fixture
def mock_prod_env(tmp_path, monkeypatch):
    """
    Provisions a transient production environment injected with dummy artifacts.

    Overrides the global configuration to map persistent storage directories
    to isolated temporary paths. Initializes a deterministic dummy model payload
    and a synthetic signal cache to simulate a live execution state.

    Args:
        tmp_path (pathlib.Path): Pytest fixture providing a unique temporary directory.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for safely mutating global state.

    Returns:
        tuple: A 2-element tuple containing:
            - pathlib.Path: The provisioned production models directory.
            - pathlib.Path: The provisioned archive models directory.
    """
    prod_dir = tmp_path / "models" / "production"
    archive_dir = tmp_path / "models" / "archive"
    cache_dir = tmp_path / "data" / "cache"
    
    prod_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)

    # Injects an immediate timestamp into the signal cache to explicitly bypass the 
    # staleness detection thresholds (T > 21 days) that would otherwise trigger a deployment abort
    pd.DataFrame({"date": [pd.Timestamp.now()]}).to_parquet(
        cache_dir / "ensemble_predictions.parquet"
    )

    dummy_model = DummyModel()
    payload = {
        "model": dummy_model,
        "feature_names": ["f_001", "f_002"],
        "trained_to": "2023-12-31",
        "oos_metrics": {"ic_mean": 0.02, "ic_std": 0.01, "n_dates": 252}
    }
    
    model_path = prod_dir / "LGBM_latest.pkl"
    joblib.dump(payload, model_path)
    
    return prod_dir, archive_dir

@pytest.mark.integration
class TestDeploymentIntegration:
    """
    Integration suite for the DeploymentManager lifecycle validation.
    """

    def test_verify_deployment(self, mock_prod_env, caplog):
        """
        Validates the deployment verification sequence against a synthetic artifact.

        Asserts that the `verify_deployment` routine successfully deserializes 
        the payload, computes the out-of-sample metrics, and accurately maps 
        the artifact to the 'PROD' tier based on its Information Coefficient bounds.

        Args:
            mock_prod_env (tuple): Injected isolated filesystem context.
            caplog (pytest.LogCaptureFixture): Pytest fixture to capture and assert emitted telemetry.

        Returns:
            None
        """
        manager = DeploymentManager()
        healthy = manager.verify_deployment()

        assert healthy is True
        assert "Verifying 1 production model(s)" in caplog.text
        assert "Lgbm" in caplog.text
        assert "IC=+0.0200" in caplog.text
        
        # Evaluates the deterministic resolution of the internal promotion tier logic
        assert "✅ PROD" in caplog.text

    def test_archive_and_prune_lifecycle(self, mock_prod_env):
        """
        Validates the state machine governing artifact persistence and garbage collection.

        Executes a full archive cycle to assert manifest generation and file persistence.
        Subsequently triggers the pruning mechanism to guarantee strict adherence to 
        retention policies, explicitly keeping recent snapshots and evicting legacy bounds.

        Args:
            mock_prod_env (tuple): Injected isolated filesystem context.

        Returns:
            None
        """
        prod_dir, archive_dir = mock_prod_env
        manager = DeploymentManager()

        # Execute explicit snapshot generation and assert structural integrity of the manifest
        archive_path = manager.archive_current_models()
        assert archive_path is not None
        assert archive_path.is_dir()
        assert (archive_path / "LGBM_latest.pkl").exists()
        assert (archive_path / "manifest.json").exists()

        # Enforce retention policy boundary and verify protection of the active state
        manager.prune_archives(keep_last=1)
        archives = list(archive_dir.iterdir())
        assert len(archives) == 1
        assert archives[0].name == archive_path.name

        # Simulate temporal progression by generating an isolated sequential artifact
        manager.archive_current_models()
        
        # Execute secondary garbage collection to definitively assert the eviction of legacy nodes
        manager.prune_archives(keep_last=1)
        archives = list(archive_dir.iterdir())
        assert len(archives) == 1
