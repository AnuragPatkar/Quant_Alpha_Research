"""
Machine Learning Models and Objectives Unit Tests
=================================================
Validates the lifecycle, determinism, and mathematical correctness of predictive models.

Purpose
-------
This module isolates and tests the behavior of wrapped Gradient Boosted Decision 
Tree (GBDT) estimators (LightGBM, XGBoost, CatBoost) and custom loss functions 
(e.g., Weighted Symmetric Mean Absolute Error). It ensures stable model serialization,
deterministic predictions, and proper handling of edge cases like NaNs and missing 
categorical variables without invoking full pipeline execution.

Role in Quantitative Workflow
-----------------------------
Acts as the foundational safety layer for the predictive modeling engine, 
guaranteeing that models accurately interpret features, respect optimization 
objectives, and can be reliably saved and deployed to production.

Dependencies
------------
- **Pytest**: Test execution framework and parameterized suite orchestration.
- **Pandas/NumPy**: Synthetic feature generation and mathematical bounds testing.
- **Joblib**: Persistence validation for model serialization mechanisms.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import joblib
import io
import _pickle
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from quant_alpha.models.lightgbm_model import LightGBMModel
except ImportError:
    LightGBMModel = None

try:
    from quant_alpha.models.xgboost_model import XGBoostModel
except ImportError:
    XGBoostModel = None

try:
    from quant_alpha.models.catboost_model import CatBoostModel
except ImportError:
    CatBoostModel = None

try:
    from tests.conftest import weighted_symmetric_mae
    _OBJECTIVE_AVAILABLE = True
except ImportError:
    _OBJECTIVE_AVAILABLE = False
    weighted_symmetric_mae = None

def _build_params(name: str, base: dict) -> dict:
    """
    Constructs model-specific hyperparameter dictionaries dynamically.

    Explicitly copies and normalizes baseline parameters to satisfy the unique 
    API requirements of different underlying GBDT libraries (e.g., translating 
    'verbose' to 'verbosity' for XGBoost).

    Args:
        name (str): The string identifier of the target model ('LightGBM', 'XGBoost', 'CatBoost').
        base (dict): The baseline parameter dictionary to adapt.

    Returns:
        dict: A strictly isolated, model-compatible parameter dictionary.
    """
    params = base.copy()

    if name == "XGBoost":
        # Maps canonical verbosity flags to XGBoost-specific argument constraints
        params.pop("verbose", None)
        params["verbosity"] = 0

    if name == "CatBoost":
        params.pop("n_jobs", None)         # CatBoost uses thread_count
        params["thread_count"]  = 1
        params.pop("verbose", None)
        params["verbose"]       = 0
        params.pop("random_state", None)   # CatBoost uses random_seed
        params["random_seed"]   = base.get("random_state", 42)
        params.pop("n_estimators", None)   # CatBoost uses iterations
        params["iterations"]    = base.get("n_estimators", 10)
        params["allow_writing_files"] = False

    return params


def _detect_cat_features(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Dynamically identifies categorical feature boundaries within the input matrix.

    Args:
        df (pd.DataFrame): The input feature matrix.
        feature_cols (list): The list of feature column identifiers to evaluate.

    Returns:
        list: A subset of column names mapped to object or categorical data types.
    """
    return [
        col for col in feature_cols
        if col in df.columns and (
            pd.api.types.is_object_dtype(df[col]) or
            isinstance(df[col].dtype, pd.CategoricalDtype)
        )
    ]

class TestModels:
    """
    Validation suite for Model Wrappers and Custom Objectives.
    """

    @pytest.mark.parametrize("model_class, name", [
        (LightGBMModel,  "LightGBM"),
        (XGBoostModel,   "XGBoost"),
        (CatBoostModel,  "CatBoost"),
    ])
    def test_model_lifecycle(self, model_class, name, synthetic_data):
        """
        Validates the strict init-fit-predict lifecycle across partitioned execution sets.

        Ensures that wrapped estimators can correctly ingest data, map categorical 
        features, and generate valid, non-constant predictions on strictly held-out 
        test bounds to verify feature alignment.

        Args:
            model_class (type): The uninstantiated wrapper class for the target model.
            name (str): The canonical string identifier for the model.
            synthetic_data (tuple): Injected synthetic data structure containing 
                the DataFrame, feature lists, and target column name.

        Returns:
            None
        """
        if model_class is None:
            pytest.skip(f"{name} not installed.")

        df, features, target_col = synthetic_data
        df = df.copy()

        # Verifies signal feature exhibits sufficient variance to prevent degenerate regression states
        signal_feature = features[0]
        feature_std    = df[signal_feature].std()
        assert feature_std > 0.01, (
            f"features[0] ('{signal_feature}') has std={feature_std:.4f} — "
            "near-constant feature cannot produce a meaningful regression signal. "
            "Check synthetic_data fixture in conftest.py."
        )

        # Inject clear signal — target proportional to first feature
        df[target_col] = df[signal_feature] * 2.0 + np.random.default_rng(42).normal(
            0, feature_std * 0.1, len(df)
        )

        # Implements strict 80/20 chronological train/test splitting to prevent structural temporal leakage
        split  = int(len(df) * 0.8)
        train  = df.iloc[:split].copy()
        test   = df.iloc[split:].copy()
        assert len(test) >= 5, (
            f"Test set has only {len(test)} rows after 80/20 split. "
            "synthetic_data fixture needs at least 25 rows."
        )

        # Dynamically isolates and maps structural categorical feature bounds
        cat_cols = _detect_cat_features(df, features)

        base_params = {
            "n_estimators":  10,
            "learning_rate": 0.1,
            "random_state":  42,
            "n_jobs":        1,
            "verbose":       -1,
        }
        params = _build_params(name, base_params)

        if name == "CatBoost":
            # Explicitly conditionally binds cat_features to prevent CatBoost instantiation failures
            if cat_cols:
                params["cat_features"] = cat_cols
            elif any(c in features for c in ("sector", "industry")):
                pytest.skip(
                    "CatBoost requires categorical columns in feature set, "
                    "but none detected. Check synthetic_data fixture."
                )

        model = model_class(params=params)

        model.fit(train[features], train[target_col])

        train_preds = model.predict(train[features])
        assert len(train_preds) == len(train),  f"{name}: train pred length mismatch"
        assert not np.isnan(train_preds).any(), f"{name}: NaN in train predictions"

        test_preds = model.predict(test[features])
        assert len(test_preds) == len(test),   f"{name}: test pred length mismatch"
        assert not np.isnan(test_preds).any(), f"{name}: NaN in test predictions"
        assert isinstance(test_preds, (np.ndarray, pd.Series))

        assert np.std(train_preds) > 1e-6, (
            f"{name} train predictions are constant (std≈0). "
            "Model may be ignoring features or objective is degenerate."
        )

        # Asserts structural constraints on model predictions to detect degenerate objective topologies
        max_abs = float(np.abs(test_preds).max())
        assert max_abs < 1e4, (
            f"{name} test predictions out of range: max_abs={max_abs:.2f}. "
            "Possible objective misconfiguration or feature scaling issue."
        )

    @pytest.mark.parametrize("model_class, name", [
        (LightGBMModel, "LightGBM"),
        (XGBoostModel,  "XGBoost"),
        (CatBoostModel, "CatBoost"),
    ])
    def test_model_persistence(self, model_class, name, synthetic_data):
        """
        Verifies the serialization integrity of fitted model estimators.

        Ensures that a fully trained model can be serialized and subsequently 
        deserialized via Joblib without state corruption or loss of prediction determinism.

        Args:
            model_class (type): The uninstantiated wrapper class for the target model.
            name (str): The canonical string identifier for the model.
            synthetic_data (tuple): Injected synthetic data structure.

        Returns:
            None
        """
        if model_class is None:
            pytest.skip(f"{name} not installed.")

        df, features, target_col = synthetic_data
        params = _build_params(name, {"n_estimators": 5, "verbose": -1})
        
        # Dynamically acquires the active class definition from sys.modules to prevent 
        # PicklingError artifacts resulting from aggressive namespace flushing in integration tests.
        import importlib
        live_module = importlib.import_module(model_class.__module__)
        live_class = getattr(live_module, model_class.__name__)
        
        model = live_class(params=params)
        model.fit(df[features], df[target_col])
        
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        loaded_model = joblib.load(buffer)
        
        preds_orig = model.predict(df[features])
        preds_load = loaded_model.predict(df[features])
        
        np.testing.assert_array_equal(preds_orig, preds_load, 
                                      err_msg=f"{name}: Pickled model predictions differ from original")

    def test_custom_objective_gradients(self):
        """
        Validates the mathematical boundaries of the custom asymmetric loss function.

        Evaluates the `weighted_symmetric_mae` objective to ensure it accurately 
        computes first-order (gradient) and second-order (hessian) derivatives. 
        Strictly enforces that directional sign errors are penalized with twice 
        the magnitude (weight=2.0) compared to magnitude-only errors.

        Args:
            None

        Returns:
            None
        """
        if not _OBJECTIVE_AVAILABLE:
            pytest.skip("weighted_symmetric_mae not importable from tests.conftest")

        y_true = np.array([1.0,   1.0,  -1.0])
        y_pred = np.array([0.5,  -0.5,   0.5])

        grad, hess = weighted_symmetric_mae(y_true, y_pred)

        assert grad.shape == y_true.shape, \
            f"grad shape {grad.shape} != y_true shape {y_true.shape}"
        assert hess.shape == y_true.shape, \
            f"hess shape {hess.shape} != y_true shape {y_true.shape}"

        expected_grad_0 = -1.0 * np.tanh(0.5)   
        assert np.isclose(grad[0], expected_grad_0, atol=1e-6), (
            f"Case 0 (correct sign): grad={grad[0]:.6f}, "
            f"expected {expected_grad_0:.6f}"
        )

        expected_grad_1 = -2.0 * np.tanh(1.5)   
        assert np.isclose(grad[1], expected_grad_1, atol=1e-6), (
            f"Case 1 (wrong sign, positive y_true): grad={grad[1]:.6f}, "
            f"expected {expected_grad_1:.6f}"
        )

        # Evaluates asymmetric loss logic for opposing signs with negative true outcomes
        # weight=2 (wrong sign), residual negative → grad = -2*tanh(-1.5) = +2*tanh(1.5)
        expected_grad_2 = -2.0 * np.tanh(-1.5)  
        assert np.isclose(grad[2], expected_grad_2, atol=1e-6), (
            f"Case 2 (wrong sign, negative y_true): grad={grad[2]:.6f}, "
            f"expected {expected_grad_2:.6f}. "
            "This catches gradient functions that ignore the sign of y_true."
        )

        assert np.sign(grad[1]) != np.sign(grad[2]), (
            f"Cases 1 and 2 have same gradient sign ({np.sign(grad[1])}) "
            "but opposite residuals — gradient is ignoring residual sign."
        )

        assert (hess > 0).all(), f"Hessian must be positive, got: {hess}"

        # Validates Hessian magnitude constraints ensuring wrong-sign derivatives exhibit 2x curvature scaling
        # hess = weight * sech²(residual)
        # hess[0] = 1.0 * sech²(0.5),  hess[1] = 2.0 * sech²(1.5)
        # Ratio hess[1]/hess[0] = 2 * sech²(1.5) / sech²(0.5)
        sech2 = lambda x: 1.0 / np.cosh(x) ** 2
        expected_hess_0 = 1.0 * sech2(0.5)
        expected_hess_1 = 2.0 * sech2(1.5)
        expected_hess_2 = 2.0 * sech2(1.5)  

        assert np.isclose(hess[0], expected_hess_0, atol=1e-6), (
            f"hess[0] (correct sign): {hess[0]:.6f}, expected {expected_hess_0:.6f}"
        )
        assert np.isclose(hess[1], expected_hess_1, atol=1e-6), (
            f"hess[1] (wrong sign): {hess[1]:.6f}, expected {expected_hess_1:.6f}. "
            "Wrong-sign hessian must be 2x the weight of correct-sign hessian."
        )
        assert np.isclose(hess[2], expected_hess_2, atol=1e-6), (
            f"hess[2] (wrong sign, negative y_true): {hess[2]:.6f}, "
            f"expected {expected_hess_2:.6f}"
        )

        # Asserts strictly that the multiplicative penalty factor dominates the diminishing hyperbolic secant component
        assert hess[1] / hess[0] == pytest.approx(
            expected_hess_1 / expected_hess_0, rel=1e-4
        ), (
            f"Hessian ratio hess[1]/hess[0] = {hess[1]/hess[0]:.4f}, "
            f"expected {expected_hess_1/expected_hess_0:.4f}. "
            "Hessian must reflect the weight multiplier for wrong-sign predictions."
        )

    @pytest.mark.parametrize("model_class, name", [
        (LightGBMModel, "LightGBM"),
        (XGBoostModel,  "XGBoost"),
        (CatBoostModel, "CatBoost"),
    ])
    def test_prediction_determinism(self, model_class, name, synthetic_data):
        """
        Guarantees deterministic prediction outputs under fixed random seed states.

        Verifies that repeated instantiations of the model with identical 
        hyperparameters and feature states yield strictly identical floating-point 
        prediction vectors, which is critical for replicable research.

        Args:
            model_class (type): The uninstantiated wrapper class for the target model.
            name (str): The canonical string identifier for the model.
            synthetic_data (tuple): Injected synthetic data structure.

        Returns:
            None
        """
        if model_class is None:
            pytest.skip(f"{name} not installed.")

        df, features, target_col = synthetic_data
        df = df.copy()

        base_params = {
            "n_estimators":  10,
            "random_state":  42,
            "n_jobs":        1,
            "verbose":       -1,
        }
        params = _build_params(name, base_params)

        if name == "CatBoost":
            cat_cols = _detect_cat_features(df, features)
            if cat_cols:
                params["cat_features"] = cat_cols

        m1 = model_class(params=params)
        m1.fit(df[features], df[target_col])
        p1 = m1.predict(df[features])

        # Clones initialization parameter mapping to strictly isolate repeated instantiation states
        m2 = model_class(params=params.copy())   
        m2.fit(df[features], df[target_col])
        p2 = m2.predict(df[features])

        np.testing.assert_allclose(
            p1, p2, rtol=1e-5, atol=1e-8,
            err_msg=(
                f"{name} predictions are not deterministic with random_state=42. "
                "Check that the model wrapper passes the seed to the underlying library. "
                "For CatBoost use random_seed; for XGBoost use seed/random_state."
            )
        )

    @pytest.mark.parametrize("model_class, name", [
        (LightGBMModel, "LightGBM"),
        (XGBoostModel,  "XGBoost"),
    ])
    def test_nan_in_features(self, model_class, name, synthetic_data):
        """
        Evaluates the structural resilience of models against missing feature inputs.

        Injects NaNs into specific numerical features to verify that the underlying 
        GBDT algorithms (which natively support missing values via directional splits) 
        do not crash during fitting or prediction phases.

        Args:
            model_class (type): The uninstantiated wrapper class for the target model.
            name (str): The canonical string identifier for the model.
            synthetic_data (tuple): Injected synthetic data structure.

        Returns:
            None
        """
        if model_class is None:
            pytest.skip(f"{name} not installed.")

        df, features, target_col = synthetic_data
        df = df.copy()

        # Synthetically injects structural sparsity to evaluate algorithm stability parameters
        rng = np.random.default_rng(seed=77)
        numeric_features = [
            f for f in features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]
        assert numeric_features, "No numeric features in synthetic_data"

        for feat in numeric_features[:3]:  
            nan_idx = rng.choice(len(df), size=max(1, len(df) // 10), replace=False)
            df.loc[nan_idx, feat] = np.nan

        params = _build_params(name, {
            "n_estimators": 5, "random_state": 42, "n_jobs": 1, "verbose": -1
        })

        model = model_class(params=params)
        try:
            model.fit(df[features], df[target_col])
            preds = model.predict(df[features])
        except Exception as e:
            pytest.fail(
                f"{name} crashed with NaN in features: {e}. "
                "LightGBM and XGBoost should handle NaN natively."
            )

        assert len(preds) == len(df)
        assert np.isfinite(preds).all(), (
            f"{name} returned non-finite predictions for rows with NaN features. "
            f"Non-finite count: {(~np.isfinite(preds)).sum()}"
        )

    def test_objective_zero_residual(self):
        """
        Validates objective gradient convergence at zero-residual states.

        Confirms the fundamental sanity check that perfectly accurate predictions 
        ($y_{pred} == y_{true}$) produce a gradient of zero, ensuring the optimization 
        engine halts adjustment for correct leaves.

        Args:
            None

        Returns:
            None
        """
        if not _OBJECTIVE_AVAILABLE:
            pytest.skip("weighted_symmetric_mae not importable")

        y = np.array([1.0, -1.0, 0.5, -0.5])
        grad, hess = weighted_symmetric_mae(y, y)  

        assert np.allclose(grad, 0.0, atol=1e-8), (
            f"Zero-residual gradient must be 0, got: {grad}"
        )
        assert (hess > 0).all(), "Hessian must remain positive at zero residual"

    def test_feature_selector_drop_low_variance(self):
        """
        Verifies that the FeatureSelector accurately prunes strictly constant columns.

        Args:
            None

        Returns:
            None
        """
        try:
            import importlib
            from quant_alpha.models.feature_selector import FeatureSelector
            import quant_alpha.models.feature_selector
            if hasattr(quant_alpha.models.feature_selector, "FeatureSelector") and \
               isinstance(quant_alpha.models.feature_selector.FeatureSelector, MagicMock):
                 del sys.modules["quant_alpha.models.feature_selector"]
            importlib.reload(quant_alpha.models.feature_selector)
        except ImportError:
            pytest.skip("FeatureSelector not importable")

        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "ticker": ["A"] * 100,
            "target": np.random.randn(100),
            "good": np.random.randn(100),
            "constant": [1.0] * 100,
        })
        
        selector = FeatureSelector(meta_cols=["date", "ticker", "target"])
        if hasattr(selector, "drop_low_variance"):
            df_out = selector.drop_low_variance(df)
            assert "good" in df_out.columns
            assert "constant" not in df_out.columns
            assert "date" in df_out.columns