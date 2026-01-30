"""
Test Model Training and Prediction
===================================
Tests for quant_alpha/models/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def success(self, name):
        self.passed += 1
        print(f"   ✅ {name}")
    
    def fail(self, name, error):
        self.failed += 1
        self.errors.append((name, str(error)))
        print(f"   ❌ {name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n   Results: {self.passed}/{total} passed")
        return self.failed == 0


def generate_training_data(n_samples=500, n_features=5, seed=42):
    """Generate synthetic training data with signal."""
    np.random.seed(seed)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    
    # Target with signal
    y = (0.3 * X['feature_0'] 
         - 0.2 * X['feature_1'] 
         + 0.1 * X['feature_2']
         + np.random.randn(n_samples) * 0.5)
    
    return X, pd.Series(y), feature_names


# =============================================================================
# TESTS
# =============================================================================

def test_model_initialization():
    """Test model initialization."""
    print("\n" + "="*60)
    print("🧪 TEST: Model Initialization")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
        result.success("LightGBMModel imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    # Test 1: Basic init
    try:
        feature_names = ['mom_5d', 'mom_10d', 'volatility_10d']
        model = LightGBMModel(feature_names)
        
        assert model.feature_names == feature_names
        assert not model.is_fitted
        assert model.model is None
        result.success("Basic initialization")
    except Exception as e:
        result.fail("Basic init", e)
    
    # Test 2: Single feature
    try:
        model = LightGBMModel(['single_feature'])
        assert len(model.feature_names) == 1
        result.success("Single feature init")
    except Exception as e:
        result.fail("Single feature", e)
    
    return result.summary()


def test_model_training():
    """Test model training."""
    print("\n" + "="*60)
    print("🧪 TEST: Model Training")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=1000)
    
    try:
        model = LightGBMModel(feature_names)
        model.fit(X_train, y_train, verbose=False)
        
        assert model.is_fitted, "Model should be fitted"
        assert model.model is not None, "Model object should exist"
        result.success(f"Training completed ({len(X_train)} samples)")
    except Exception as e:
        result.fail("Training", e)
    
    return result.summary()


def test_model_prediction():
    """Test model prediction."""
    print("\n" + "="*60)
    print("🧪 TEST: Model Prediction")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=500)
    X_test, _, _ = generate_training_data(n_samples=100, seed=99)
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    # Test 1: Basic prediction
    try:
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test), "Length mismatch"
        assert not np.isnan(predictions).all(), "All NaN"
        assert np.isfinite(predictions).all(), "Contains inf"
        result.success(f"Predictions generated ({len(predictions)} values)")
    except Exception as e:
        result.fail("Prediction", e)
    
    # Test 2: Single sample
    try:
        single_pred = model.predict(X_test.iloc[[0]])
        assert len(single_pred) == 1
        result.success("Single sample prediction")
    except Exception as e:
        result.fail("Single sample", e)
    
    # Test 3: Predictions have variance
    try:
        assert predictions.std() > 0, "Predictions are constant"
        result.success("Predictions have variance")
    except Exception as e:
        result.fail("Variance", e)
    
    return result.summary()


def test_model_evaluation():
    """Test model evaluation metrics."""
    print("\n" + "="*60)
    print("🧪 TEST: Model Evaluation")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=500)
    X_test, y_test, _ = generate_training_data(n_samples=100, seed=99)
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    try:
        metrics = model.evaluate(X_test, y_test)
        
        required = ['ic', 'rank_ic', 'hit_rate', 'rmse']
        for m in required:
            assert m in metrics, f"Missing: {m}"
        result.success("All metrics present")
        
        assert -1 <= metrics['ic'] <= 1, f"IC out of range"
        assert -1 <= metrics['rank_ic'] <= 1, f"Rank IC out of range"
        assert 0 <= metrics['hit_rate'] <= 1, f"Hit rate out of range"
        assert metrics['rmse'] >= 0, f"RMSE negative"
        result.success("Metric values valid")
        
        print(f"      IC: {metrics['ic']:.4f}")
        print(f"      Rank IC: {metrics['rank_ic']:.4f}")
        print(f"      Hit Rate: {metrics['hit_rate']:.1%}")
        print(f"      RMSE: {metrics['rmse']:.4f}")
        
    except Exception as e:
        result.fail("Evaluation", e)
    
    return result.summary()


def test_feature_importance():
    """Test feature importance extraction."""
    print("\n" + "="*60)
    print("🧪 TEST: Feature Importance")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=500)
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    try:
        importance_df = model.get_feature_importance()
        
        assert len(importance_df) == len(feature_names), "Count mismatch"
        result.success(f"Importance for {len(importance_df)} features")
        
        required_cols = ['feature', 'importance', 'importance_pct']
        for col in required_cols:
            assert col in importance_df.columns, f"Missing: {col}"
        result.success("Required columns present")
        
        total_pct = importance_df['importance_pct'].sum()
        assert 99 <= total_pct <= 101, f"Percentages sum to {total_pct}"
        result.success("Percentages sum correctly")
        
        assert (importance_df['importance'] >= 0).all(), "Negative importance"
        result.success("All importances non-negative")
        
        print(f"\n      Top 3 Features:")
        for i, row in importance_df.head(3).iterrows():
            print(f"         {row['feature']}: {row['importance_pct']:.2f}%")
        
    except Exception as e:
        result.fail("Feature importance", e)
    
    return result.summary()


def test_model_save_load():
    """Test model save/load."""
    print("\n" + "="*60)
    print("🧪 TEST: Model Save/Load")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=500)
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    save_path = ROOT / 'test_outputs' / 'test_model.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save
        model.save(str(save_path))
        assert save_path.exists(), "File not created"
        result.success(f"Model saved ({save_path.stat().st_size / 1024:.1f} KB)")
        
        # Load
        loaded_model = LightGBMModel.load(str(save_path))
        assert loaded_model.is_fitted, "Loaded model not fitted"
        assert loaded_model.feature_names == feature_names, "Feature names mismatch"
        result.success("Model loaded")
        
        # Predictions match
        X_test, _, _ = generate_training_data(n_samples=10, seed=99)
        pred_original = model.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        
        assert np.allclose(pred_original, pred_loaded), "Predictions don't match"
        result.success("Predictions match")
        
        # Cleanup
        save_path.unlink()
        result.success("Cleanup done")
        
    except Exception as e:
        result.fail("Save/Load", e)
        if save_path.exists():
            save_path.unlink()
    
    return result.summary()


def test_model_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("🧪 TEST: Edge Cases")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    feature_names = ['f1', 'f2', 'f3']
    
    # Test 1: Predict before fit
    try:
        model = LightGBMModel(feature_names)
        X_test = pd.DataFrame(np.random.randn(10, 3), columns=feature_names)
        model.predict(X_test)
        result.fail("Predict before fit", "Should raise error")
    except Exception:
        result.success("Predict before fit raises error")
    
    # Test 2: Wrong columns
    try:
        X_train, y_train, _ = generate_training_data(n_samples=100, n_features=3)
        model = LightGBMModel(feature_names)
        model.fit(X_train, y_train, verbose=False)
        
        X_wrong = pd.DataFrame(np.random.randn(10, 3), columns=['wrong1', 'wrong2', 'wrong3'])
        model.predict(X_wrong)
        result.fail("Wrong columns", "Should raise error")
    except Exception:
        result.success("Wrong columns raises error")
    
    # Test 3: NaN in features (LightGBM handles this)
    try:
        X_nan = pd.DataFrame(np.random.randn(100, 3), columns=feature_names)
        X_nan.iloc[10:20, 0] = np.nan
        y_nan = pd.Series(np.random.randn(100))
        
        model = LightGBMModel(feature_names)
        model.fit(X_nan, y_nan, verbose=False)
        result.success("NaN in features handled")
    except Exception as e:
        result.fail("NaN handling", e)
    
    # Test 4: Constant target
    try:
        X_const, _, _ = generate_training_data(n_samples=100, n_features=3)
        y_const = pd.Series([1.0] * 100)
        
        model = LightGBMModel(feature_names)
        model.fit(X_const, y_const, verbose=False)
        preds = model.predict(X_const)
        
        assert preds.std() < 0.01, "Predictions should be ~constant"
        result.success("Constant target handled")
    except Exception as e:
        result.fail("Constant target", e)
    
    return result.summary()


def test_prediction_speed():
    """Test prediction speed."""
    print("\n" + "="*60)
    print("🧪 TEST: Prediction Speed")
    print("="*60)
    
    result = TestResult()
    import time
    
    try:
        from quant_alpha.models.boosting import LightGBMModel
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    X_train, y_train, feature_names = generate_training_data(n_samples=5000, n_features=20)
    
    model = LightGBMModel(feature_names)
    
    # Time training
    start = time.time()
    model.fit(X_train, y_train, verbose=False)
    train_time = time.time() - start
    result.success(f"Training: {train_time:.2f}s")
    
    # Time prediction
    X_test, _, _ = generate_training_data(n_samples=10000, n_features=20, seed=99)
    
    start = time.time()
    predictions = model.predict(X_test)
    pred_time = time.time() - start
    
    speed = len(X_test) / pred_time
    result.success(f"Prediction: {pred_time:.3f}s ({speed:.0f} samples/sec)")
    
    assert pred_time < 5.0, f"Too slow: {pred_time}s"
    result.success("Speed acceptable")
    
    return result.summary()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MODEL TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    tests = [
        test_model_initialization,
        test_model_training,
        test_model_prediction,
        test_model_evaluation,
        test_feature_importance,
        test_model_save_load,
        test_model_edge_cases,
        test_prediction_speed,
    ]
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL MODEL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)