"""
Test Model Training and Prediction
===================================
Tests for the LightGBM model - making sure training and predictions work.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root so we can import our modules
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from quant_alpha.models.boosting import LightGBMModel
from quant_alpha.features import compute_all_features
from quant_alpha.data import DataLoader


def test_model_initialization():
    """Can we create a model without errors?"""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL INITIALIZATION")
    print("="*60)
    
    # Try creating a simple model
    feature_names = ['mom_5d', 'mom_10d', 'volatility_10d']
    model = LightGBMModel(feature_names)
    
    # Basic checks
    assert model.feature_names == feature_names
    assert not model.is_fitted
    assert model.model is None
    
    print("\n   ✅ Model initialized successfully!")
    return True


def test_model_training():
    """Test if we can train the model."""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL TRAINING")
    print("="*60)
    
    # Create some random data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    # Try training
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    assert model.is_fitted
    assert model.model is not None
    
    print("\n   ✅ Model trained successfully!")
    print(f"   📊 Training samples: {len(X_train)}")
    print(f"   📈 Features: {len(feature_names)}")
    
    return True


def test_model_prediction():
    """Test model can make predictions."""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL PREDICTION")
    print("="*60)
    
    # Create and train model
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    # Make predictions
    X_test = pd.DataFrame(
        np.random.randn(100, n_features),
        columns=feature_names
    )
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert not np.isnan(predictions).all()
    
    print("\n   ✅ Predictions generated successfully!")
    print(f"   📊 Test samples: {len(X_test)}")
    print(f"   📈 Predictions: {len(predictions)}")
    print(f"   📉 Mean prediction: {predictions.mean():.4f}")
    
    return True


def test_model_evaluation():
    """Test model evaluation metrics."""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL EVALUATION")
    print("="*60)
    
    # Create and train model
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    X_test = pd.DataFrame(
        np.random.randn(100, n_features),
        columns=feature_names
    )
    y_test = pd.Series(np.random.randn(100))
    
    metrics = model.evaluate(X_test, y_test)
    
    # Check metrics exist
    assert 'ic' in metrics
    assert 'rank_ic' in metrics
    assert 'hit_rate' in metrics
    assert 'rmse' in metrics
    
    print("\n   ✅ Evaluation metrics computed!")
    print(f"   📊 IC: {metrics['ic']:.4f}")
    print(f"   📈 Rank IC: {metrics['rank_ic']:.4f}")
    print(f"   🎯 Hit Rate: {metrics['hit_rate']:.1%}")
    
    return True


def test_feature_importance():
    """Test feature importance extraction."""
    print("\n" + "="*60)
    print("🧪 TESTING FEATURE IMPORTANCE")
    print("="*60)
    
    # Create and train model
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    # Get importance
    importance_df = model.get_feature_importance()
    
    assert len(importance_df) == n_features
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert 'importance_pct' in importance_df.columns
    
    print("\n   ✅ Feature importance extracted!")
    print(f"\n   Top 3 Features:")
    for i, row in importance_df.head(3).iterrows():
        print(f"      {i+1}. {row['feature']}: {row['importance_pct']:.2f}%")
    
    return True


def test_model_save_load():
    """Test model persistence."""
    print("\n" + "="*60)
    print("🧪 TESTING MODEL SAVE/LOAD")
    print("="*60)
    
    # Create and train model
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    model = LightGBMModel(feature_names)
    model.fit(X_train, y_train, verbose=False)
    
    # Save model
    save_path = ROOT / 'test_outputs' / 'test_model.pkl'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    
    assert save_path.exists()
    
    # Load model
    loaded_model = LightGBMModel.load(str(save_path))
    
    assert loaded_model.is_fitted
    assert loaded_model.feature_names == feature_names
    
    # Test predictions match
    X_test = pd.DataFrame(
        np.random.randn(10, n_features),
        columns=feature_names
    )
    pred_original = model.predict(X_test)
    pred_loaded = loaded_model.predict(X_test)
    
    assert np.allclose(pred_original, pred_loaded)
    
    print("\n   ✅ Model saved and loaded successfully!")
    print(f"   💾 File: {save_path}")
    
    # Cleanup
    save_path.unlink()
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RUNNING MODEL TEST SUITE")
    print("="*60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Model Training", test_model_training),
        ("Model Prediction", test_model_prediction),
        ("Model Evaluation", test_model_evaluation),
        ("Feature Importance", test_feature_importance),
        ("Model Save/Load", test_model_save_load)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n   ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TESTS FAILED!")
        sys.exit(1)
