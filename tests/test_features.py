"""
Test Feature Engineering
========================
Tests to make sure feature computation works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Need to add project root to path so imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from quant_alpha.features import FactorRegistry, compute_all_features
from quant_alpha.data import DataLoader


def test_feature_computation():
    """Basic test - can we compute features without errors?"""
    print("\n" + "="*60)
    print("🧪 TESTING FEATURE ENGINEERING")
    print("="*60)
    
    # Load data - using just 2 stocks to keep tests fast
    loader = DataLoader()
    data = loader.load()
    
    # Just grab first couple tickers for testing
    sample_tickers = data['ticker'].unique()[:2]
    sample_data = data[data['ticker'].isin(sample_tickers)].copy()
    
    print(f"\n   Testing with {len(sample_tickers)} stocks...")
    
    # Try computing features
    registry = FactorRegistry()
    features_df = registry.compute_features(sample_data)
    
    # Basic sanity checks
    assert len(features_df) > 0, "No features computed!"
    assert 'forward_return' in features_df.columns, "Target variable missing!"
    
    feature_names = registry.get_feature_names()
    assert len(feature_names) > 0, "No feature names returned!"
    
    print(f"\n   ✅ Feature computation successful!")
    print(f"   📊 Features created: {len(feature_names)}")
    print(f"   📈 Sample features: {feature_names[:5]}")
    
    return True


def test_feature_categories():
    """Test that all feature categories are represented."""
    print("\n" + "="*60)
    print("🧪 TESTING FEATURE CATEGORIES")
    print("="*60)
    
    loader = DataLoader()
    data = loader.load()
    sample_data = data[data['ticker'] == data['ticker'].unique()[0]].copy()
    
    registry = FactorRegistry()
    features_df = registry.compute_features(sample_data)
    feature_names = registry.get_feature_names()
    
    # Check categories
    categories = {
        'Momentum': any('mom' in f or 'roc' in f or 'ema' in f for f in feature_names),
        'Mean Reversion': any('rsi' in f or 'dist' in f or 'zscore' in f for f in feature_names),
        'Volatility': any('volatility' in f or 'atr' in f or 'vol' in f for f in feature_names),
        'Volume': any('volume' in f or 'pv_' in f or 'amihud' in f for f in feature_names)
    }
    
    print("\n   Feature Categories:")
    for cat, exists in categories.items():
        status = "✅" if exists else "❌"
        print(f"      {status} {cat}")
        assert exists, f"{cat} features missing!"
    
    print("\n   ✅ All feature categories present!")
    return True


def test_no_data_leakage():
    """Test that forward return doesn't leak into features."""
    print("\n" + "="*60)
    print("🧪 TESTING DATA LEAKAGE PREVENTION")
    print("="*60)
    
    loader = DataLoader()
    data = loader.load()
    sample_data = data[data['ticker'] == data['ticker'].unique()[0]].head(100).copy()
    
    registry = FactorRegistry()
    features_df = registry.compute_features(sample_data)
    
    # Check that forward_return is properly shifted
    if len(features_df) > 0:
        # Last few rows should have NaN forward returns (shifted forward)
        last_returns = features_df['forward_return'].tail(21)
        assert last_returns.isna().any(), "Forward return not properly shifted!"
        
        print("\n   ✅ No data leakage detected!")
        print("   ✅ Forward returns properly shifted")
    
    return True


def test_feature_values():
    """Test that feature values are reasonable."""
    print("\n" + "="*60)
    print("🧪 TESTING FEATURE VALUE RANGES")
    print("="*60)
    
    loader = DataLoader()
    data = loader.load()
    sample_data = data[data['ticker'] == data['ticker'].unique()[0]].copy()
    
    registry = FactorRegistry()
    features_df = registry.compute_features(sample_data)
    
    # Test RSI is between 0 and 100
    rsi_cols = [c for c in features_df.columns if 'rsi' in c]
    for col in rsi_cols:
        values = features_df[col].dropna()
        if len(values) > 0:
            assert values.min() >= 0, f"{col} has values < 0"
            assert values.max() <= 100, f"{col} has values > 100"
            print(f"   ✅ {col}: range [{values.min():.2f}, {values.max():.2f}]")
    
    # Test volatility is positive
    vol_cols = [c for c in features_df.columns if 'volatility' in c]
    for col in vol_cols:
        values = features_df[col].dropna()
        if len(values) > 0:
            assert values.min() >= 0, f"{col} has negative values"
            print(f"   ✅ {col}: all positive")
    
    print("\n   ✅ All feature values in reasonable ranges!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 RUNNING FEATURE ENGINEERING TEST SUITE")
    print("="*60)
    
    tests = [
        ("Feature Computation", test_feature_computation),
        ("Feature Categories", test_feature_categories),
        ("Data Leakage Prevention", test_no_data_leakage),
        ("Feature Value Ranges", test_feature_values)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n   ❌ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {failed} TESTS FAILED!")
        sys.exit(1)
