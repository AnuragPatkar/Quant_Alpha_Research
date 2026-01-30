"""
Test Data Loading
=================
Tests for quant_alpha/data/loader.py
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


def generate_synthetic_ohlcv(n_days=100, n_tickers=5):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'][:n_tickers]
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    all_data = []
    for ticker in tickers:
        base_price = np.random.uniform(50, 500)
        returns = np.random.randn(n_days) * 0.02
        price = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': price * (1 + np.random.randn(n_days) * 0.005),
            'high': price * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            'low': price * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            'close': price,
            'volume': np.random.randint(1_000_000, 10_000_000, n_days)
        })
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# TESTS
# =============================================================================

def test_synthetic_data_validation():
    """Test synthetic data structure and validity."""
    print("\n" + "="*60)
    print("🧪 TEST: Synthetic Data Validation")
    print("="*60)
    
    result = TestResult()
    data = generate_synthetic_ohlcv()
    
    # Test 1: Required columns
    try:
        required = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in data.columns]
        assert len(missing) == 0, f"Missing: {missing}"
        result.success(f"Required columns present")
    except AssertionError as e:
        result.fail("Required columns", e)
    
    # Test 2: Not empty
    try:
        assert len(data) > 0, "Data is empty"
        result.success(f"Data not empty ({len(data)} rows)")
    except AssertionError as e:
        result.fail("Not empty", e)
    
    # Test 3: Multiple tickers
    try:
        n_tickers = data['ticker'].nunique()
        assert n_tickers == 5, f"Expected 5 tickers, got {n_tickers}"
        result.success(f"Multiple tickers ({n_tickers})")
    except AssertionError as e:
        result.fail("Multiple tickers", e)
    
    # Test 4: Positive prices
    try:
        for col in ['open', 'high', 'low', 'close']:
            assert (data[col] > 0).all(), f"{col} has non-positive values"
        result.success("All prices positive")
    except AssertionError as e:
        result.fail("Positive prices", e)
    
    # Test 5: High >= Low
    try:
        assert (data['high'] >= data['low']).all(), "High < Low found"
        result.success("High >= Low constraint")
    except AssertionError as e:
        result.fail("High >= Low", e)
    
    # Test 6: Positive volume
    try:
        assert (data['volume'] > 0).all(), "Non-positive volume found"
        result.success("Volume positive")
    except AssertionError as e:
        result.fail("Volume positive", e)
    
    # Test 7: No duplicates
    try:
        dups = data.duplicated(subset=['date', 'ticker']).sum()
        assert dups == 0, f"Found {dups} duplicates"
        result.success("No duplicate date-ticker pairs")
    except AssertionError as e:
        result.fail("No duplicates", e)
    
    # Test 8: No NaN in prices
    try:
        for col in ['open', 'high', 'low', 'close']:
            assert data[col].isna().sum() == 0, f"NaN in {col}"
        result.success("No NaN in prices")
    except AssertionError as e:
        result.fail("No NaN", e)
    
    return result.summary()


def test_dataloader_integration():
    """Test actual DataLoader class."""
    print("\n" + "="*60)
    print("🧪 TEST: DataLoader Integration")
    print("="*60)
    
    result = TestResult()
    
    # Test 1: Import
    try:
        from quant_alpha.data import DataLoader
        result.success("DataLoader imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    # Test 2: Instantiation
    try:
        loader = DataLoader()
        assert loader is not None
        result.success("DataLoader instantiated")
    except Exception as e:
        result.fail("Instantiation", e)
        return result.summary()
    
    # Test 3: Has load method
    try:
        assert hasattr(loader, 'load'), "Missing 'load' method"
        assert callable(loader.load), "'load' not callable"
        result.success("Has load() method")
    except AssertionError as e:
        result.fail("load() method", e)
    
    # Test 4: Load data
    try:
        data = loader.load()
        assert isinstance(data, pd.DataFrame), "Should return DataFrame"
        assert len(data) > 0, "DataFrame empty"
        result.success(f"Data loaded ({len(data):,} rows)")
    except FileNotFoundError:
        result.fail("Load data", "Data files not found (OK for quick test)")
        return result.summary()
    except Exception as e:
        result.fail("Load data", e)
        return result.summary()
    
    # Test 5: Required columns
    try:
        required = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in data.columns]
        assert len(missing) == 0, f"Missing: {missing}"
        result.success("Required columns present")
    except AssertionError as e:
        result.fail("Required columns", e)
    
    # Test 6: Date range
    try:
        date_min = pd.to_datetime(data['date']).min()
        date_max = pd.to_datetime(data['date']).max()
        result.success(f"Date range: {date_min.date()} to {date_max.date()}")
    except Exception as e:
        result.fail("Date range", e)
    
    # Test 7: Ticker count
    try:
        n_tickers = data['ticker'].nunique()
        assert n_tickers > 0
        result.success(f"Tickers: {n_tickers}")
    except AssertionError as e:
        result.fail("Ticker count", e)
    
    return result.summary()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 DATA LOADING TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    if not test_synthetic_data_validation():
        all_passed = False
    
    if not test_dataloader_integration():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL DATA LOADING TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)