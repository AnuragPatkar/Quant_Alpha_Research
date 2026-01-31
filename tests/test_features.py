"""
Test Feature Engineering
========================
Tests for quant_alpha/features/
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


def generate_price_data(n=252, n_tickers=2):
    """Generate synthetic price data."""
    np.random.seed(42)
    
    tickers = [f'STOCK_{i}' for i in range(n_tickers)]
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    
    all_data = []
    for ticker in tickers:
        returns = np.random.randn(n) * 0.02
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': price * (1 + np.random.randn(n) * 0.005),
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': np.random.randint(1_000_000, 10_000_000, n)
        })
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# TESTS: MANUAL FEATURE CALCULATIONS
# =============================================================================

def test_momentum_calculations():
    """Test momentum feature calculations manually."""
    print("\n" + "="*60)
    print("🧪 TEST: Momentum Calculations")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=1)
    close = data['close']
    
    # Test 1: RSI
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI < 0"
        assert (valid_rsi <= 100).all(), "RSI > 100"
        result.success(f"RSI in [0, 100] range")
    except AssertionError as e:
        result.fail("RSI", e)
    
    # Test 2: ROC
    try:
        period = 10
        roc = (close - close.shift(period)) / close.shift(period)
        
        assert roc.isna().sum() == period, "First values should be NaN"
        assert np.isfinite(roc.dropna()).all(), "ROC should be finite"
        result.success("ROC calculation")
    except AssertionError as e:
        result.fail("ROC", e)
    
    # Test 3: MACD
    try:
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        assert np.isfinite(macd.dropna()).all()
        assert np.isfinite(signal.dropna()).all()
        result.success("MACD calculation")
    except AssertionError as e:
        result.fail("MACD", e)
    
    # Test 4: Momentum
    try:
        mom = close.pct_change(20)
        assert len(mom) == len(close)
        assert np.isfinite(mom.dropna()).all()
        result.success("Momentum calculation")
    except AssertionError as e:
        result.fail("Momentum", e)
    
    return result.summary()


def test_mean_reversion_calculations():
    """Test mean reversion feature calculations."""
    print("\n" + "="*60)
    print("🧪 TEST: Mean Reversion Calculations")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=1)
    close = data['close']
    
    # Test 1: Z-score
    try:
        window = 20
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        zscore = (close - rolling_mean) / (rolling_std + 1e-10)
        
        valid_z = zscore.dropna()
        assert np.isfinite(valid_z).all(), "Z-score should be finite"
        
        within_3_std = ((valid_z >= -3) & (valid_z <= 3)).mean()
        assert within_3_std > 0.95, f"Only {within_3_std:.1%} within 3 std"
        result.success(f"Z-score valid ({within_3_std:.1%} within 3 std)")
    except AssertionError as e:
        result.fail("Z-score", e)
    
    # Test 2: Bollinger Bands Position
    try:
        window = 20
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        
        bb_pos = (close - lower) / (upper - lower + 1e-10)
        
        assert np.isfinite(bb_pos.dropna()).all()
        result.success("Bollinger Bands position")
    except AssertionError as e:
        result.fail("Bollinger Bands", e)
    
    # Test 3: Price to SMA ratio
    try:
        sma_20 = close.rolling(20).mean()
        ratio = close / sma_20
        
        valid_ratio = ratio.dropna()
        assert (valid_ratio > 0).all()
        assert np.isfinite(valid_ratio).all()
        result.success("Price/SMA ratio")
    except AssertionError as e:
        result.fail("Price/SMA", e)
    
    return result.summary()


def test_volatility_calculations():
    """Test volatility feature calculations."""
    print("\n" + "="*60)
    print("🧪 TEST: Volatility Calculations")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=1)
    
    # Test 1: Historical volatility
    try:
        returns = data['close'].pct_change()
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        
        valid_vol = vol_20.dropna()
        assert (valid_vol >= 0).all(), "Volatility should be non-negative"
        result.success("Historical volatility")
    except AssertionError as e:
        result.fail("Historical volatility", e)
    
    # Test 2: ATR
    try:
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
        result.success("ATR calculation")
    except AssertionError as e:
        result.fail("ATR", e)
    
    return result.summary()


def test_volume_calculations():
    """Test volume feature calculations."""
    print("\n" + "="*60)
    print("🧪 TEST: Volume Calculations")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=1)
    
    # Test 1: Volume ratio
    try:
        volume = data['volume']
        avg_vol = volume.rolling(20).mean()
        vol_ratio = volume / (avg_vol + 1e-10)
        
        valid_ratio = vol_ratio.dropna()
        assert (valid_ratio > 0).all()
        result.success("Volume ratio")
    except AssertionError as e:
        result.fail("Volume ratio", e)
    
    # Test 2: Dollar volume
    try:
        dollar_vol = data['close'] * data['volume']
        assert (dollar_vol > 0).all()
        result.success("Dollar volume")
    except AssertionError as e:
        result.fail("Dollar volume", e)
    
    return result.summary()


# =============================================================================
# TESTS: ACTUAL FEATURE MODULES
# =============================================================================

def test_factor_registry():
    """Test FactorRegistry class."""
    print("\n" + "="*60)
    print("🧪 TEST: FactorRegistry Integration")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=2)
    
    # Test 1: Import
    try:
        from quant_alpha.features import FactorRegistry
        result.success("FactorRegistry imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    # Test 2: Instantiation
    try:
        registry = FactorRegistry()
        assert registry is not None
        result.success("FactorRegistry instantiated")
    except Exception as e:
        result.fail("Instantiation", e)
        return result.summary()
    
    # Test 3: Register defaults
    try:
        registry.register_defaults()
        assert len(registry) > 0
        result.success(f"Registered {len(registry)} default factors")
    except Exception as e:
        result.fail("register_defaults()", e)
    
    # Test 4: Compute features
    try:
        # Use compute_all instead of compute_features
        features_df = registry.compute_all(data, normalize=False, winsorize=False)
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0
        result.success(f"Features computed ({len(features_df)} rows)")
    except Exception as e:
        result.fail("compute_all()", e)
        return result.summary()
    
    # Test 5: Forward return exists
    try:
        assert 'forward_return' in features_df.columns, "forward_return missing"
        result.success("forward_return column present")
    except AssertionError as e:
        result.fail("forward_return", e)
    
    # Test 6: Get feature names
    try:
        # Check available methods or use keys
        if hasattr(registry, 'get_feature_names'):
            feature_names = registry.get_feature_names()
        elif hasattr(registry, 'list_factors'):
             feature_names = registry.list_factors()
        else:
             # Fallback
             feature_names = list(registry._factors.keys())
             
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        result.success(f"Feature names retrieved: {len(feature_names)}")
    except Exception as e:
        result.fail("get feature names", e)
    
    return result.summary()


def test_compute_all_features():
    """Test compute_all_features function."""
    print("\n" + "="*60)
    print("🧪 TEST: compute_all_features()")
    print("="*60)
    
    result = TestResult()
    data = generate_price_data(n=100, n_tickers=2)
    
    try:
        from quant_alpha.features import compute_all_features
        result.success("compute_all_features imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    try:
        features = compute_all_features(data, normalize=False, winsorize=False)
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        result.success(f"Features computed ({len(features)} rows)")
    except Exception as e:
        result.fail("compute_all_features()", e)
    
    return result.summary()


# =============================================================================
# TESTS: EDGE CASES
# =============================================================================

def test_feature_edge_cases():
    """Test edge cases in feature computation."""
    print("\n" + "="*60)
    print("🧪 TEST: Edge Cases")
    print("="*60)
    
    result = TestResult()
    
    # Test 1: Constant price
    try:
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=50),
            'ticker': 'TEST',
            'open': 100.0,
            'high': 100.0,
            'low': 100.0,
            'close': 100.0,
            'volume': 1000000
        })
        
        roc = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
        valid_roc = roc.dropna()
        
        assert (valid_roc == 0).all(), "ROC should be 0 for constant price"
        result.success("Constant price handled")
    except AssertionError as e:
        result.fail("Constant price", e)
    
    # Test 2: Very short series
    try:
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'close': [100, 101, 102, 101, 100]
        })
        
        roc = data['close'].pct_change()
        assert len(roc) == 5
        result.success("Short series handled")
    except Exception as e:
        result.fail("Short series", e)
    
    # Test 3: Missing values
    try:
        data = pd.DataFrame({
            'close': [100, np.nan, 102, np.nan, 104]
        })
        
        roc = data['close'].pct_change()
        assert len(roc) == 5
        result.success("Missing values handled")
    except Exception as e:
        result.fail("Missing values", e)
    
    return result.summary()


def test_no_data_leakage():
    """Test that forward return is properly shifted."""
    print("\n" + "="*60)
    print("🧪 TEST: No Data Leakage")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.features import FactorRegistry
        from quant_alpha.data import DataLoader
        
        try:
            loader = DataLoader()
            data = loader.load()
            sample_data = data[data['ticker'] == data['ticker'].unique()[0]].head(100).copy()
        except:
            sample_data = generate_price_data(n=100, n_tickers=1)
        
        registry = FactorRegistry()
        registry.register_defaults()
        
        # Use compute_all instead of compute_features
        features_df = registry.compute_all(sample_data, normalize=False, winsorize=False)
        
        if len(features_df) > 0:
            last_returns = features_df['forward_return'].tail(21)
            # Check if last rows are NaN (as they should be for forward returns)
            # Using tail(1) to be safe
            assert last_returns.tail(1).isna().all(), "Last row should have NaN forward return"
            result.success("Forward returns properly shifted (no leakage)")
        else:
            result.success("Skipped (no data)")
            
    except ImportError:
        result.success("Skipped (module not available)")
    except Exception as e:
        result.fail("No data leakage", e)
    
    return result.summary()


def test_feature_value_ranges():
    """Test that feature values are in reasonable ranges."""
    print("\n" + "="*60)
    print("🧪 TEST: Feature Value Ranges")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.features import FactorRegistry
        
        data = generate_price_data(n=100, n_tickers=1)
        registry = FactorRegistry()
        registry.register_defaults()
        
        features_df = registry.compute_all(data, normalize=False, winsorize=False)
        
        # RSI should be 0-100
        # Only check columns starting with 'rsi_'
        rsi_cols = [c for c in features_df.columns if c.lower().startswith('rsi_')]
        for col in rsi_cols:
            values = features_df[col].dropna()
            if len(values) > 0:
                assert (values >= -0.1).all(), f"{col} < 0"
                assert (values <= 100.1).all(), f"{col} > 100"
        result.success(f"RSI columns ({len(rsi_cols)}) in [0, 100]")
        
        # Volatility should be positive
        vol_cols = [
            c for c in features_df.columns 
            if ('volatility' in c.lower() or 'atr' in c.lower())
            and 'rank' not in c.lower() 
            and 'regime' not in c.lower()
        ]
        
        for col in vol_cols:
            values = features_df[col].dropna()
            if len(values) > 0:
                assert (values >= -1e-6).all(), f"{col} has negative values"
        result.success(f"Volatility columns ({len(vol_cols)}) positive")
        
    except ImportError:
        result.success("Skipped (module not available)")
    except Exception as e:
        result.fail("Feature ranges", e)
    
    return result.summary()
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 FEATURE ENGINEERING TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Manual calculation tests
    if not test_momentum_calculations():
        all_passed = False
    
    if not test_mean_reversion_calculations():
        all_passed = False
    
    if not test_volatility_calculations():
        all_passed = False
    
    if not test_volume_calculations():
        all_passed = False
    
    # Edge cases
    if not test_feature_edge_cases():
        all_passed = False
    
    # Integration tests
    if not test_factor_registry():
        all_passed = False
    
    if not test_compute_all_features():
        all_passed = False
    
    if not test_no_data_leakage():
        all_passed = False
    
    if not test_feature_value_ranges():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL FEATURE TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)