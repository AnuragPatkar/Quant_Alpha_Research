"""
PERFORMANCE TEST: Critical Path Benchmarks
==========================================
Measures execution time of performance-critical components.
Fails if execution time exceeds defined thresholds.

Benchmarks:
  1. Feature Engineering (Numba JIT vs Cold)
  2. Backtest Engine (Event Loop throughput)
  3. Portfolio Optimization (Solver overhead)
"""

import sys
import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports (Guarded)
# ---------------------------------------------------------------------------
try:
    from quant_alpha.backtest.engine import BacktestEngine
    from quant_alpha.optimization.allocator import PortfolioAllocator
    from quant_alpha.features.registry import FactorRegistry
    # Ensure feature modules are loaded to populate registry
    import quant_alpha.features.technical.volatility
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

@pytest.mark.skipif(not _IMPORTS_OK, reason="Quant Alpha modules not installed")
class TestPerformance:
    
    @pytest.fixture(scope="class")
    def large_market_data(self):
        """
        Generate synthetic market data for benchmarking.
        100 tickers x 1000 days = 100,000 rows.
        """
        n_tickers = 100
        n_days = 1000
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        tickers = [f"T{i:03d}" for i in range(n_tickers)]
        
        # Create DataFrame
        index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=index).reset_index()
        
        rng = np.random.default_rng(42)
        # Vectorized price generation
        returns = rng.normal(0, 0.02, size=len(df))
        df["close"] = 100 * np.exp(returns.cumsum()) # Random walk (not grouped but fine for speed test)
        
        df["open"] = df["close"]
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["volume"] = rng.integers(1000, 1000000, size=len(df)).astype(float)
        
        # Sort for features
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    def test_feature_engineering_speed(self, large_market_data):
        """
        Benchmark Numba-accelerated feature calculation.
        """
        registry = FactorRegistry()
        if "volatility_21d" not in registry.factors:
            pytest.skip("volatility_21d not registered")
            
        factor = registry.factors["volatility_21d"]
        
        # Warmup (compile JIT)
        warmup_data = large_market_data.iloc[:1000].copy()
        _ = factor.calculate(warmup_data)
        
        # Benchmark
        start = time.time()
        _ = factor.calculate(large_market_data)
        elapsed = time.time() - start
        
        # 100k rows should be very fast with Numba (< 0.5s)
        # Adjust threshold based on typical CI environment
        print(f"\nFeature Calc (100k rows): {elapsed:.4f}s")
        assert elapsed < 1.0, f"Feature calculation too slow: {elapsed:.4f}s"

    def test_backtest_throughput(self):
        """
        Benchmark BacktestEngine event loop.
        """
        n_days = 252 # 1 year
        n_tickers = 50
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        
        # Generate data
        records = []
        for d in dates:
            for i in range(n_tickers):
                records.append({
                    "date": d,
                    "ticker": f"T{i}",
                    "close": 100.0,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 1e6,
                    "volatility": 0.01
                })
        prices = pd.DataFrame(records)
        
        preds = prices[["date", "ticker"]].copy()
        preds["prediction"] = np.random.random(len(preds))
        
        engine = BacktestEngine(initial_capital=1_000_000)
        
        start = time.time()
        engine.run(preds, prices, top_n=10)
        elapsed = time.time() - start
        
        print(f"\nBacktest (1 year, 50 tickers): {elapsed:.4f}s")
        assert elapsed < 3.0, f"Backtest too slow: {elapsed:.4f}s"

    def test_optimization_latency(self):
        """
        Benchmark Mean-Variance optimization latency.
        """
        n_assets = 50
        tickers = [f"T{i}" for i in range(n_assets)]
        
        # Synthetic covariance
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_assets, 100))
        cov = pd.DataFrame(np.cov(X), index=tickers, columns=tickers)
        
        # Expected returns
        er = {t: 0.05 for t in tickers}
        
        allocator = PortfolioAllocator(method="mean_variance")
        
        start = time.time()
        _ = allocator.allocate(er, cov)
        elapsed = time.time() - start
        
        print(f"\nOptimization (50 assets): {elapsed:.4f}s")
        assert elapsed < 0.5, f"Optimization too slow: {elapsed:.4f}s"
