"""
Critical Path Performance and Throughput Benchmarks
===================================================
Measures the execution latency of performance-critical algorithmic components.

Purpose
-------
This module establishes computational bounds for the quantitative platform's 
heaviest sub-systems: Numba JIT-compiled feature engineering, the event-driven 
backtest engine throughput, and the quadratic programming solver latency during 
portfolio optimization. It enforces hard limits to prevent silent performance 
regressions that could cause pipeline stalls or unacceptable execution times.

Role in Quantitative Workflow
-----------------------------
Acts as an automated performance guardrail in the continuous integration (CI) 
pipeline, guaranteeing that the core processing engines scale linearly and 
meet institutional execution speed requirements prior to deployment.

Dependencies
------------
- **Pytest**: Orchestrates benchmark execution boundaries.
- **Pandas/NumPy**: In-memory synthesis of massive historical market matrices.
- **Time**: High-resolution clock access for strict threshold evaluation.
"""

import sys
import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from quant_alpha.backtest.engine import BacktestEngine
    from quant_alpha.optimization.allocator import PortfolioAllocator
    from quant_alpha.features.registry import FactorRegistry
    import quant_alpha.features.technical.volatility
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

@pytest.mark.skipif(not _IMPORTS_OK, reason="Quant Alpha modules not installed")
class TestPerformance:
    """
    Throughput validation suite for systemic data transformation and execution boundaries.
    """
    
    @pytest.fixture(scope="class")
    def large_market_data(self):
        """
        Provisions a synthetic high-density market dataset for benchmarking.

        Generates a continuous geometric random walk for 100 tickers over 1,000 
        trading days (100,000 total observations) to stress-test the vectorized 
        feature calculation pipeline.

        Args:
            None

        Returns:
            pd.DataFrame: A populated OHLCV historical data matrix.
        """
        n_tickers = 100
        n_days = 1000
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        tickers = [f"T{i:03d}" for i in range(n_tickers)]
        
        index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        df = pd.DataFrame(index=index).reset_index()
        
        rng = np.random.default_rng(42)
        
        # Simulates Geometric Brownian Motion (GBM) structural properties via vectorized random walk
        returns = rng.normal(0, 0.02, size=len(df))
        df["close"] = 100 * np.exp(returns.cumsum()) 
        
        df["open"] = df["close"]
        df["high"] = df["close"] * 1.01
        df["low"] = df["close"] * 0.99
        df["volume"] = rng.integers(1000, 1000000, size=len(df)).astype(float)
        
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    def test_feature_engineering_speed(self, large_market_data):
        """
        Benchmarks the throughput of Numba JIT-compiled feature engineering.

        Executes a warmup sequence to isolate compilation overhead, then measures 
        pure runtime execution latency to ensure the pipeline meets high-frequency 
        vectorization standards.

        Args:
            large_market_data (pd.DataFrame): The synthetic 100,000-row benchmark dataset.

        Returns:
            None
        """
        registry = FactorRegistry()
        if "volatility_21d" not in registry.factors:
            pytest.skip("volatility_21d not registered")
            
        factor = registry.factors["volatility_21d"]
        
        # Triggers initial JIT trace compilation to prevent cold-start latency from skewing the benchmark
        warmup_data = large_market_data.iloc[:1000].copy()
        _ = factor.calculate(warmup_data)
        
        start = time.time()
        _ = factor.calculate(large_market_data)
        elapsed = time.time() - start
        
        print(f"\nFeature Calc (100k rows): {elapsed:.4f}s")
        
        # Evaluates structural execution bound; 100k rows must resolve under 1.0s via JIT
        assert elapsed < 1.0, f"Feature calculation too slow: {elapsed:.4f}s"

    def test_backtest_throughput(self):
        """
        Benchmarks the event loop throughput of the historical BacktestEngine.

        Evaluates the system's capacity to process daily signal ingestion, 
        order allocation, and equity curve P&L accounting across a dense 
        multi-asset universe.

        Args:
            None

        Returns:
            None
        """
        n_days = 252 # 1 year
        n_tickers = 50
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        
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
        Benchmarks the convergence latency of the Mean-Variance optimization solver.

        Generates a synthetic positive semi-definite covariance matrix to evaluate 
        the computational overhead of the quadratic programming (QP) layer during 
        capital allocation.

        Args:
            None

        Returns:
            None
        """
        n_assets = 50
        tickers = [f"T{i}" for i in range(n_assets)]
        
        # Synthesis of robust semi-definite covariance bounds
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_assets, 100))
        cov = pd.DataFrame(np.cov(X), index=tickers, columns=tickers)
        
        er = {t: 0.05 for t in tickers}
        
        allocator = PortfolioAllocator(method="mean_variance")
        
        start = time.time()
        _ = allocator.allocate(er, cov)
        elapsed = time.time() - start
        
        print(f"\nOptimization (50 assets): {elapsed:.4f}s")
        assert elapsed < 0.5, f"Optimization too slow: {elapsed:.4f}s"
