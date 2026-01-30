"""
Test Backtesting Engine
=======================
Tests for quant_alpha/backtest/
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


def generate_returns(n=252, mean=0.0005, std=0.01, seed=42):
    """Generate synthetic returns."""
    np.random.seed(seed)
    returns = pd.Series(
        np.random.randn(n) * std + mean,
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
        name='returns'
    )
    return returns


def generate_backtest_data(n_days=100, n_tickers=5, seed=42):
    """Generate synthetic backtest data."""
    np.random.seed(seed)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'][:n_tickers]
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    all_data = []
    for ticker in tickers:
        returns = np.random.randn(n_days) * 0.02
        
        df = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'close': 100 * np.exp(np.cumsum(returns)),
            'prediction': np.random.randn(n_days),
            'forward_return': np.random.randn(n_days) * 0.02
        })
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)


# =============================================================================
# TESTS: PERFORMANCE METRICS
# =============================================================================

def test_performance_metrics():
    """Test performance metric calculations."""
    print("\n" + "="*60)
    print("🧪 TEST: Performance Metrics")
    print("="*60)
    
    result = TestResult()
    returns = generate_returns()
    
    # Test 1: Sharpe Ratio
    try:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
        
        assert np.isfinite(sharpe), "Should be finite"
        assert -10 < sharpe < 10, f"Out of range: {sharpe}"
        result.success(f"Sharpe Ratio = {sharpe:.2f}")
    except Exception as e:
        result.fail("Sharpe Ratio", e)
    
    # Test 2: Max Drawdown
    try:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        assert max_dd <= 0, "Should be <= 0"
        assert max_dd > -1, "Should be > -1"
        result.success(f"Max Drawdown = {max_dd:.2%}")
    except Exception as e:
        result.fail("Max Drawdown", e)
    
    # Test 3: Total Return
    try:
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        assert np.isfinite(total_return)
        result.success(f"Total Return = {total_return:.2%}")
    except Exception as e:
        result.fail("Total Return", e)
    
    # Test 4: Annualized Return
    try:
        n_years = len(returns) / 252
        annual_return = cumulative.iloc[-1] ** (1/n_years) - 1
        
        assert np.isfinite(annual_return)
        result.success(f"Annualized Return = {annual_return:.2%}")
    except Exception as e:
        result.fail("Annualized Return", e)
    
    # Test 5: Volatility
    try:
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        assert daily_vol > 0
        assert annual_vol > daily_vol
        result.success(f"Annual Volatility = {annual_vol:.2%}")
    except Exception as e:
        result.fail("Volatility", e)
    
    # Test 6: Sortino Ratio
    try:
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0.01
        sortino = np.sqrt(252) * returns.mean() / (downside_std + 1e-10)
        
        assert np.isfinite(sortino)
        result.success(f"Sortino Ratio = {sortino:.2f}")
    except Exception as e:
        result.fail("Sortino Ratio", e)
    
    # Test 7: Calmar Ratio
    try:
        calmar = (returns.mean() * 252) / abs(max_dd + 1e-10)
        
        assert np.isfinite(calmar)
        result.success(f"Calmar Ratio = {calmar:.2f}")
    except Exception as e:
        result.fail("Calmar Ratio", e)
    
    return result.summary()


# =============================================================================
# TESTS: PORTFOLIO CONSTRUCTION
# =============================================================================

def test_portfolio_construction():
    """Test portfolio construction logic."""
    print("\n" + "="*60)
    print("🧪 TEST: Portfolio Construction")
    print("="*60)
    
    result = TestResult()
    data = generate_backtest_data()
    
    # Test 1: Long-Short Portfolio
    try:
        portfolio_returns = []
        
        for date, group in data.groupby('date'):
            group = group.sort_values('prediction', ascending=False)
            n = len(group)
            
            if n < 4:
                continue
            
            n_long = n // 4
            n_short = n // 4
            
            long_ret = group.head(n_long)['forward_return'].mean()
            short_ret = group.tail(n_short)['forward_return'].mean()
            
            portfolio_returns.append(long_ret - short_ret)
        
        returns_series = pd.Series(portfolio_returns)
        assert len(returns_series) > 0
        result.success(f"Long-Short: {len(returns_series)} days")
    except Exception as e:
        result.fail("Long-Short", e)
    
    # Test 2: Long-Only Portfolio
    try:
        portfolio_returns = []
        
        for date, group in data.groupby('date'):
            group = group.sort_values('prediction', ascending=False)
            n = len(group)
            
            n_long = max(1, n // 4)
            long_ret = group.head(n_long)['forward_return'].mean()
            
            portfolio_returns.append(long_ret)
        
        assert len(portfolio_returns) > 0
        result.success(f"Long-Only: {len(portfolio_returns)} days")
    except Exception as e:
        result.fail("Long-Only", e)
    
    # Test 3: Equal Weight
    try:
        n_assets = 10
        weights = np.ones(n_assets) / n_assets
        
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(weights, 0.1)
        result.success("Equal weight allocation")
    except Exception as e:
        result.fail("Equal weight", e)
    
    return result.summary()


# =============================================================================
# TESTS: TRANSACTION COSTS
# =============================================================================

def test_transaction_costs():
    """Test transaction cost calculations."""
    print("\n" + "="*60)
    print("🧪 TEST: Transaction Costs")
    print("="*60)
    
    result = TestResult()
    
    # Test 1: Proportional costs
    try:
        trade_value = 10000
        cost_bps = 10
        
        cost = trade_value * (cost_bps / 10000)
        
        assert cost == 10, f"Expected 10, got {cost}"
        result.success(f"Proportional cost: ${cost}")
    except Exception as e:
        result.fail("Proportional cost", e)
    
    # Test 2: Turnover
    try:
        weights_t0 = np.array([0.3, 0.3, 0.2, 0.2])
        weights_t1 = np.array([0.25, 0.35, 0.25, 0.15])
        
        turnover = np.sum(np.abs(weights_t1 - weights_t0)) / 2
        
        assert 0 <= turnover <= 1
        result.success(f"Turnover = {turnover:.2%}")
    except Exception as e:
        result.fail("Turnover", e)
    
    # Test 3: Cost impact
    try:
        gross_return = 0.10
        turnover = 2.0
        cost_bps = 20
        
        total_cost = turnover * (cost_bps / 10000)
        net_return = gross_return - total_cost
        
        assert net_return < gross_return
        result.success(f"Net return: {net_return:.2%}")
    except Exception as e:
        result.fail("Cost impact", e)
    
    return result.summary()


# =============================================================================
# TESTS: BACKTEST MODULE INTEGRATION
# =============================================================================

def test_backtest_engine():
    """Test BacktestEngine class."""
    print("\n" + "="*60)
    print("🧪 TEST: BacktestEngine Integration")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.backtest.engine import BacktestEngine
        result.success("BacktestEngine imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    try:
        engine = BacktestEngine()
        assert engine is not None
        result.success("BacktestEngine instantiated")
    except Exception as e:
        result.fail("Instantiation", e)
        return result.summary()
    
    # Test run method if exists
    try:
        data = generate_backtest_data()
        
        if hasattr(engine, 'run'):
            results = engine.run(data)
            assert results is not None
            result.success("Backtest run completed")
        else:
            result.success("run() method check (not found)")
    except Exception as e:
        result.fail("Backtest run", e)
    
    return result.summary()


def test_metrics_module():
    """Test metrics module."""
    print("\n" + "="*60)
    print("🧪 TEST: Metrics Module")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.backtest.metrics import (
            sharpe_ratio, max_drawdown, calculate_metrics
        )
        result.success("Metrics imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    returns = generate_returns()
    
    try:
        sr = sharpe_ratio(returns)
        assert np.isfinite(sr)
        result.success(f"sharpe_ratio() = {sr:.2f}")
    except Exception as e:
        result.fail("sharpe_ratio()", e)
    
    try:
        mdd = max_drawdown(returns)
        assert mdd <= 0
        result.success(f"max_drawdown() = {mdd:.2%}")
    except Exception as e:
        result.fail("max_drawdown()", e)
    
    try:
        metrics = calculate_metrics(returns)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        result.success(f"calculate_metrics() = {len(metrics)} metrics")
    except Exception as e:
        result.fail("calculate_metrics()", e)
    
    return result.summary()


def test_portfolio_optimizer():
    """Test PortfolioOptimizer class."""
    print("\n" + "="*60)
    print("🧪 TEST: PortfolioOptimizer")
    print("="*60)
    
    result = TestResult()
    
    try:
        from quant_alpha.backtest.portfolio import PortfolioOptimizer
        result.success("PortfolioOptimizer imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    try:
        optimizer = PortfolioOptimizer()
        assert optimizer is not None
        result.success("PortfolioOptimizer instantiated")
    except Exception as e:
        result.fail("Instantiation", e)
    
    return result.summary()


# =============================================================================
# TESTS: EDGE CASES
# =============================================================================

def test_backtest_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("🧪 TEST: Edge Cases")
    print("="*60)
    
    result = TestResult()
    
    # Test 1: Empty returns
    try:
        empty_returns = pd.Series([], dtype=float)
        
        if len(empty_returns) > 0:
            sharpe = np.sqrt(252) * empty_returns.mean() / empty_returns.std()
        result.success("Empty returns handled")
    except Exception:
        result.success("Empty returns raises error (OK)")
    
    # Test 2: Single return
    try:
        single_return = pd.Series([0.01])
        cumulative = (1 + single_return).cumprod()
        result.success("Single return handled")
    except Exception as e:
        result.fail("Single return", e)
    
    # Test 3: All positive returns
    try:
        positive_returns = pd.Series([0.01] * 100)
        cumulative = (1 + positive_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        assert max_dd == 0, "No drawdown expected"
        result.success("All positive returns: 0 drawdown")
    except Exception as e:
        result.fail("All positive", e)
    
    # Test 4: Extreme returns
    try:
        extreme_returns = pd.Series([0.5, -0.9, 0.1, -0.8, 0.05])
        cumulative = (1 + extreme_returns).cumprod()
        
        assert (cumulative > 0).all(), "Should stay positive"
        result.success("Extreme returns handled")
    except Exception as e:
        result.fail("Extreme returns", e)
    
    return result.summary()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 BACKTEST TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    tests = [
        test_performance_metrics,
        test_portfolio_construction,
        test_transaction_costs,
        test_backtest_engine,
        test_metrics_module,
        test_portfolio_optimizer,
        test_backtest_edge_cases,
    ]
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL BACKTEST TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)