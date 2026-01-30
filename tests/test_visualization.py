"""
Test Visualization Module
=========================
Tests for quant_alpha/visualization/
- plots.py (PerformancePlotter, FactorPlotter, RiskPlotter)
- reports.py (ReportGenerator, exports)
- dashboards.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

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


def generate_viz_data(n_days=252, n_trades=100):
    """Generate comprehensive test data for visualization."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    # Returns with realistic properties
    returns = np.random.normal(0.0005, 0.015, n_days)
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum
        if np.random.random() < 0.02:  # Fat tails
            returns[i] *= np.random.choice([2, -2])
    
    returns = pd.Series(returns, index=dates)
    
    # Equity curve
    initial_capital = 10_000_000
    equity_curve = (1 + returns).cumprod() * initial_capital
    
    # Trades
    tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
               'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC']
    
    trades = pd.DataFrame({
        'date': pd.to_datetime(np.random.choice(dates, n_trades)),
        'ticker': np.random.choice(tickers, n_trades),
        'side': np.random.choice(['LONG', 'SHORT'], n_trades, p=[0.6, 0.4]),
        'quantity': np.random.randint(10, 100, n_trades),
        'entry_price': np.random.uniform(500, 3000, n_trades),
        'pnl': np.random.normal(5000, 20000, n_trades),
        'return_pct': np.random.normal(0.01, 0.04, n_trades)
    }).sort_values('date').reset_index(drop=True)
    
    # Feature importance
    features = [
        'momentum_20d', 'momentum_60d', 'volatility_20d', 'volume_zscore',
        'rsi_14', 'macd_signal', 'bb_position', 'atr_ratio',
        'price_to_sma50', 'sector_momentum', 'market_regime',
        'earnings_momentum', 'analyst_revisions', 'short_interest',
        'institutional_flow', 'volatility_regime', 'beta',
        'idiosyncratic_vol', 'liquidity_score', 'price_to_sma200'
    ]
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': np.abs(np.random.exponential(0.05, len(features)))
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Calculate metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = n_days / 252
    cagr = (1 + total_return) ** (1/n_years) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility
    
    peak = equity_curve.expanding().max()
    max_drawdown = ((equity_curve - peak) / peak).min()
    
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]
    
    metrics = {
        'cagr': cagr,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sharpe * 1.3,
        'calmar_ratio': cagr / abs(max_drawdown) if max_drawdown != 0 else 0,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else 0,
        'total_trades': len(trades),
        'avg_trade_return': trades['return_pct'].mean(),
        'var_95': np.percentile(returns, 5),
        'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'best_day': returns.max(),
        'worst_day': returns.min(),
        'avg_daily_return': returns.mean()
    }
    
    config = {
        'start_date': str(dates[0].date()),
        'end_date': str(dates[-1].date()),
        'universe': 'NIFTY 50',
        'initial_capital': initial_capital,
        'strategy': 'ML Multi-Factor Alpha'
    }
    
    return {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'returns': returns,
        'trades': trades,
        'feature_importance': feature_importance,
        'config': config
    }


def get_output_dir():
    """Create and return output directory."""
    output_dir = ROOT / 'test_outputs' / f"viz_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# TEST: PERFORMANCE PLOTTER
# =============================================================================

def test_performance_plotter(data, output_dir):
    """Test PerformancePlotter class."""
    print("\n" + "="*60)
    print("🧪 TEST: PerformancePlotter")
    print("="*60)
    
    result = TestResult()
    plots_dir = output_dir / 'performance_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.plots import PerformancePlotter
        result.success("PerformancePlotter imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    perf = PerformancePlotter(style='default')
    equity_curve = data['equity_curve']
    returns = data['returns']
    trades = data['trades']
    
    # Test: Equity Curve
    try:
        perf.plot_equity_curve(
            equity_curve,
            save_path=str(plots_dir / 'equity_curve.png'),
            show=False
        )
        assert (plots_dir / 'equity_curve.png').exists(), "File not created"
        result.success("plot_equity_curve()")
    except Exception as e:
        result.fail("plot_equity_curve()", e)
    
    # Test: Returns Distribution
    try:
        perf.plot_returns_distribution(
            returns,
            save_path=str(plots_dir / 'returns_distribution.png'),
            show=False
        )
        assert (plots_dir / 'returns_distribution.png').exists()
        result.success("plot_returns_distribution()")
    except Exception as e:
        result.fail("plot_returns_distribution()", e)
    
    # Test: Rolling Metrics
    try:
        perf.plot_rolling_metrics(
            returns,
            save_path=str(plots_dir / 'rolling_metrics.png'),
            show=False
        )
        assert (plots_dir / 'rolling_metrics.png').exists()
        result.success("plot_rolling_metrics()")
    except Exception as e:
        result.fail("plot_rolling_metrics()", e)
    
    # Test: Monthly Heatmap
    try:
        perf.plot_monthly_heatmap(
            returns,
            save_path=str(plots_dir / 'monthly_heatmap.png'),
            show=False
        )
        assert (plots_dir / 'monthly_heatmap.png').exists()
        result.success("plot_monthly_heatmap()")
    except Exception as e:
        result.fail("plot_monthly_heatmap()", e)
    
    # Test: Trade Analysis
    try:
        perf.plot_trade_analysis(
            trades,
            save_path=str(plots_dir / 'trade_analysis.png'),
            show=False
        )
        assert (plots_dir / 'trade_analysis.png').exists()
        result.success("plot_trade_analysis()")
    except Exception as e:
        result.fail("plot_trade_analysis()", e)
    
    print(f"\n   📁 Plots saved to: {plots_dir}")
    return result.summary()


# =============================================================================
# TEST: FACTOR PLOTTER
# =============================================================================

def test_factor_plotter(data, output_dir):
    """Test FactorPlotter class."""
    print("\n" + "="*60)
    print("🧪 TEST: FactorPlotter")
    print("="*60)
    
    result = TestResult()
    plots_dir = output_dir / 'factor_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.plots import FactorPlotter
        result.success("FactorPlotter imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    factor = FactorPlotter()
    feature_importance = data['feature_importance']
    returns = data['returns']
    
    # Test: Feature Importance
    try:
        factor.plot_feature_importance(
            feature_importance,
            save_path=str(plots_dir / 'feature_importance.png'),
            show=False
        )
        assert (plots_dir / 'feature_importance.png').exists()
        result.success("plot_feature_importance()")
    except Exception as e:
        result.fail("plot_feature_importance()", e)
    
    # Test: Factor Returns
    try:
        factor_returns = pd.DataFrame(
            np.random.randn(len(returns), 5) * 0.01,
            index=returns.index,
            columns=['momentum', 'value', 'quality', 'volatility', 'size']
        )
        factor.plot_factor_returns(
            factor_returns,
            save_path=str(plots_dir / 'factor_returns.png'),
            show=False
        )
        assert (plots_dir / 'factor_returns.png').exists()
        result.success("plot_factor_returns()")
    except Exception as e:
        result.fail("plot_factor_returns()", e)
    
    print(f"\n   📁 Plots saved to: {plots_dir}")
    return result.summary()


# =============================================================================
# TEST: RISK PLOTTER
# =============================================================================

def test_risk_plotter(data, output_dir):
    """Test RiskPlotter class."""
    print("\n" + "="*60)
    print("🧪 TEST: RiskPlotter")
    print("="*60)
    
    result = TestResult()
    plots_dir = output_dir / 'risk_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.plots import RiskPlotter
        result.success("RiskPlotter imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    risk = RiskPlotter()
    returns = data['returns']
    equity_curve = data['equity_curve']
    
    # Test: VaR Analysis
    try:
        risk.plot_var_analysis(
            returns,
            save_path=str(plots_dir / 'var_analysis.png'),
            show=False
        )
        assert (plots_dir / 'var_analysis.png').exists()
        result.success("plot_var_analysis()")
    except Exception as e:
        result.fail("plot_var_analysis()", e)
    
    # Test: Risk Dashboard
    try:
        risk.plot_risk_dashboard(
            returns,
            equity_curve,
            save_path=str(plots_dir / 'risk_dashboard.png'),
            show=False
        )
        assert (plots_dir / 'risk_dashboard.png').exists()
        result.success("plot_risk_dashboard()")
    except Exception as e:
        result.fail("plot_risk_dashboard()", e)
    
    print(f"\n   📁 Plots saved to: {plots_dir}")
    return result.summary()


# =============================================================================
# TEST: QUICK PLOT ALL
# =============================================================================

def test_quick_plot_all(data, output_dir):
    """Test quick_plot_all convenience function."""
    print("\n" + "="*60)
    print("🧪 TEST: quick_plot_all()")
    print("="*60)
    
    result = TestResult()
    plots_dir = output_dir / 'quick_all'
    
    # Import
    try:
        from quant_alpha.visualization.plots import quick_plot_all
        result.success("quick_plot_all imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    # Test
    try:
        saved_files = quick_plot_all(
            equity_curve=data['equity_curve'],
            returns=data['returns'],
            trades=data['trades'],
            feature_importance=data['feature_importance'],
            output_dir=str(plots_dir),
            show=False
        )
        
        assert len(saved_files) > 0, "No files saved"
        result.success(f"Generated {len(saved_files)} plots")
        
        # Verify files exist
        for filepath in saved_files:
            if Path(filepath).exists():
                continue
            else:
                result.fail(f"File missing", filepath)
                break
        else:
            result.success("All files exist")
        
    except Exception as e:
        result.fail("quick_plot_all()", e)
    
    print(f"\n   📁 Plots saved to: {plots_dir}")
    return result.summary()


# =============================================================================
# TEST: REPORT GENERATOR
# =============================================================================

def test_report_generator(data, output_dir):
    """Test ReportGenerator class."""
    print("\n" + "="*60)
    print("🧪 TEST: ReportGenerator")
    print("="*60)
    
    result = TestResult()
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.reports import ReportGenerator
        result.success("ReportGenerator imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    generator = ReportGenerator(output_dir=str(reports_dir))
    
    # Test: JSON Export
    try:
        json_path = generator.export_to_json(data, 'test_metrics')
        assert Path(json_path).exists(), "JSON file not created"
        result.success(f"export_to_json()")
    except Exception as e:
        result.fail("export_to_json()", e)
    
    # Test: CSV Export
    try:
        csv_path = generator.export_to_csv(data, 'test_data')
        assert Path(csv_path).exists(), "CSV file not created"
        result.success(f"export_to_csv()")
    except Exception as e:
        result.fail("export_to_csv()", e)
    
    # Test: Excel Export
    try:
        excel_path = generator.export_to_excel(data, 'test_excel')
        assert Path(excel_path).exists(), "Excel file not created"
        result.success(f"export_to_excel()")
    except Exception as e:
        result.fail("export_to_excel()", e)
    
    # Test: Pickle Export
    try:
        pkl_path = generator.export_to_pickle(data, 'test_pickle')
        assert Path(pkl_path).exists(), "Pickle file not created"
        result.success(f"export_to_pickle()")
    except Exception as e:
        result.fail("export_to_pickle()", e)
    
    # Test: Save All Charts
    try:
        charts_path = generator.save_all_charts(data, 'test_charts')
        assert Path(charts_path).exists(), "Charts directory not created"
        result.success(f"save_all_charts()")
    except Exception as e:
        result.fail("save_all_charts()", e)
    
    # Test: Terminal Report
    try:
        print("\n   --- Terminal Report Preview ---")
        generator.print_terminal_report(data)
        print("   --- End Preview ---\n")
        result.success("print_terminal_report()")
    except Exception as e:
        result.fail("print_terminal_report()", e)
    
    # Test: PDF (optional - may need extra dependencies)
    try:
        pdf_path = generator.generate_pdf(data, 'test_report')
        assert Path(pdf_path).exists(), "PDF file not created"
        result.success(f"generate_pdf()")
    except Exception as e:
        result.fail("generate_pdf()", f"(Optional) {e}")
    
    print(f"\n   📁 Reports saved to: {reports_dir}")
    return result.summary()


# =============================================================================
# TEST: CONVENIENCE FUNCTIONS
# =============================================================================

def test_convenience_functions(data, output_dir):
    """Test convenience functions from reports module."""
    print("\n" + "="*60)
    print("🧪 TEST: Convenience Functions")
    print("="*60)
    
    result = TestResult()
    conv_dir = output_dir / 'convenience'
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.reports import (
            generate_report,
            print_metrics,
            export_to_json,
            export_to_excel,
            export_to_csv
        )
        result.success("Convenience functions imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    # Test: generate_report
    try:
        outputs = generate_report(
            data,
            output_dir=str(conv_dir),
            formats=['json', 'csv']  # Skip PDF for faster test
        )
        assert len(outputs) > 0, "No outputs generated"
        result.success(f"generate_report() - {len(outputs)} formats")
    except Exception as e:
        result.fail("generate_report()", e)
    
    # Test: print_metrics
    try:
        print_metrics(data['metrics'])
        result.success("print_metrics()")
    except Exception as e:
        result.fail("print_metrics()", e)
    
    # Test: Individual exports
    try:
        export_to_json(data, str(conv_dir / 'direct_export.json'))
        result.success("export_to_json() direct")
    except Exception as e:
        result.fail("export_to_json() direct", e)
    
    print(f"\n   📁 Files saved to: {conv_dir}")
    return result.summary()


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

def test_edge_cases(output_dir):
    """Test edge cases in visualization."""
    print("\n" + "="*60)
    print("🧪 TEST: Edge Cases")
    print("="*60)
    
    result = TestResult()
    edge_dir = output_dir / 'edge_cases'
    edge_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.plots import PerformancePlotter
        from quant_alpha.visualization.reports import ReportGenerator
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    perf = PerformancePlotter()
    
    # Test 1: Empty returns
    try:
        empty_returns = pd.Series([], dtype=float)
        perf.plot_returns_distribution(empty_returns, show=False)
        result.fail("Empty returns", "Should have raised error")
    except Exception:
        result.success("Empty returns raises error (expected)")
    
    # Test 2: Single data point
    try:
        single_return = pd.Series([0.01], index=pd.date_range('2020-01-01', periods=1))
        perf.plot_returns_distribution(single_return, show=False)
        result.success("Single data point handled")
    except Exception:
        result.success("Single data point raises error (OK)")
    
    # Test 3: NaN values in returns
    try:
        nan_returns = pd.Series(
            [0.01, np.nan, 0.02, np.nan, 0.03],
            index=pd.date_range('2020-01-01', periods=5)
        )
        perf.plot_returns_distribution(nan_returns, show=False)
        result.success("NaN values handled")
    except Exception as e:
        result.fail("NaN values", e)
    
    # Test 4: Empty trades DataFrame
    try:
        empty_trades = pd.DataFrame(columns=['date', 'ticker', 'side', 'pnl', 'return_pct'])
        perf.plot_trade_analysis(empty_trades, show=False)
        result.success("Empty trades handled")
    except Exception:
        result.success("Empty trades raises error (OK)")
    
    # Test 5: Very large dataset
    try:
        np.random.seed(42)
        large_returns = pd.Series(
            np.random.randn(10000) * 0.01,
            index=pd.date_range('2000-01-01', periods=10000)
        )
        perf.plot_returns_distribution(
            large_returns,
            save_path=str(edge_dir / 'large_dataset.png'),
            show=False
        )
        result.success("Large dataset (10k points) handled")
    except Exception as e:
        result.fail("Large dataset", e)
    
    # Test 6: Negative equity curve
    try:
        # This shouldn't happen in practice, but test anyway
        neg_equity = pd.Series(
            [100, 90, 80, -10, -20],  # Goes negative
            index=pd.date_range('2020-01-01', periods=5)
        )
        perf.plot_equity_curve(neg_equity, show=False)
        result.success("Negative equity handled")
    except Exception:
        result.success("Negative equity raises error (OK)")
    
    # Test 7: All same returns (zero volatility)
    try:
        constant_returns = pd.Series(
            [0.01] * 100,
            index=pd.date_range('2020-01-01', periods=100)
        )
        perf.plot_returns_distribution(constant_returns, show=False)
        result.success("Constant returns handled")
    except Exception as e:
        result.fail("Constant returns", e)
    
    return result.summary()


# =============================================================================
# TEST: FULL REPORT GENERATION
# =============================================================================

def test_full_report(data, output_dir):
    """Test complete report generation with all formats."""
    print("\n" + "="*60)
    print("🧪 TEST: Full Report Generation")
    print("="*60)
    
    result = TestResult()
    full_dir = output_dir / 'full_report'
    full_dir.mkdir(parents=True, exist_ok=True)
    
    # Import
    try:
        from quant_alpha.visualization.reports import ReportGenerator
        result.success("ReportGenerator imported")
    except ImportError as e:
        result.fail("Import", e)
        return result.summary()
    
    generator = ReportGenerator(output_dir=str(full_dir))
    
    # Test: generate_full_report
    try:
        outputs = generator.generate_full_report(
            data,
            formats=['terminal', 'json', 'csv', 'png', 'pickle'],  # Skip PDF for speed
            report_name='complete_backtest_report'
        )
        
        assert len(outputs) > 0, "No outputs generated"
        result.success(f"generate_full_report() - {len(outputs)} formats")
        
        print("\n   Generated files:")
        for fmt, path in outputs.items():
            print(f"      {fmt}: {path}")
        
    except Exception as e:
        result.fail("generate_full_report()", e)
    
    print(f"\n   📁 Full report saved to: {full_dir}")
    return result.summary()


# =============================================================================
# SUMMARY HELPER
# =============================================================================

def print_file_summary(output_dir):
    """Print summary of generated files."""
    print("\n" + "="*60)
    print("📊 FILE SUMMARY")
    print("="*60)
    
    # Count files by type
    png_files = list(output_dir.rglob('*.png'))
    pdf_files = list(output_dir.rglob('*.pdf'))
    json_files = list(output_dir.rglob('*.json'))
    excel_files = list(output_dir.rglob('*.xlsx'))
    csv_files = list(output_dir.rglob('*.csv'))
    pkl_files = list(output_dir.rglob('*.pkl'))
    
    total = len(png_files) + len(pdf_files) + len(json_files) + len(excel_files) + len(csv_files) + len(pkl_files)
    
    print(f"""
   Files Generated:
   ─────────────────────
   PNG Charts:     {len(png_files):>5}
   PDF Reports:    {len(pdf_files):>5}
   JSON Files:     {len(json_files):>5}
   Excel Files:    {len(excel_files):>5}
   CSV Files:      {len(csv_files):>5}
   Pickle Files:   {len(pkl_files):>5}
   ─────────────────────
   TOTAL:          {total:>5}
   
   📁 Output Location: {output_dir}
    """)
    
    return total


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 VISUALIZATION MODULE TEST SUITE")
    print("="*60)
    
    # Setup output directory
    output_dir = get_output_dir()
    print(f"\n📂 Output directory: {output_dir}")
    
    # Generate test data
    print("\n📊 Generating test data...")
    data = generate_viz_data(n_days=252, n_trades=100)
    print(f"   ✅ Data ready:")
    print(f"      - Equity curve: {len(data['equity_curve'])} days")
    print(f"      - Returns: {len(data['returns'])} days")
    print(f"      - Trades: {len(data['trades'])} trades")
    print(f"      - Features: {len(data['feature_importance'])} features")
    
    # Run tests
    all_passed = True
    
    if not test_performance_plotter(data, output_dir):
        all_passed = False
    
    if not test_factor_plotter(data, output_dir):
        all_passed = False
    
    if not test_risk_plotter(data, output_dir):
        all_passed = False
    
    if not test_quick_plot_all(data, output_dir):
        all_passed = False
    
    if not test_report_generator(data, output_dir):
        all_passed = False
    
    if not test_convenience_functions(data, output_dir):
        all_passed = False
    
    if not test_edge_cases(output_dir):
        all_passed = False
    
    if not test_full_report(data, output_dir):
        all_passed = False
    
    # Print file summary
    total_files = print_file_summary(output_dir)
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VISUALIZATION TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60)
    
    print(f"""
   Next Steps:
   1. Open PNG files to view charts
   2. Open Excel/CSV for data export verification
   3. Check JSON files for metrics export
   4. Run dashboard: streamlit run quant_alpha/visualization/dashboards.py
    """)
    
    sys.exit(0 if all_passed else 1)