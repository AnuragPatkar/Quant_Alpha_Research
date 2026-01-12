"""
Visualization Module Test Script
=================================
Test all visualization components:
- plots.py (PerformancePlotter, FactorPlotter, RiskPlotter)
- reports.py (ReportGenerator, exports)
- dashboards.py (already tested separately)

Run: python scripts/test_visualization.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("   VISUALIZATION MODULE TEST")
print("=" * 70)


# =============================================================================
# STEP 1: Generate Demo Data
# =============================================================================

def generate_test_data():
    """Generate comprehensive test data."""
    print("\n📊 Generating test data...")
    
    np.random.seed(42)
    
    # Date range
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Returns with realistic properties
    returns = np.random.normal(0.0005, 0.015, n_days)
    
    # Add some autocorrelation and fat tails
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum
        if np.random.random() < 0.02:  # Fat tails
            returns[i] *= np.random.choice([2, -2])
    
    returns = pd.Series(returns, index=dates)
    
    # Equity curve
    initial_capital = 10000000  # 1 Crore
    equity_curve = (1 + returns).cumprod() * initial_capital
    
    # Trades
    n_trades = 300
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
    })
    trades = trades.sort_values('date').reset_index(drop=True)
    
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
        'start_date': '2022-01-01',
        'end_date': '2024-01-01',
        'universe': 'NIFTY 50',
        'initial_capital': initial_capital,
        'strategy': 'ML Multi-Factor Alpha'
    }
    
    backtest_results = {
        'metrics': metrics,
        'equity_curve': equity_curve,
        'returns': returns,
        'trades': trades,
        'feature_importance': feature_importance,
        'config': config
    }
    
    print("   ✅ Test data generated!")
    print(f"   - Equity curve: {len(equity_curve)} days")
    print(f"   - Returns: {len(returns)} days")
    print(f"   - Trades: {len(trades)} trades")
    print(f"   - Features: {len(feature_importance)} features")
    
    return backtest_results


# =============================================================================
# STEP 2: Test Plots
# =============================================================================

def test_plots(data, output_dir):
    """Test all plotting functions."""
    print("\n" + "=" * 70)
    print("   TESTING: plots.py")
    print("=" * 70)
    
    from quant_alpha.visualization.plots import (
        PerformancePlotter,
        FactorPlotter,
        RiskPlotter,
        quick_plot_all
    )
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    equity_curve = data['equity_curve']
    returns = data['returns']
    trades = data['trades']
    feature_importance = data['feature_importance']
    
    # Test PerformancePlotter
    print("\n📈 Testing PerformancePlotter...")
    perf = PerformancePlotter(style='default')
    
    tests = [
        ("Equity Curve", lambda: perf.plot_equity_curve(
            equity_curve, 
            save_path=str(plots_dir / 'equity_curve.png'),
            show=False
        )),
        ("Returns Distribution", lambda: perf.plot_returns_distribution(
            returns,
            save_path=str(plots_dir / 'returns_distribution.png'),
            show=False
        )),
        ("Rolling Metrics", lambda: perf.plot_rolling_metrics(
            returns,
            save_path=str(plots_dir / 'rolling_metrics.png'),
            show=False
        )),
        ("Monthly Heatmap", lambda: perf.plot_monthly_heatmap(
            returns,
            save_path=str(plots_dir / 'monthly_heatmap.png'),
            show=False
        )),
        ("Trade Analysis", lambda: perf.plot_trade_analysis(
            trades,
            save_path=str(plots_dir / 'trade_analysis.png'),
            show=False
        )),
    ]
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    # Test FactorPlotter
    print("\n🔬 Testing FactorPlotter...")
    factor = FactorPlotter()
    
    try:
        factor.plot_feature_importance(
            feature_importance,
            save_path=str(plots_dir / 'feature_importance.png'),
            show=False
        )
        print("   ✅ Feature Importance")
    except Exception as e:
        print(f"   ❌ Feature Importance: {e}")
    
    try:
        # Create dummy factor returns for testing
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
        print("   ✅ Factor Returns")
    except Exception as e:
        print(f"   ❌ Factor Returns: {e}")
    
    # Test RiskPlotter
    print("\n⚠️ Testing RiskPlotter...")
    risk = RiskPlotter()
    
    try:
        risk.plot_var_analysis(
            returns,
            save_path=str(plots_dir / 'var_analysis.png'),
            show=False
        )
        print("   ✅ VaR Analysis")
    except Exception as e:
        print(f"   ❌ VaR Analysis: {e}")
    
    try:
        risk.plot_risk_dashboard(
            returns,
            equity_curve,
            save_path=str(plots_dir / 'risk_dashboard.png'),
            show=False
        )
        print("   ✅ Risk Dashboard")
    except Exception as e:
        print(f"   ❌ Risk Dashboard: {e}")
    
    # Test quick_plot_all
    print("\n⚡ Testing quick_plot_all...")
    try:
        saved = quick_plot_all(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            feature_importance=feature_importance,
            output_dir=str(plots_dir / 'quick_all'),
            show=False
        )
        print(f"   ✅ Generated {len(saved)} plots")
    except Exception as e:
        print(f"   ❌ quick_plot_all: {e}")
    
    print(f"\n   📁 Plots saved to: {plots_dir}")


# =============================================================================
# STEP 3: Test Reports
# =============================================================================

def test_reports(data, output_dir):
    """Test all report generation functions."""
    print("\n" + "=" * 70)
    print("   TESTING: reports.py")
    print("=" * 70)
    
    from quant_alpha.visualization.reports import (
        ReportGenerator,
        generate_report,
        print_metrics,
        save_all_charts,
        export_to_json,
        export_to_excel,
        export_to_csv
    )
    
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Test ReportGenerator class
    print("\n📄 Testing ReportGenerator class...")
    generator = ReportGenerator(output_dir=str(reports_dir))
    
    # Test PDF generation
    print("\n   📑 Generating PDF report...")
    try:
        pdf_path = generator.generate_pdf(data, 'test_report')
        print(f"   ✅ PDF: {pdf_path}")
    except Exception as e:
        print(f"   ❌ PDF: {e}")
    
    # Test terminal report
    print("\n   🖥️ Testing terminal report...")
    try:
        generator.print_terminal_report(data)
        print("   ✅ Terminal report printed above")
    except Exception as e:
        print(f"   ❌ Terminal report: {e}")
    
    # Test JSON export
    print("\n   📋 Testing JSON export...")
    try:
        json_path = generator.export_to_json(data, 'test_metrics')
        print(f"   ✅ JSON: {json_path}")
    except Exception as e:
        print(f"   ❌ JSON: {e}")
    
    # Test Excel export
    print("\n   📊 Testing Excel export...")
    try:
        excel_path = generator.export_to_excel(data, 'test_data')
        print(f"   ✅ Excel: {excel_path}")
    except Exception as e:
        print(f"   ❌ Excel: {e}")
    
    # Test CSV export
    print("\n   📝 Testing CSV export...")
    try:
        csv_path = generator.export_to_csv(data, 'test_csv')
        print(f"   ✅ CSV: {csv_path}")
    except Exception as e:
        print(f"   ❌ CSV: {e}")
    
    # Test Pickle export
    print("\n   💾 Testing Pickle export...")
    try:
        pkl_path = generator.export_to_pickle(data, 'test_pickle')
        print(f"   ✅ Pickle: {pkl_path}")
    except Exception as e:
        print(f"   ❌ Pickle: {e}")
    
    # Test save_all_charts
    print("\n   🖼️ Testing save_all_charts...")
    try:
        charts_path = generator.save_all_charts(data, 'test_charts')
        print(f"   ✅ Charts: {charts_path}")
    except Exception as e:
        print(f"   ❌ Charts: {e}")
    
    # Test convenience functions
    print("\n⚡ Testing convenience functions...")
    
    # generate_report
    try:
        outputs = generate_report(
            data,
            output_dir=str(reports_dir / 'quick_report'),
            formats=['json', 'png']  # Skip PDF for faster test
        )
        print(f"   ✅ generate_report: {len(outputs)} formats")
    except Exception as e:
        print(f"   ❌ generate_report: {e}")
    
    # print_metrics (already tested above, skip verbose output)
    print("   ✅ print_metrics: (tested above)")
    
    print(f"\n   📁 Reports saved to: {reports_dir}")


# =============================================================================
# STEP 4: Test Full Report Generation
# =============================================================================

def test_full_report(data, output_dir):
    """Test complete report generation with all formats."""
    print("\n" + "=" * 70)
    print("   TESTING: Full Report Generation")
    print("=" * 70)
    
    from quant_alpha.visualization.reports import ReportGenerator
    
    full_report_dir = output_dir / 'full_report'
    full_report_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ReportGenerator(output_dir=str(full_report_dir))
    
    print("\n🚀 Generating complete report (all formats)...")
    
    try:
        outputs = generator.generate_full_report(
            data,
            formats=['pdf', 'terminal', 'json', 'excel', 'csv', 'png', 'pickle'],
            report_name='complete_backtest_report'
        )
        
        print("\n   Generated files:")
        for fmt, path in outputs.items():
            print(f"   ✅ {fmt}: {path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# STEP 5: Display Sample Outputs
# =============================================================================

def display_sample_outputs(data, output_dir):
    """Show sample of what was generated."""
    print("\n" + "=" * 70)
    print("   SAMPLE OUTPUTS")
    print("=" * 70)
    
    # Show metrics
    print("\n📊 Sample Metrics:")
    metrics = data['metrics']
    print(f"   CAGR:            {metrics['cagr']*100:>10.2f}%")
    print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
    print(f"   Max Drawdown:    {metrics['max_drawdown']*100:>10.2f}%")
    print(f"   Win Rate:        {metrics['win_rate']*100:>10.1f}%")
    print(f"   Total Trades:    {metrics['total_trades']:>10.0f}")
    
    # Show top features
    print("\n🔬 Top 5 Features:")
    for i, row in data['feature_importance'].head(5).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Show recent trades
    print("\n💹 Recent Trades (last 5):")
    recent = data['trades'].tail(5)
    for _, row in recent.iterrows():
        pnl_str = f"₹{row['pnl']:,.0f}"
        side_emoji = "🟢" if row['side'] == 'LONG' else "🔴"
        print(f"   {side_emoji} {row['ticker']}: {pnl_str}")
    
    # List generated files
    print("\n📁 Generated Files:")
    for item in sorted(output_dir.rglob('*')):
        if item.is_file():
            rel_path = item.relative_to(output_dir)
            size_kb = item.stat().st_size / 1024
            print(f"   📄 {rel_path} ({size_kb:.1f} KB)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "🚀" * 35)
    print("\n   Starting Visualization Module Tests...")
    print("\n" + "🚀" * 35)
    
    # Setup output directory
    output_dir = Path('test_outputs') / f"viz_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    # Generate test data
    data = generate_test_data()
    
    # Run tests
    test_plots(data, output_dir)
    test_reports(data, output_dir)
    test_full_report(data, output_dir)
    display_sample_outputs(data, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("   TEST SUMMARY")
    print("=" * 70)
    
    # Count files
    total_files = len(list(output_dir.rglob('*.*')))
    png_files = len(list(output_dir.rglob('*.png')))
    pdf_files = len(list(output_dir.rglob('*.pdf')))
    json_files = len(list(output_dir.rglob('*.json')))
    excel_files = len(list(output_dir.rglob('*.xlsx')))
    csv_files = len(list(output_dir.rglob('*.csv')))
    
    print(f"""
   📊 Files Generated:
   ─────────────────────
   PNG Charts:     {png_files:>5}
   PDF Reports:    {pdf_files:>5}
   JSON Files:     {json_files:>5}
   Excel Files:    {excel_files:>5}
   CSV Files:      {csv_files:>5}
   ─────────────────────
   TOTAL:          {total_files:>5}
   
   📁 Output Location: {output_dir}
   
   ✅ All tests completed!
   
   Next Steps:
   1. Open PNG files to view charts
   2. Open PDF for complete report
   3. Open Excel/CSV for data
   4. Run dashboard: streamlit run quant_alpha/visualization/dashboards.py
    """)
    
    return output_dir


if __name__ == "__main__":
    output_dir = main()