#!/usr/bin/env python3
"""
Run Analysis
============
Comprehensive performance analysis and reporting.

Author: Anurag Patkar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import json
import argparse

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils import Timer, print_header, print_section, save_results, ensure_dir, load_results

try:
    from config import settings, print_welcome
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class PerformanceAnalyzer:
    """Comprehensive performance analysis."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else (settings.results_dir if SETTINGS_AVAILABLE else ROOT / "output")
        self.plots_dir = self.output_dir / "plots"
        ensure_dir(self.plots_dir)
    
    def analyze_model_performance(self, validation_results: pd.DataFrame) -> dict:
        """Analyze model validation performance."""
        print("🤖 Analyzing model performance...")
        
        analysis = {}
        
        # ✅ FIX: Check multiple possible column names
        metric_mappings = {
            'test_ic': ['test_ic', 'ic'],
            'test_rank_ic': ['test_rank_ic', 'rank_ic'],
            'test_hit_rate': ['test_hit_rate', 'hit_rate'],
            'test_rmse': ['test_rmse', 'rmse']
        }
        
        for metric_key, possible_names in metric_mappings.items():
            for col_name in possible_names:
                if col_name in validation_results.columns:
                    values = validation_results[col_name].dropna()
                    if len(values) > 0:
                        analysis[metric_key] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
                    break
        
        # Stability analysis
        ic_col = None
        for col in ['test_ic', 'ic']:
            if col in validation_results.columns:
                ic_col = col
                break
        
        if ic_col:
            ic = validation_results[ic_col].dropna()
            if len(ic) > 0:
                analysis['stability'] = {
                    'positive_folds': int((ic > 0).sum()),
                    'total_folds': int(len(ic)),
                    'positive_rate': float((ic > 0).mean())
                }
        
        return analysis
    
    def analyze_feature_importance(self, importance_df: pd.DataFrame) -> dict:
        """Analyze feature importance patterns."""
        print("🔧 Analyzing feature importance...")
        
        if len(importance_df) == 0:
            return {}
        
        # ✅ FIX: Detect correct column name
        if 'importance_mean' in importance_df.columns:
            imp_col = 'importance_mean'
        elif 'importance' in importance_df.columns:
            imp_col = 'importance'
        else:
            # Find any column with 'importance' in name
            imp_cols = [c for c in importance_df.columns if 'importance' in c.lower()]
            if imp_cols:
                imp_col = imp_cols[0]
            else:
                print("⚠️ No importance column found")
                return {'top_features': importance_df.head(10).to_dict('records')}
        
        print(f"   Using column: '{imp_col}'")
        
        analysis = {'top_features': importance_df.head(10).to_dict('records')}
        
        # Category analysis
        categories = {
            'momentum': ['mom', 'roc', 'ema', 'return'],
            'mean_reversion': ['rsi', 'zscore', 'bb', 'dist_ma'],
            'volatility': ['volatility', 'atr', 'vol_', 'gk_'],
            'volume': ['volume', 'pv_', 'amihud', 'obv']
        }
        
        cat_importance = {}
        for cat, keywords in categories.items():
            pattern = '|'.join(keywords)
            matching = importance_df[importance_df['feature'].str.contains(pattern, case=False, na=False)]
            if len(matching) > 0:
                cat_importance[cat] = {
                    'total': float(matching[imp_col].sum()),
                    'count': int(len(matching)),
                    'top_feature': matching.iloc[0]['feature'] if len(matching) > 0 else None
                }
        
        analysis['categories'] = cat_importance
        return analysis
    
    def analyze_backtest(self, results_df: pd.DataFrame, metrics: dict) -> dict:
        """Analyze backtest performance."""
        print("📈 Analyzing backtest performance...")
        
        analysis = {'metrics': metrics}
        
        if 'returns' not in results_df.columns and 'portfolio_value' in results_df.columns:
            results_df = results_df.copy()
            results_df['returns'] = results_df['portfolio_value'].pct_change()
        
        # ✅ FIX: Check for 'daily_return' column too
        ret_col = None
        for col in ['returns', 'daily_return', 'return']:
            if col in results_df.columns:
                ret_col = col
                break
        
        if ret_col:
            returns = results_df[ret_col].dropna()
            if len(returns) > 0:
                try:
                    from scipy import stats
                    analysis['risk'] = {
                        'var_95': float(returns.quantile(0.05)),
                        'var_99': float(returns.quantile(0.01)),
                        'skewness': float(stats.skew(returns)),
                        'kurtosis': float(stats.kurtosis(returns)),
                        'worst_day': float(returns.min()),
                        'best_day': float(returns.max())
                    }
                except ImportError:
                    analysis['risk'] = {
                        'var_95': float(returns.quantile(0.05)),
                        'var_99': float(returns.quantile(0.01)),
                        'worst_day': float(returns.min()),
                        'best_day': float(returns.max())
                    }
        
        return analysis
    
    def create_plots(self, validation_results: pd.DataFrame, backtest_results: pd.DataFrame):
        """Create performance plots."""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Matplotlib not available, skipping plots")
            return
        
        print("📊 Creating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ML Alpha Model - Performance Analysis', fontsize=14, fontweight='bold')
        
        # ✅ FIX: Detect correct column names
        ic_col = 'ic' if 'ic' in validation_results.columns else 'test_ic'
        date_col = 'test_start' if 'test_start' in validation_results.columns else None
        
        # 1. IC over time
        if ic_col in validation_results.columns:
            if date_col and date_col in validation_results.columns:
                vr = validation_results.copy()
                vr[date_col] = pd.to_datetime(vr[date_col])
                vr = vr.sort_values(date_col)
                axes[0, 0].plot(vr[date_col], vr[ic_col], marker='o', linewidth=2, markersize=4)
                axes[0, 0].axhline(0, color='r', linestyle='--', alpha=0.5, linewidth=1)
                axes[0, 0].axhline(vr[ic_col].mean(), color='g', linestyle='--', alpha=0.5, linewidth=1, label=f'Mean: {vr[ic_col].mean():.4f}')
                axes[0, 0].set_title('IC Over Time')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].plot(validation_results[ic_col].values, marker='o', linewidth=2, markersize=4)
                axes[0, 0].axhline(0, color='r', linestyle='--', alpha=0.5)
                axes[0, 0].set_title('IC by Fold')
                axes[0, 0].set_xlabel('Fold')
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. IC distribution
        if ic_col in validation_results.columns:
            ic_values = validation_results[ic_col].dropna()
            axes[0, 1].hist(ic_values, bins=15, edgecolor='white', alpha=0.7, color='steelblue')
            axes[0, 1].axvline(ic_values.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {ic_values.mean():.4f}')
            axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[0, 1].set_title('IC Distribution')
            axes[0, 1].set_xlabel('Information Coefficient')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Portfolio value
        if len(backtest_results) > 0 and 'portfolio_value' in backtest_results.columns:
            br = backtest_results.copy()
            br['date'] = pd.to_datetime(br['date'])
            axes[1, 0].plot(br['date'], br['portfolio_value'], linewidth=2, color='green')
            axes[1, 0].set_title('Portfolio Value Over Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add drawdown shading
            br['peak'] = br['portfolio_value'].cummax()
            br['dd'] = (br['portfolio_value'] - br['peak']) / br['peak']
            axes[1, 0].fill_between(br['date'], br['portfolio_value'], br['peak'], alpha=0.3, color='red')
        
        # 4. Returns distribution
        ret_col = 'daily_return' if 'daily_return' in backtest_results.columns else 'returns'
        if len(backtest_results) > 0 and ret_col in backtest_results.columns:
            returns = backtest_results[ret_col].dropna() * 100
            axes[1, 1].hist(returns, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
            axes[1, 1].axvline(returns.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
            axes[1, 1].axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Return (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        elif len(backtest_results) > 0 and 'portfolio_value' in backtest_results.columns:
            returns = backtest_results['portfolio_value'].pct_change().dropna() * 100
            axes[1, 1].hist(returns, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
            axes[1, 1].set_title('Daily Returns Distribution')
            axes[1, 1].set_xlabel('Return (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / "performance_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved: {plot_path}")
    
    def generate_report(self, analysis: dict):
        """Generate analysis report."""
        print("📄 Generating report...")
        
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ML ALPHA MODEL - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance
            if 'model_performance' in analysis:
                f.write("MODEL PERFORMANCE\n" + "-" * 40 + "\n")
                mp = analysis['model_performance']
                for metric, stats in mp.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"  {metric}:\n")
                        f.write(f"    Mean: {stats['mean']:.4f}\n")
                        f.write(f"    Std:  {stats['std']:.4f}\n")
                        f.write(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write("\n")
                
                if 'stability' in mp:
                    stab = mp['stability']
                    f.write(f"  Stability:\n")
                    f.write(f"    Positive Folds: {stab['positive_folds']}/{stab['total_folds']}\n")
                    f.write(f"    Positive Rate: {stab['positive_rate']:.1%}\n\n")
            
            # Feature Importance
            if 'feature_importance' in analysis:
                f.write("TOP FEATURES\n" + "-" * 40 + "\n")
                fi = analysis['feature_importance']
                if 'top_features' in fi:
                    for i, feat in enumerate(fi['top_features'][:10], 1):
                        feat_name = feat.get('feature', 'Unknown')
                        # Try different importance column names
                        imp_val = feat.get('importance_mean', feat.get('importance', 0))
                        f.write(f"  {i:2d}. {feat_name:<30} {imp_val:.4f}\n")
                f.write("\n")
                
                if 'categories' in fi:
                    f.write("FEATURE CATEGORIES\n" + "-" * 40 + "\n")
                    for cat, stats in fi['categories'].items():
                        f.write(f"  {cat.upper()}:\n")
                        f.write(f"    Total Importance: {stats['total']:.4f}\n")
                        f.write(f"    Feature Count: {stats['count']}\n")
                        if stats.get('top_feature'):
                            f.write(f"    Top Feature: {stats['top_feature']}\n")
                    f.write("\n")
            
            # Backtest
            if 'backtest' in analysis and 'metrics' in analysis['backtest']:
                f.write("BACKTEST PERFORMANCE\n" + "-" * 40 + "\n")
                m = analysis['backtest']['metrics']
                
                if 'total_return' in m:
                    f.write(f"  Total Return:     {m['total_return']:>+10.2%}\n")
                if 'annual_return' in m:
                    f.write(f"  Annual Return:    {m['annual_return']:>+10.2%}\n")
                if 'sharpe_ratio' in m:
                    f.write(f"  Sharpe Ratio:     {m['sharpe_ratio']:>10.3f}\n")
                if 'max_drawdown' in m:
                    f.write(f"  Max Drawdown:     {m['max_drawdown']:>10.2%}\n")
                if 'win_rate' in m:
                    f.write(f"  Win Rate:         {m['win_rate']:>10.2%}\n")
                if 'total_impact_cost' in m:
                    f.write(f"  Total Impact Cost: ${m['total_impact_cost']:>10,.0f}\n")
                if 'annual_cost_drag' in m:
                    f.write(f"  Annual Cost Drag: {m['annual_cost_drag']:>10.2%}\n")
                f.write("\n")
                
                # Risk metrics
                if 'risk' in analysis['backtest']:
                    f.write("RISK METRICS\n" + "-" * 40 + "\n")
                    risk = analysis['backtest']['risk']
                    if 'var_95' in risk:
                        f.write(f"  VaR (95%):    {risk['var_95']:>10.2%}\n")
                    if 'var_99' in risk:
                        f.write(f"  VaR (99%):    {risk['var_99']:>10.2%}\n")
                    if 'skewness' in risk:
                        f.write(f"  Skewness:     {risk['skewness']:>10.3f}\n")
                    if 'kurtosis' in risk:
                        f.write(f"  Kurtosis:     {risk['kurtosis']:>10.3f}\n")
                    if 'worst_day' in risk:
                        f.write(f"  Worst Day:    {risk['worst_day']:>10.2%}\n")
                    if 'best_day' in risk:
                        f.write(f"  Best Day:     {risk['best_day']:>10.2%}\n")
                    f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"   📄 Saved: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Performance Analysis')
    parser.add_argument('--no-plots', action='store_true', help='Skip plots')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if SETTINGS_AVAILABLE:
        print_welcome()
    
    print_header("PERFORMANCE ANALYSIS")
    
    try:
        # Paths
        if SETTINGS_AVAILABLE:
            results_dir = settings.results_dir
        else:
            results_dir = ROOT / "output" / "results"
        
        # ✅ FIX: Try multiple file names
        val_paths = [
            results_dir / "validation_results_ml.csv",
            results_dir / "validation_results.csv"
        ]
        
        val_path = None
        for p in val_paths:
            if p.exists():
                val_path = p
                break
        
        if val_path is None:
            print("❌ Missing validation results! Run run_research.py first.")
            print(f"   Looked for: {val_paths}")
            return 1
        
        imp_path = results_dir / "feature_importance.csv"
        bt_path = results_dir / "backtest_results.csv"
        metrics_path = results_dir / "backtest_metrics.json"
        
        # Load data
        validation_results = pd.read_csv(val_path)
        print(f"✅ Loaded validation: {len(validation_results)} folds")
        
        importance_df = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
        print(f"✅ Loaded importance: {len(importance_df)} features")
        
        backtest_results = pd.read_csv(bt_path) if bt_path.exists() else pd.DataFrame()
        backtest_metrics = load_results(metrics_path) if metrics_path.exists() else {}
        print(f"✅ Loaded backtest: {len(backtest_results)} periods")
        
        # Analyze
        output_dir = Path(args.output) if args.output else results_dir
        analyzer = PerformanceAnalyzer(output_dir)
        
        analysis = {}
        analysis['model_performance'] = analyzer.analyze_model_performance(validation_results)
        analysis['feature_importance'] = analyzer.analyze_feature_importance(importance_df)
        
        if len(backtest_results) > 0:
            analysis['backtest'] = analyzer.analyze_backtest(backtest_results, backtest_metrics)
        
        # Plots
        if not args.no_plots:
            analyzer.create_plots(validation_results, backtest_results)
        
        # Report
        analyzer.generate_report(analysis)
        
        # Save analysis
        save_results(analysis, output_dir / "analysis_results.json")
        
        # Print summary
        print_header("ANALYSIS SUMMARY")
        
        if 'model_performance' in analysis and 'test_ic' in analysis['model_performance']:
            ic_stats = analysis['model_performance']['test_ic']
            print(f"  🎯 Average IC: {ic_stats['mean']:.4f} ± {ic_stats['std']:.4f}")
        
        if 'backtest' in analysis and 'metrics' in analysis['backtest']:
            m = analysis['backtest']['metrics']
            if 'sharpe_ratio' in m:
                print(f"  📊 Sharpe Ratio: {m['sharpe_ratio']:.3f}")
            if 'total_return' in m:
                print(f"  📈 Total Return: {m['total_return']:.2%}")
            if 'max_drawdown' in m:
                print(f"  📉 Max Drawdown: {m['max_drawdown']:.2%}")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Analysis completed in {elapsed:.1f}s")
        
        print("\n" + "="*60)
        print("📁 Generated Files:")
        print("="*60)
        print(f"   📄 {output_dir / 'analysis_report.txt'}")
        print(f"   📊 {output_dir / 'plots' / 'performance_analysis.png'}")
        print(f"   📋 {output_dir / 'analysis_results.json'}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())