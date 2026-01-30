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
        
        for metric in ['test_ic', 'test_rank_ic', 'test_hit_rate', 'test_rmse']:
            if metric in validation_results.columns:
                values = validation_results[metric].dropna()
                if len(values) > 0:
                    analysis[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
        
        if 'test_ic' in validation_results.columns:
            ic = validation_results['test_ic'].dropna()
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
        
        analysis = {'top_features': importance_df.head(10).to_dict('records')}
        
        # Category analysis
        categories = {
            'momentum': ['mom', 'roc', 'ema'],
            'mean_reversion': ['rsi', 'zscore', 'bb'],
            'volatility': ['volatility', 'atr', 'vol'],
            'volume': ['volume', 'pv_', 'amihud']
        }
        
        cat_importance = {}
        for cat, keywords in categories.items():
            matching = importance_df[importance_df['feature'].str.contains('|'.join(keywords), case=False, na=False)]
            if len(matching) > 0:
                cat_importance[cat] = {
                    'total': float(matching['importance_mean'].sum()),
                    'count': int(len(matching))
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
        
        if 'returns' in results_df.columns:
            returns = results_df['returns'].dropna()
            if len(returns) > 0:
                from scipy import stats
                analysis['risk'] = {
                    'var_95': float(returns.quantile(0.05)),
                    'skewness': float(stats.skew(returns)),
                    'kurtosis': float(stats.kurtosis(returns))
                }
        
        return analysis
    
    def create_plots(self, validation_results: pd.DataFrame, backtest_results: pd.DataFrame):
        """Create performance plots."""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Matplotlib not available")
            return
        
        print("📊 Creating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ML Alpha Model - Performance Analysis', fontsize=14)
        
        # IC over time
        if 'test_ic' in validation_results.columns and 'test_start' in validation_results.columns:
            vr = validation_results.copy()
            vr['test_start'] = pd.to_datetime(vr['test_start'])
            vr = vr.sort_values('test_start')
            axes[0, 0].plot(vr['test_start'], vr['test_ic'], marker='o')
            axes[0, 0].axhline(0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('IC Over Time')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # IC distribution
        if 'test_ic' in validation_results.columns:
            axes[0, 1].hist(validation_results['test_ic'], bins=15, edgecolor='white')
            axes[0, 1].axvline(validation_results['test_ic'].mean(), color='r', linestyle='--')
            axes[0, 1].set_title('IC Distribution')
        
        # Portfolio value
        if len(backtest_results) > 0 and 'portfolio_value' in backtest_results.columns:
            br = backtest_results.copy()
            br['date'] = pd.to_datetime(br['date'])
            axes[1, 0].plot(br['date'], br['portfolio_value'])
            axes[1, 0].set_title('Portfolio Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Returns distribution
        if len(backtest_results) > 0 and 'portfolio_value' in backtest_results.columns:
            returns = backtest_results['portfolio_value'].pct_change().dropna()
            if len(returns) > 0:
                axes[1, 1].hist(returns * 100, bins=15, edgecolor='white')
                axes[1, 1].set_title('Monthly Returns Distribution')
                axes[1, 1].set_xlabel('Return (%)')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_analysis.png", dpi=150)
        plt.close()
        print(f"   📊 Saved: performance_analysis.png")
    
    def generate_report(self, analysis: dict):
        """Generate analysis report."""
        print("📄 Generating report...")
        
        report_path = self.output_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ML ALPHA MODEL - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance
            if 'model_performance' in analysis:
                f.write("MODEL PERFORMANCE\n" + "-" * 40 + "\n")
                mp = analysis['model_performance']
                for metric, stats in mp.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"{metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}\n")
                f.write("\n")
            
            # Feature Importance
            if 'feature_importance' in analysis:
                f.write("TOP FEATURES\n" + "-" * 40 + "\n")
                fi = analysis['feature_importance']
                if 'top_features' in fi:
                    for feat in fi['top_features'][:10]:
                        f.write(f"{feat['feature']}: {feat['importance_mean']:.4f}\n")
                f.write("\n")
            
            # Backtest
            if 'backtest' in analysis and 'metrics' in analysis['backtest']:
                f.write("BACKTEST PERFORMANCE\n" + "-" * 40 + "\n")
                m = analysis['backtest']['metrics']
                if 'total_return' in m:
                    f.write(f"Total Return: {m['total_return']:.2%}\n")
                if 'sharpe_ratio' in m:
                    f.write(f"Sharpe Ratio: {m['sharpe_ratio']:.3f}\n")
                if 'max_drawdown' in m:
                    f.write(f"Max Drawdown: {m['max_drawdown']:.2%}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"   📄 Saved: {report_path.name}")


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
        
        # Load data
        val_path = results_dir / "validation_results.csv"
        imp_path = results_dir / "feature_importance.csv"
        bt_path = results_dir / "backtest_results.csv"
        metrics_path = results_dir / "backtest_metrics.json"
        
        if not val_path.exists():
            print("❌ Missing validation_results.csv! Run run_research.py first.")
            return 1
        
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
        
        print(f"\n✅ Analysis completed in {time.time() - start_time:.1f}s")
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())