# Complete `scripts/run_analysis.py` (Full Code)


"""
Run Analysis
============
Comprehensive performance analysis and reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import json

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import settings, print_welcome

warnings.filterwarnings('ignore')

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Matplotlib not available. Plots will be skipped.")


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis.
    
    Features:
        - Model performance analysis
        - Feature importance analysis
        - Backtest performance analysis
        - Comparative analysis
        - Report generation
    """
    
    def __init__(self):
        """Initialize analyzer."""
        self.results_dir = settings.results_dir
        self.plots_dir = settings.plots_dir
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_model_performance(self, validation_results: pd.DataFrame) -> dict:
        """Analyze model validation performance."""
        print("🤖 Analyzing model performance...")
        
        analysis = {}
        
        # Basic statistics
        metrics = ['test_ic', 'test_rank_ic', 'test_hit_rate', 'test_rmse']
        
        for metric in metrics:
            if metric in validation_results.columns:
                values = validation_results[metric].dropna()
                analysis[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75)
                }
        
        # Stability analysis
        if 'test_ic' in validation_results.columns:
            ic_values = validation_results['test_ic'].dropna()
            analysis['stability'] = {
                'positive_folds': (ic_values > 0).sum(),
                'total_folds': len(ic_values),
                'positive_rate': (ic_values > 0).mean(),
                'consistency_score': 1 - (ic_values.std() / abs(ic_values.mean())) if ic_values.mean() != 0 else 0
            }
        
        # Time series analysis
        if 'test_start' in validation_results.columns:
            validation_results['test_start'] = pd.to_datetime(validation_results['test_start'])
            validation_results = validation_results.sort_values('test_start')
            
            # Rolling performance
            window = 3
            if len(validation_results) >= window:
                analysis['rolling_performance'] = {
                    'rolling_ic_mean': validation_results['test_ic'].rolling(window).mean().tolist(),
                    'rolling_ic_std': validation_results['test_ic'].rolling(window).std().tolist(),
                    'dates': validation_results['test_start'].dt.strftime('%Y-%m-%d').tolist()
                }
        
        return analysis
    
    def analyze_feature_importance(self, importance_df: pd.DataFrame) -> dict:
        """Analyze feature importance patterns."""
        print("🔧 Analyzing feature importance...")
        
        analysis = {}
        
        if len(importance_df) == 0:
            return analysis
        
        # Top features
        analysis['top_features'] = importance_df.head(10).to_dict('records')
        
        # Feature categories
        categories = {
            'momentum': [f for f in importance_df['feature'] if 'mom' in f or 'roc' in f or 'ema' in f],
            'mean_reversion': [f for f in importance_df['feature'] if 'rsi' in f or 'dist' in f or 'zscore' in f or 'bb' in f],
            'volatility': [f for f in importance_df['feature'] if 'volatility' in f or 'atr' in f or 'vol_ratio' in f or 'skew' in f],
            'volume': [f for f in importance_df['feature'] if 'volume' in f or 'pv_' in f or 'amihud' in f or 'relative' in f]
        }
        
        category_importance = {}
        for category, features in categories.items():
            if features:
                cat_importance = importance_df[importance_df['feature'].isin(features)]['importance_mean'].sum()
                category_importance[category] = {
                    'total_importance': cat_importance,
                    'feature_count': len(features),
                    'avg_importance': cat_importance / len(features) if len(features) > 0 else 0
                }
        
        analysis['category_analysis'] = category_importance
        
        # Feature stability (if multiple folds available)
        if 'importance_std' in importance_df.columns:
            importance_df['stability_ratio'] = importance_df['importance_mean'] / (importance_df['importance_std'] + 1e-10)
            analysis['most_stable_features'] = importance_df.nlargest(5, 'stability_ratio')[['feature', 'stability_ratio']].to_dict('records')
        
        return analysis
    
    def analyze_backtest_performance(self, backtest_results: pd.DataFrame, backtest_metrics: dict) -> dict:
        """Analyze backtest performance."""
        print("📈 Analyzing backtest performance...")
        
        analysis = {}
        
        if len(backtest_results) == 0:
            return analysis
        
        # Basic performance metrics
        analysis['performance_metrics'] = backtest_metrics
        
        # Calculate returns if not present
        if 'returns' not in backtest_results.columns and 'portfolio_value' in backtest_results.columns:
            backtest_results['returns'] = backtest_results['portfolio_value'].pct_change()
        
        # Risk analysis
        if 'returns' in backtest_results.columns:
            returns = backtest_results['returns'].dropna()
            
            if len(returns) > 0:
                analysis['risk_analysis'] = {
                    'var_95': returns.quantile(0.05),
                    'var_99': returns.quantile(0.01),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'worst_month': returns.min(),
                    'best_month': returns.max(),
                    'negative_months': (returns < 0).sum(),
                    'positive_months': (returns > 0).sum()
                }
        
        # Drawdown analysis
        if 'portfolio_value' in backtest_results.columns:
            portfolio_values = backtest_results['portfolio_value']
            cumulative = portfolio_values / portfolio_values.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            analysis['drawdown_analysis'] = {
                'max_drawdown': drawdown.min(),
                'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
                'drawdown_periods': (drawdown < -0.05).sum(),  # Periods with >5% drawdown
                'recovery_time': self._calculate_recovery_time(drawdown)
            }
        
        return analysis
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        in_drawdown = False
        drawdown_start = None
        recovery_times = []
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:  # Recovery
                in_drawdown = False
                if drawdown_start is not None:
                    recovery_times.append(i - drawdown_start)
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def generate_comparison_analysis(self, validation_results: pd.DataFrame, backtest_metrics: dict) -> dict:
        """Generate comparison between model and backtest performance."""
        print("🔍 Generating comparison analysis...")
        
        comparison = {}
        
        # Model vs Backtest
        if 'test_ic' in validation_results.columns and backtest_metrics:
            avg_ic = validation_results['test_ic'].mean()
            sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0)
            
            comparison['model_vs_backtest'] = {
                'avg_ic': avg_ic,
                'sharpe_ratio': sharpe_ratio,
                'ic_to_sharpe_ratio': sharpe_ratio / avg_ic if avg_ic != 0 else 0,
                'interpretation': self._interpret_ic_to_sharpe(avg_ic, sharpe_ratio)
            }
        
        # Performance consistency
        if 'test_ic' in validation_results.columns:
            ic_std = validation_results['test_ic'].std()
            comparison['consistency'] = {
                'ic_volatility': ic_std,
                'consistency_rating': 'High' if ic_std < 0.05 else 'Medium' if ic_std < 0.10 else 'Low'
            }
        
        return comparison
    
    def _interpret_ic_to_sharpe(self, ic: float, sharpe: float) -> str:
        """Interpret IC to Sharpe ratio relationship."""
        if ic <= 0:
            return "Poor: Negative or zero IC"
        elif sharpe <= 0:
            return "Poor: Negative Sharpe despite positive IC - high transaction costs or poor execution"
        elif sharpe / ic > 2:
            return "Excellent: High Sharpe relative to IC - efficient strategy"
        elif sharpe / ic > 1:
            return "Good: Reasonable Sharpe relative to IC"
        else:
            return "Fair: Low Sharpe relative to IC - consider reducing costs or improving execution"
    
    def create_performance_plots(self, validation_results: pd.DataFrame, backtest_results: pd.DataFrame):
        """Create performance visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("⚠️ Skipping plots - matplotlib not available")
            return
        
        print("📊 Creating performance plots...")
        
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ML Alpha Model - Performance Analysis', fontsize=16)
        
        # Plot 1: IC over time
        if 'test_ic' in validation_results.columns and 'test_start' in validation_results.columns:
            validation_results['test_start'] = pd.to_datetime(validation_results['test_start'])
            validation_results = validation_results.sort_values('test_start')
            
            axes[0, 0].plot(validation_results['test_start'], validation_results['test_ic'], marker='o', linewidth=2)
            axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Information Coefficient Over Time')
            axes[0, 0].set_ylabel('IC')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: IC distribution
            axes[0, 1].hist(validation_results['test_ic'], bins=10, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 1].axvline(x=validation_results['test_ic'].mean(), color='r', linestyle='--', label='Mean', linewidth=2)
            axes[0, 1].set_title('IC Distribution')
            axes[0, 1].set_xlabel('IC')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Portfolio value over time
        if len(backtest_results) > 0 and 'portfolio_value' in backtest_results.columns:
            backtest_results['date'] = pd.to_datetime(backtest_results['date'])
            axes[1, 0].plot(backtest_results['date'], backtest_results['portfolio_value'], linewidth=2, color='green')
            axes[1, 0].set_title('Portfolio Value Over Time')
            axes[1, 0].set_ylabel('Portfolio Value ($)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Format y-axis to show currency
            axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Plot 4: Monthly returns
            if 'returns' not in backtest_results.columns:
                backtest_results['returns'] = backtest_results['portfolio_value'].pct_change()
            
            returns = backtest_results['returns'].dropna()
            if len(returns) > 0:
                axes[1, 1].hist(returns, bins=10, alpha=0.7, edgecolor='black', color='lightcoral')
                axes[1, 1].axvline(x=returns.mean(), color='r', linestyle='--', label='Mean', linewidth=2)
                axes[1, 1].set_title('Monthly Returns Distribution')
                axes[1, 1].set_xlabel('Monthly Return')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Format x-axis as percentage
                axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        plot_path = self.plots_dir / "performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Performance plots saved: {plot_path.name}")

    def generate_analysis_report(self, analysis_results: dict):
        """Generate comprehensive analysis report."""
        print("📄 Generating analysis report...")
        
        report_path = self.results_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ML ALPHA MODEL - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance
            if 'model_performance' in analysis_results:
                f.write("MODEL PERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                model_perf = analysis_results['model_performance']
                
                if 'test_ic' in model_perf:
                    ic_stats = model_perf['test_ic']
                    f.write(f"Information Coefficient:\n")
                    f.write(f"  Mean: {ic_stats['mean']:.4f}\n")
                    f.write(f"  Std:  {ic_stats['std']:.4f}\n")
                    f.write(f"  Min:  {ic_stats['min']:.4f}\n")
                    f.write(f"  Max:  {ic_stats['max']:.4f}\n")
                    f.write(f"  Median: {ic_stats['median']:.4f}\n\n")
                
                if 'test_rank_ic' in model_perf:
                    rank_ic_stats = model_perf['test_rank_ic']
                    f.write(f"Rank Information Coefficient:\n")
                    f.write(f"  Mean: {rank_ic_stats['mean']:.4f}\n")
                    f.write(f"  Std:  {rank_ic_stats['std']:.4f}\n\n")
                
                if 'test_hit_rate' in model_perf:
                    hit_rate_stats = model_perf['test_hit_rate']
                    f.write(f"Hit Rate:\n")
                    f.write(f"  Mean: {hit_rate_stats['mean']:.2%}\n")
                    f.write(f"  Std:  {hit_rate_stats['std']:.2%}\n\n")
                
                if 'stability' in model_perf:
                    stability = model_perf['stability']
                    f.write(f"Model Stability:\n")
                    f.write(f"  Positive Folds: {stability['positive_folds']}/{stability['total_folds']}\n")
                    f.write(f"  Positive Rate: {stability['positive_rate']:.2%}\n")
                    f.write(f"  Consistency Score: {stability['consistency_score']:.3f}\n\n")
            
            # Feature Analysis
            if 'feature_importance' in analysis_results:
                f.write("FEATURE IMPORTANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                feat_analysis = analysis_results['feature_importance']
                
                if 'top_features' in feat_analysis:
                    f.write("Top 10 Features:\n")
                    for i, feat in enumerate(feat_analysis['top_features'][:10]):
                        f.write(f"  {i+1:2d}. {feat['feature']:20s}: {feat['importance_mean']:.4f}\n")
                    f.write("\n")
                
                if 'category_analysis' in feat_analysis:
                    f.write("Feature Category Analysis:\n")
                    for category, stats in feat_analysis['category_analysis'].items():
                        f.write(f"  {category.title():15s}: {stats['total_importance']:.4f} ")
                        f.write(f"({stats['feature_count']} features, avg: {stats['avg_importance']:.4f})\n")
                    f.write("\n")
                
                if 'most_stable_features' in feat_analysis:
                    f.write("Most Stable Features:\n")
                    for feat in feat_analysis['most_stable_features']:
                        f.write(f"  {feat['feature']:20s}: {feat['stability_ratio']:.3f}\n")
                    f.write("\n")
            
            # Backtest Performance
            if 'backtest_performance' in analysis_results:
                f.write("BACKTEST PERFORMANCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                backtest_perf = analysis_results['backtest_performance']
                
                if 'performance_metrics' in backtest_perf:
                    metrics = backtest_perf['performance_metrics']
                    f.write("Portfolio Performance:\n")
                    if 'total_return' in metrics:
                        f.write(f"  Total Return: {metrics['total_return']:.2%}\n")
                    if 'annualized_return' in metrics:
                        f.write(f"  Annualized Return: {metrics['annualized_return']:.2%}\n")
                    if 'volatility' in metrics:
                        f.write(f"  Volatility: {metrics['volatility']:.2%}\n")
                    if 'sharpe_ratio' in metrics:
                        f.write(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
                    if 'max_drawdown' in metrics:
                        f.write(f"  Max Drawdown: {metrics['max_drawdown']:.2%}\n")
                    if 'win_rate' in metrics:
                        f.write(f"  Win Rate: {metrics['win_rate']:.2%}\n")
                    f.write("\n")
                
                if 'risk_analysis' in backtest_perf:
                    risk = backtest_perf['risk_analysis']
                    f.write("Risk Analysis:\n")
                    if 'var_95' in risk:
                        f.write(f"  VaR (95%): {risk['var_95']:.2%}\n")
                    if 'var_99' in risk:
                        f.write(f"  VaR (99%): {risk['var_99']:.2%}\n")
                    if 'skewness' in risk:
                        f.write(f"  Skewness: {risk['skewness']:.3f}\n")
                    if 'kurtosis' in risk:
                        f.write(f"  Kurtosis: {risk['kurtosis']:.3f}\n")
                    if 'worst_month' in risk:
                        f.write(f"  Worst Month: {risk['worst_month']:.2%}\n")
                    if 'best_month' in risk:
                        f.write(f"  Best Month: {risk['best_month']:.2%}\n")
                    f.write("\n")
                
                if 'drawdown_analysis' in backtest_perf:
                    dd = backtest_perf['drawdown_analysis']
                    f.write("Drawdown Analysis:\n")
                    if 'max_drawdown' in dd:
                        f.write(f"  Max Drawdown: {dd['max_drawdown']:.2%}\n")
                    if 'avg_drawdown' in dd:
                        f.write(f"  Avg Drawdown: {dd['avg_drawdown']:.2%}\n")
                    if 'drawdown_periods' in dd:
                        f.write(f"  Periods >5% DD: {dd['drawdown_periods']}\n")
                    if 'recovery_time' in dd:
                        f.write(f"  Avg Recovery Time: {dd['recovery_time']:.1f} periods\n")
                    f.write("\n")
            
            # Comparison Analysis
            if 'comparison' in analysis_results:
                f.write("COMPARISON ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                comparison = analysis_results['comparison']
                
                if 'model_vs_backtest' in comparison:
                    mvb = comparison['model_vs_backtest']
                    f.write("Model vs Backtest:\n")
                    if 'avg_ic' in mvb:
                        f.write(f"  Average IC: {mvb['avg_ic']:.4f}\n")
                    if 'sharpe_ratio' in mvb:
                        f.write(f"  Sharpe Ratio: {mvb['sharpe_ratio']:.3f}\n")
                    if 'ic_to_sharpe_ratio' in mvb:
                        f.write(f"  IC to Sharpe Ratio: {mvb['ic_to_sharpe_ratio']:.2f}\n")
                    if 'interpretation' in mvb:
                        f.write(f"  Interpretation: {mvb['interpretation']}\n")
                    f.write("\n")
                
                if 'consistency' in comparison:
                    consistency = comparison['consistency']
                    f.write("Performance Consistency:\n")
                    if 'ic_volatility' in consistency:
                        f.write(f"  IC Volatility: {consistency['ic_volatility']:.4f}\n")
                    if 'consistency_rating' in consistency:
                        f.write(f"  Consistency Rating: {consistency['consistency_rating']}\n")
                    f.write("\n")
            
            # Performance Interpretation
            f.write("PERFORMANCE INTERPRETATION\n")
            f.write("-" * 40 + "\n")
            
            # Overall assessment
            if 'model_performance' in analysis_results and 'test_ic' in analysis_results['model_performance']:
                avg_ic = analysis_results['model_performance']['test_ic']['mean']
                
                if avg_ic > 0.05:
                    f.write("Overall Assessment: EXCELLENT\n")
                    f.write("  Strong predictive power with high IC\n")
                elif avg_ic > 0.02:
                    f.write("Overall Assessment: GOOD\n")
                    f.write("  Moderate predictive power, suitable for trading\n")
                elif avg_ic > 0:
                    f.write("Overall Assessment: FAIR\n")
                    f.write("  Weak but positive predictive power\n")
                else:
                    f.write("Overall Assessment: POOR\n")
                    f.write("  No predictive power, model needs improvement\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            recommendations = []
            
            # IC-based recommendations
            if 'model_performance' in analysis_results and 'test_ic' in analysis_results['model_performance']:
                avg_ic = analysis_results['model_performance']['test_ic']['mean']
                ic_std = analysis_results['model_performance']['test_ic']['std']
                
                if avg_ic < 0.02:
                    recommendations.append("Improve feature engineering (cross-sectional ranking, interaction features)")
                
                if ic_std > 0.10:
                    recommendations.append("Increase model regularization to reduce overfitting")
                    recommendations.append("Consider ensemble methods for stability")
                
                if 'stability' in analysis_results['model_performance']:
                    pos_rate = analysis_results['model_performance']['stability']['positive_rate']
                    if pos_rate < 0.6:
                        recommendations.append("Model is inconsistent - review feature selection")
            
            # Backtest-based recommendations
            if 'backtest_performance' in analysis_results:
                if 'performance_metrics' in analysis_results['backtest_performance']:
                    metrics = analysis_results['backtest_performance']['performance_metrics']
                    
                    if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] < 0.5:
                        recommendations.append("Low Sharpe ratio - consider reducing transaction costs")
                    
                    if 'max_drawdown' in metrics and abs(metrics['max_drawdown']) > 0.15:
                        recommendations.append("High drawdown - implement better risk management")
            
            # Feature-based recommendations
            if 'feature_importance' in analysis_results:
                if 'category_analysis' in analysis_results['feature_importance']:
                    cat_analysis = analysis_results['feature_importance']['category_analysis']
                    
                    # Check if any category dominates
                    total_importance = sum(cat['total_importance'] for cat in cat_analysis.values())
                    for category, stats in cat_analysis.items():
                        if stats['total_importance'] / total_importance > 0.6:
                            recommendations.append(f"Feature diversity: {category} dominates - add more diverse features")
            
            if not recommendations:
                recommendations.append("Model performance is satisfactory - continue monitoring")
            
            for i, rec in enumerate(recommendations, 1):
                f.write(f"  {i}. {rec}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"   📄 Analysis report saved: {report_path.name}")
        
            


def main():
    """Run comprehensive analysis pipeline."""
    
    start_time = time.time()
    
    print_welcome()
    print("\n🔍 COMPREHENSIVE ANALYSIS PIPELINE")
    print("="*60)
    
    try:
        # Check for required files
        results_path = settings.results_dir / "validation_results.csv"
        importance_path = settings.results_dir / "feature_importance.csv"
        backtest_results_path = settings.results_dir / "backtest_results.csv"
        backtest_metrics_path = settings.results_dir / "backtest_metrics.json"
        
        if not results_path.exists():
            print("❌ Missing validation results! Run research pipeline first:")
            print("   python scripts/run_research.py")
            return False
        
        # Load data
        print("📊 Loading analysis data...")
        validation_results = pd.read_csv(results_path)
        
        importance_df = pd.DataFrame()
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
        
        backtest_results = pd.DataFrame()
        backtest_metrics = {}
        if backtest_results_path.exists():
            backtest_results = pd.read_csv(backtest_results_path)
        if backtest_metrics_path.exists():
            with open(backtest_metrics_path, 'r') as f:
                backtest_metrics = json.load(f)
        
        print(f"✅ Loaded validation results: {len(validation_results)} folds")
        print(f"✅ Loaded feature importance: {len(importance_df)} features")
        print(f"✅ Loaded backtest results: {len(backtest_results)} periods")
        
        # Initialize analyzer
        analyzer = PerformanceAnalyzer()
        
        # Run analysis
        analysis_results = {}
        
        # Model performance analysis
        analysis_results['model_performance'] = analyzer.analyze_model_performance(validation_results)
        
        # Feature importance analysis
        if len(importance_df) > 0:
            analysis_results['feature_importance'] = analyzer.analyze_feature_importance(importance_df)
        
        # Backtest performance analysis
        if len(backtest_results) > 0:
            analysis_results['backtest_performance'] = analyzer.analyze_backtest_performance(backtest_results, backtest_metrics)
        
        # Comparison analysis
        analysis_results['comparison'] = analyzer.generate_comparison_analysis(validation_results, backtest_metrics)
        
        # Create plots
        analyzer.create_performance_plots(validation_results, backtest_results)
        
        # Generate comprehensive report
        analyzer.generate_analysis_report(analysis_results)
        
        # Save analysis results as JSON
        analysis_json_path = settings.results_dir / "analysis_results.json"
        with open(analysis_json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(analysis_results, f, indent=2, default=convert_numpy)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Files created:")
        print(f"   • analysis_report.txt")
        print(f"   • analysis_results.json")
        print(f"   • performance_analysis.png (if matplotlib available)")
        
        total_time = time.time() - start_time
        print(f"⏱️  Execution time: {total_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Performance Analysis')
    parser.add_argument('--model-only', action='store_true', help='Analyze model performance only')
    parser.add_argument('--features-only', action='store_true', help='Analyze feature importance only')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    success = main()
    if success:
        print("\n🎯 Analysis completed successfully!")
    else:
        print("\n💥 Analysis failed!")
        sys.exit(1)