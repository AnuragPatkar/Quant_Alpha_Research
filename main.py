"""
ML-Based Multi-Factor Alpha Model
==================================
Main entry point for the complete pipeline.

Uses REAL S&P 500 data from Stooq.

Run: python main.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Imports
from config.settings import settings
from quant_alpha.data import DataLoader
from quant_alpha.features import compute_all_features
from quant_alpha.research import WalkForwardValidator
from quant_alpha.backtest import Backtester


def create_plots(val_results: pd.DataFrame, backtest_result, importance: pd.DataFrame):
    """Create and save result plots."""
    print("\n" + "="*60)
    print("📊 CREATING PLOTS")
    print("="*60)
    
    # Create output directory
    settings.create_dirs()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. IC by Fold
    ax = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in val_results['ic']]
    ax.bar(val_results['fold'], val_results['ic'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=val_results['ic'].mean(), color='blue', linestyle='--',
               label=f"Mean: {val_results['ic'].mean():.4f}")
    ax.set_xlabel('Fold')
    ax.set_ylabel('IC')
    ax.set_title('Information Coefficient by Fold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative Returns
    ax = axes[0, 1]
    if backtest_result is not None:
        ax.plot(backtest_result.cumulative.index, 
                backtest_result.cumulative.values, 
                linewidth=2, color='blue', label='Strategy')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Portfolio Cumulative Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Feature Importance
    ax = axes[1, 0]
    top_15 = importance.head(15).sort_values('importance_pct')
    ax.barh(top_15['feature'], top_15['importance_pct'], color='steelblue')
    ax.set_xlabel('Importance (%)')
    ax.set_title('Top 15 Feature Importance')
    
    # 4. Returns Distribution
    ax = axes[1, 1]
    if backtest_result is not None:
        returns = backtest_result.returns.dropna()
        ax.hist(returns, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=returns.mean(), color='red', linestyle='--',
                   label=f"Mean: {returns.mean():.2%}")
        ax.set_xlabel('Monthly Return')
        ax.set_ylabel('Frequency')
        ax.set_title('Monthly Returns Distribution')
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = settings.plots_dir / 'results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   💾 Saved: {plot_path}")
    
    plt.show()


def print_final_report(val_results: pd.DataFrame, backtest_result, importance: pd.DataFrame):
    """Print final report."""
    print("\n" + "="*60)
    print("📄 FINAL REPORT")
    print("="*60)
    
    # Validation summary
    mean_ic = val_results['ic'].mean()
    std_ic = val_results['ic'].std()
    ir = mean_ic / (std_ic + 1e-10)
    
    print(f"""
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATION RESULTS                         │
├──────────────────────────────────────────────────────────────┤
│   Folds:              {len(val_results):>10}                              │
│   Mean IC:            {mean_ic:>10.4f}                              │
│   Std IC:             {std_ic:>10.4f}                              │
│   Information Ratio:  {ir:>10.4f}                              │
│   Mean Hit Rate:      {val_results['hit_rate'].mean():>10.1%}                              │
└──────────────────────────────────────────────────────────────┘
    """)
    
    if backtest_result is not None:
        m = backtest_result.metrics
        print(f"""
┌──────────────────────────────────────────────────────────────┐
│                    BACKTEST RESULTS                          │
├──────────────────────────────────────────────────────────────┤
│   Total Return:       {m['total_return']:>10.1%}                              │
│   Annual Return:      {m['annual_return']:>10.1%}                              │
│   Sharpe Ratio:       {m['sharpe_ratio']:>10.2f}                              │
│   Max Drawdown:       {m['max_drawdown']:>10.1%}                              │
│   Win Rate:           {m['win_rate']:>10.1%}                              │
└──────────────────────────────────────────────────────────────┘
        """)
    
    print("\n📊 TOP 10 FEATURES:")
    print("─" * 40)
    for i, row in importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance_pct']:>6.2f}%")


def main():
    """Main pipeline - runs everything from data loading to backtesting."""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ML-BASED MULTI-FACTOR ALPHA MODEL                        ║
║     Cross-Sectional Stock Return Prediction                  ║
║                                                              ║
║     Data: Real S&P 500 (Stooq)                               ║
║     Model: LightGBM                                          ║
║     Validation: Walk-Forward                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = datetime.now()
    
    # Show current configuration
    settings.print_config()
    
    # =========================================================
    # STEP 1: Load Data
    # =========================================================
    loader = DataLoader()
    data = loader.load()
    
    # =========================================================
    # STEP 2: Feature Engineering
    # =========================================================
    # This creates all 27 alpha factors
    features_df, feature_names = compute_all_features(data)
    
    # =========================================================
    # STEP 3: Walk-Forward Validation
    # =========================================================
    # Train model using time-series cross-validation
    validator = WalkForwardValidator()
    val_results, predictions, model = validator.validate(features_df, feature_names)
    
    # =========================================================
    # STEP 4: Feature Importance
    # =========================================================
    # Let's see which features the model thinks are most important
    print("\n" + "="*60)
    print("📊 FEATURE IMPORTANCE")
    print("="*60)
    
    importance = model.get_feature_importance()
    print("\n   Top 10 Features:")
    for i, row in importance.head(10).iterrows():
        print(f"      {i+1:2d}. {row['feature']:<25} {row['importance_pct']:>6.2f}%")
    
    # Save the trained model for later use
    settings.create_dirs()
    model.save(str(settings.models_dir / 'alpha_model.pkl'))
    
    # =========================================================
    # STEP 5: Backtesting
    # =========================================================
    # Now test the strategy with realistic transaction costs
    backtester = Backtester()
    backtest_result = backtester.run(predictions)
    
    # =========================================================
    # STEP 6: Visualization
    # =========================================================
    create_plots(val_results, backtest_result, importance)
    
    # =========================================================
    # STEP 7: Final Report
    # =========================================================
    print_final_report(val_results, backtest_result, importance)
    
    # =========================================================
    # COMPLETION
    # =========================================================
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED!")
    print("="*60)
    print(f"   ⏱️  Total Time: {elapsed}")
    print(f"\n   📁 Output Files:")
    print(f"      • {settings.models_dir / 'alpha_model.pkl'}")
    print(f"      • {settings.plots_dir / 'results.png'}")
    
    return {
        'validation': val_results,
        'predictions': predictions,
        'backtest': backtest_result,
        'importance': importance,
        'model': model
    }


if __name__ == "__main__":
    results = main()