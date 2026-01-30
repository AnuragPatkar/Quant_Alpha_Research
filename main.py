"""
ML-Based Multi-Factor Alpha Model
==================================
Main entry point for the complete pipeline.

Uses REAL S&P 500 data from Stooq.

Usage:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test with less data
    python main.py --skip-backtest    # Skip backtesting
    python main.py --help             # Show help
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import argparse
from pathlib import Path
from datetime import datetime
import traceback

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


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ML-Based Multi-Factor Alpha Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                    # Full pipeline
    python main.py --quick            # Quick test mode
    python main.py --skip-backtest    # Skip backtesting step
    python main.py --no-plots         # Skip plot generation
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test mode (fewer stocks, shorter period)'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtesting step'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Save computed features to CSV'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


# =============================================================================
# BANNER & LOGGING
# =============================================================================

def print_banner():
    """Print welcome banner."""
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


def print_step(step_num: int, title: str):
    """Print step header."""
    print(f"\n{'='*60}")
    print(f"📌 STEP {step_num}: {title}")
    print(f"{'='*60}")


def print_success(message: str):
    """Print success message."""
    print(f"   ✅ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"   ❌ {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"   ⚠️ {message}")


def print_info(message: str):
    """Print info message."""
    print(f"   ℹ️ {message}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plots(val_results: pd.DataFrame, backtest_result, importance: pd.DataFrame, save: bool = True):
    """Create and save result plots."""
    print("\n" + "="*60)
    print("📊 CREATING PLOTS")
    print("="*60)
    
    try:
        settings.create_dirs()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. IC by Fold - find correct column name
        ax = axes[0, 0]
        
        # Try different IC column names
        ic_col = None
        for col in ['ic', 'test_ic', 'rank_ic', 'test_rank_ic', 'IC']:
            if col in val_results.columns:
                ic_col = col
                break
        
        if ic_col:
            ic_values = val_results[ic_col]
            colors = ['green' if x > 0 else 'red' for x in ic_values]
            ax.bar(range(len(ic_values)), ic_values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axhline(y=ic_values.mean(), color='blue', linestyle='--',
                       label=f"Mean: {ic_values.mean():.4f}")
            ax.set_xlabel('Fold')
            ax.set_ylabel('IC')
            ax.set_title(f'Information Coefficient by Fold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No IC data', ha='center', va='center')
            ax.set_title('IC by Fold (No Data)')
        
        # 2. Cumulative Returns (keep same)
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
        else:
            ax.text(0.5, 0.5, 'No Backtest Data', ha='center', va='center', 
                   fontsize=14, color='gray')
            ax.set_title('Portfolio Cumulative Returns (Skipped)')
        
        # 3. Feature Importance (keep same)
        ax = axes[1, 0]
        top_15 = importance.head(15).sort_values('importance_pct')
        ax.barh(top_15['feature'], top_15['importance_pct'], color='steelblue')
        ax.set_xlabel('Importance (%)')
        ax.set_title('Top 15 Feature Importance')
        
        # 4. Returns Distribution (keep same)
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
        else:
            ax.text(0.5, 0.5, 'No Backtest Data', ha='center', va='center',
                   fontsize=14, color='gray')
            ax.set_title('Monthly Returns Distribution (Skipped)')
        
        plt.tight_layout()
        
        if save:
            plot_path = settings.plots_dir / 'results.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print_success(f"Saved: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print_error(f"Plot creation failed: {e}")
        traceback.print_exc()


# =============================================================================
# REPORTING
# =============================================================================

def print_final_report(val_results: pd.DataFrame, backtest_result, importance: pd.DataFrame):
    """Print final report."""
    print("\n" + "="*60)
    print("📄 FINAL REPORT")
    print("="*60)
    
    # Find IC column
    ic_col = None
    for col in ['ic', 'test_ic', 'rank_ic', 'test_rank_ic', 'IC']:
        if col in val_results.columns:
            ic_col = col
            break
    
    if ic_col:
        mean_ic = val_results[ic_col].mean()
        std_ic = val_results[ic_col].std()
    else:
        mean_ic = 0
        std_ic = 0
    
    ir = mean_ic / (std_ic + 1e-10)
    
    # Find hit rate column
    hit_col = None
    for col in ['hit_rate', 'test_hit_rate', 'Hit_Rate']:
        if col in val_results.columns:
            hit_col = col
            break
    
    hit_rate = val_results[hit_col].mean() if hit_col else 0
    
    print(f"""
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATION RESULTS                         │
├──────────────────────────────────────────────────────────────┤
│   Folds:              {len(val_results):>10}                              │
│   Mean IC:            {mean_ic:>10.4f}                              │
│   Std IC:             {std_ic:>10.4f}                              │
│   Information Ratio:  {ir:>10.4f}                              │
│   Mean Hit Rate:      {hit_rate:>10.1%}                              │
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
    else:
        print("\n   ⚠️ Backtest was skipped.")
    
    print("\n📊 TOP 10 FEATURES:")
    print("─" * 40)
    for i, row in importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance_pct']:>6.2f}%")
        
# =============================================================================
# PIPELINE STEPS
# =============================================================================

def step_load_data(args) -> pd.DataFrame:
    """Step 1: Load data."""
    print_step(1, "LOAD DATA")
    
    loader = DataLoader()
    data = loader.load()
    
    # Quick mode: use fewer stocks
    if args.quick:
        tickers = data['ticker'].unique()[:20]  # Only 20 stocks
        data = data[data['ticker'].isin(tickers)]
        print_warning(f"Quick mode: Using only {len(tickers)} stocks")
    
    print_success(f"Loaded {len(data):,} rows, {data['ticker'].nunique()} stocks")
    print_info(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data


def step_feature_engineering(data: pd.DataFrame, args) -> tuple:
    """Step 2: Feature engineering."""
    print_step(2, "FEATURE ENGINEERING")
    
    # compute_all_features returns DataFrame only (not tuple)
    features_df = compute_all_features(data)
    
    # Auto-detect feature names
    non_feature_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'forward_return'}
    feature_names = [c for c in features_df.columns if c not in non_feature_cols]
    
    print_success(f"Created {len(feature_names)} features")
    print_success(f"Feature matrix: {len(features_df):,} rows")
    
    # Optionally save features
    if hasattr(args, 'save_features') and args.save_features:
        features_path = settings.output_dir / 'features.csv'
        features_df.to_csv(features_path, index=False)
        print_info(f"Saved features to: {features_path}")
    
    return features_df, feature_names


def step_validation(features_df: pd.DataFrame, feature_names: list, args) -> tuple:
    """Step 3: Walk-forward validation."""
    print_step(3, "WALK-FORWARD VALIDATION")
    
    validator = WalkForwardValidator(feature_names=feature_names)
    
    # Run validation
    _ = validator.train_and_validate(features_df)
    
    # Convert FoldResult objects to DataFrame - flatten metrics
    fold_dicts = []
    for fr in validator.fold_results:
        d = fr.to_dict()
        # Flatten the nested 'metrics' dict
        if 'metrics' in d and isinstance(d['metrics'], dict):
            metrics = d.pop('metrics')
            d.update(metrics)  # Add all metric keys to main dict
        fold_dicts.append(d)
    
    val_results = pd.DataFrame(fold_dicts)
    
    # Get predictions and model
    predictions = validator.predict_out_of_sample(features_df)
    model = validator.get_latest_model()
    
    # Print columns for debugging
    print(f"   📋 Val results columns: {list(val_results.columns)}")
    
    # Calculate mean IC - try different column names
    ic_col = None
    for col in ['ic', 'test_ic', 'rank_ic', 'test_rank_ic', 'IC']:
        if col in val_results.columns:
            ic_col = col
            break
    
    if ic_col:
        mean_ic = val_results[ic_col].mean()
    else:
        mean_ic = 0
    
    print_success(f"Validation complete: {len(val_results)} folds")
    print_success(f"Mean IC: {mean_ic:.4f}")
    
    return val_results, predictions, model


def step_feature_importance(model) -> pd.DataFrame:
    """Step 4: Extract feature importance."""
    print_step(4, "FEATURE IMPORTANCE")
    
    importance = model.get_feature_importance()
    
    print("\n   Top 10 Features:")
    for i, row in importance.head(10).iterrows():
        print(f"      {i+1:2d}. {row['feature']:<25} {row['importance_pct']:>6.2f}%")
    
    return importance


def step_save_model(model):
    """Step 4b: Save model."""
    settings.create_dirs()
    model_path = settings.models_dir / 'alpha_model.pkl'
    model.save(str(model_path))
    print_success(f"Model saved: {model_path}")


def step_backtest(predictions: pd.DataFrame, args):
    """Step 5: Backtesting."""
    print_step(5, "BACKTESTING")
    
    if args.skip_backtest:
        print_warning("Backtesting skipped (--skip-backtest flag)")
        return None
    
    backtester = Backtester()
    backtest_result = backtester.run(predictions)
    
    if backtest_result is not None:
        m = backtest_result.metrics
        print_success(f"Backtest complete")
        print_success(f"Total Return: {m['total_return']:.1%}")
        print_success(f"Sharpe Ratio: {m['sharpe_ratio']:.2f}")
    
    return backtest_result


def step_visualization(val_results, backtest_result, importance, args):
    """Step 6: Create visualizations."""
    print_step(6, "VISUALIZATION")
    
    if args.no_plots:
        print_warning("Plots skipped (--no-plots flag)")
        return
    
    create_plots(val_results, backtest_result, importance)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline - runs everything from data loading to backtesting."""
    
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    
    start_time = datetime.now()
    
    # Show current configuration
    settings.print_config()
    
    if args.quick:
        print("\n⚡ QUICK MODE ENABLED - Using reduced dataset")
    
    try:
        # =========================================================
        # STEP 1: Load Data
        # =========================================================
        data = step_load_data(args)
        
        # =========================================================
        # STEP 2: Feature Engineering
        # =========================================================
        features_df, feature_names = step_feature_engineering(data, args)
        
        # =========================================================
        # STEP 3: Walk-Forward Validation
        # =========================================================
        val_results, predictions, model = step_validation(features_df, feature_names, args)
        
        # =========================================================
        # STEP 4: Feature Importance & Save Model
        # =========================================================
        importance = step_feature_importance(model)
        step_save_model(model)
        
        # =========================================================
        # STEP 5: Backtesting
        # =========================================================
        backtest_result = step_backtest(predictions, args)
        
        # =========================================================
        # STEP 6: Visualization
        # =========================================================
        step_visualization(val_results, backtest_result, importance, args)
        
        # =========================================================
        # STEP 7: Final Report
        # =========================================================
        print_final_report(val_results, backtest_result, importance)
        
        # =========================================================
        # COMPLETION
        # =========================================================
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"   ⏱️  Total Time: {elapsed}")
        print(f"\n   📁 Output Files:")
        print(f"      • {settings.models_dir / 'alpha_model.pkl'}")
        if not args.no_plots:
            print(f"      • {settings.plots_dir / 'results.png'}")
        
        return {
            'validation': val_results,
            'predictions': predictions,
            'backtest': backtest_result,
            'importance': importance,
            'model': model,
            'success': True
        }
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        return {'success': False, 'error': 'Interrupted'}
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ PIPELINE FAILED!")
        print("="*60)
        print(f"\n   Error: {e}")
        print("\n   Traceback:")
        traceback.print_exc()
        
        return {'success': False, 'error': str(e)}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)