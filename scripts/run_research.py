"""
Run Research Pipeline
====================
Complete ML Alpha Model research pipeline.
Execute end-to-end: Data → Features → Model → Results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Imports
from config import settings, print_welcome
from quant_alpha.data import DataLoader
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models import run_walk_forward_validation

warnings.filterwarnings('ignore')


def main():
    """Execute complete research pipeline."""
    
    start_time = time.time()
    
    # Welcome & Setup
    print_welcome()
    settings.setup()
    
    # Validate configuration
    if not settings.validate_config():
        print("❌ Configuration validation failed!")
        return
    
    settings.print_config()
    
    try:
        # ==============================================
        # STEP 1: DATA LOADING
        # ==============================================
        print("\n" + "🔄 "*25)
        print("STEP 1: DATA LOADING")
        print("🔄 "*25)
        
        loader = DataLoader()
        raw_data = loader.data
        
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Shape: {raw_data.shape}")
        print(f"   📈 Stocks: {raw_data['ticker'].nunique()}")
        print(f"   📅 Date range: {raw_data['date'].min().date()} to {raw_data['date'].max().date()}")
        
        # ==============================================
        # STEP 2: FEATURE ENGINEERING
        # ==============================================
        print("\n" + "🔄 "*25)
        print("STEP 2: FEATURE ENGINEERING")
        print("🔄 "*25)
        
        registry = FactorRegistry()
        features_df = registry.compute_features(raw_data)
        feature_names = registry.get_feature_names()
        
        print(f"✅ Features computed successfully!")
        print(f"   🔧 Features: {len(feature_names)}")
        print(f"   📊 Dataset shape: {features_df.shape}")
        print(f"   📉 Data coverage: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
        
        # Save features dataset
        features_path = settings.data.processed_dir / "features_dataset.pkl"
        features_df.to_pickle(features_path)
        print(f"   💾 Features saved: {features_path.name}")
        
        # ==============================================
        # STEP 3: MODEL TRAINING & VALIDATION
        # ==============================================
        print("\n" + "🔄 "*25)
        print("STEP 3: MODEL TRAINING & VALIDATION")
        print("🔄 "*25)
        
        # Run walk-forward validation
        results_df, trainer = run_walk_forward_validation(features_df, feature_names)
        
        print(f"✅ Model validation completed!")
        print(f"   📈 Validation folds: {len(results_df)}")
        
        # ==============================================
        # STEP 4: RESULTS & ANALYSIS
        # ==============================================
        print("\n" + "🔄 "*25)
        print("STEP 4: RESULTS & ANALYSIS")
        print("🔄 "*25)
        
        # Save validation results
        results_path = settings.results_dir / "validation_results.csv"
        trainer.save_results(str(results_path))
        
        # Feature importance analysis
        importance_df = trainer.get_feature_importance_summary()
        importance_path = settings.results_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        print(f"✅ Analysis completed!")
        print(f"   📊 Validation results: {results_path.name}")
        print(f"   🔧 Feature importance: {importance_path.name}")
        
        # ==============================================
        # STEP 5: SUMMARY REPORT
        # ==============================================
        print("\n" + "🔄 "*25)
        print("STEP 5: SUMMARY REPORT")
        print("🔄 "*25)
        
        generate_summary_report(results_df, importance_df, features_df, start_time)
        
        # ==============================================
        # COMPLETION
        # ==============================================
        total_time = time.time() - start_time
        print("\n" + "🎉 "*25)
        print("RESEARCH PIPELINE COMPLETED!")
        print("🎉 "*25)
        print(f"⏱️  Total execution time: {total_time/60:.1f} minutes")
        print(f"📁 Results directory: {settings.results_dir}")
        print(f"📊 Key files created:")
        print(f"   • {results_path.name}")
        print(f"   • {importance_path.name}")
        print(f"   • features_dataset.pkl")
        print(f"   • research_summary.txt")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def generate_summary_report(results_df: pd.DataFrame, 
                          importance_df: pd.DataFrame,
                          features_df: pd.DataFrame,
                          start_time: float):
    """Generate comprehensive summary report (Unicode fixed)."""
    
    report_path = settings.results_dir / "research_summary.txt"
    
    # Fix: Use UTF-8 encoding for Windows compatibility
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ML ALPHA MODEL - RESEARCH SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution time: {(time.time() - start_time)/60:.1f} minutes\n\n")
        
        # Configuration Summary
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Universe: {settings.data.universe}\n")
        f.write(f"Date range: {settings.data.start_date} to {settings.data.end_date}\n")  # Fixed: removed →
        f.write(f"Forward return days: {settings.features.forward_return_days}\n")
        f.write(f"Model type: {settings.model.model_type}\n")
        f.write(f"Train window: {settings.validation.train_months} months\n")
        f.write(f"Test window: {settings.validation.test_months} months\n\n")
        
        # Data Summary
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total observations: {len(features_df):,}\n")
        f.write(f"Unique stocks: {features_df['ticker'].nunique()}\n")
        f.write(f"Date range: {features_df['date'].min().date()} to {features_df['date'].max().date()}\n")  # Fixed: removed →
        f.write(f"Features created: {len(importance_df)}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        if len(results_df) > 0:
            metrics = ['test_ic', 'test_rank_ic', 'test_hit_rate', 'test_rmse']
            for metric in metrics:
                if metric in results_df.columns:
                    mean_val = results_df[metric].mean()
                    std_val = results_df[metric].std()
                    f.write(f"{metric:15s}: {mean_val:7.4f} +/- {std_val:6.4f}\n")  # Fixed: removed ±
            
            f.write(f"\nValidation folds: {len(results_df)}\n")
            f.write(f"Best IC: {results_df['test_ic'].max():.4f}\n")
            f.write(f"Worst IC: {results_df['test_ic'].min():.4f}\n")
            
            # Performance interpretation
            f.write(f"\nPERFORMANCE INTERPRETATION\n")
            f.write("-" * 40 + "\n")
            avg_ic = results_df['test_ic'].mean()
            if avg_ic > 0.05:
                f.write("EXCELLENT: Strong predictive power\n")
            elif avg_ic > 0.02:
                f.write("GOOD: Moderate predictive power\n")
            elif avg_ic > 0:
                f.write("FAIR: Weak but positive predictive power\n")
            else:
                f.write("POOR: No predictive power\n")
                
        else:
            f.write("No validation results available.\n")
        
        # Top Features
        f.write("\nTOP 10 FEATURES\n")
        f.write("-" * 40 + "\n")
        if len(importance_df) > 0:
            top_features = importance_df.head(10)
            for idx, row in top_features.iterrows():
                f.write(f"{row['feature']:20s}: {row['importance_mean']:8.4f}\n")
        else:
            f.write("No feature importance data available.\n")
        
        # Feature Categories
        f.write("\nFEATURE CATEGORIES\n")
        f.write("-" * 40 + "\n")
        if len(importance_df) > 0:
            categories = {
                'Momentum': [f for f in importance_df['feature'] if 'mom' in f or 'roc' in f or 'ema' in f],
                'Mean Reversion': [f for f in importance_df['feature'] if 'rsi' in f or 'dist' in f or 'zscore' in f or 'bb' in f],
                'Volatility': [f for f in importance_df['feature'] if 'volatility' in f or 'atr' in f or 'vol_ratio' in f or 'skew' in f],
                'Volume': [f for f in importance_df['feature'] if 'volume' in f or 'pv_' in f or 'amihud' in f or 'relative' in f]
            }
            
            for category, features in categories.items():
                f.write(f"{category:15s}: {len(features):2d} features\n")
        
        # Model Stability Analysis
        f.write("\nMODEL STABILITY\n")
        f.write("-" * 40 + "\n")
        if len(results_df) > 0:
            ic_std = results_df['test_ic'].std()
            positive_folds = (results_df['test_ic'] > 0).sum()
            total_folds = len(results_df)
            
            f.write(f"IC Standard Deviation: {ic_std:.4f}\n")
            f.write(f"Positive IC Folds: {positive_folds}/{total_folds} ({positive_folds/total_folds*100:.1f}%)\n")
            
            if ic_std < 0.05:
                f.write("STABLE: Low IC volatility\n")
            elif ic_std < 0.10:
                f.write("MODERATE: Medium IC volatility\n")
            else:
                f.write("UNSTABLE: High IC volatility\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"   📄 Summary report: {report_path.name}")


def quick_test():
    """Quick test with small dataset."""
    print("🧪 Running quick test...")
    
    # Load small sample
    loader = DataLoader()
    raw_data = loader.data.head(1000)  # Small sample
    
    # Features
    registry = FactorRegistry()
    features_df = registry.compute_features(raw_data)
    
    print(f"✅ Quick test completed: {features_df.shape}")


def generate_report_only():
    """Generate report from existing results (if pipeline already ran)."""
    print("📄 Generating report from existing results...")
    
    try:
        # Load existing results
        results_path = settings.results_dir / "validation_results.csv"
        importance_path = settings.results_dir / "feature_importance.csv"
        features_path = settings.data.processed_dir / "features_dataset.pkl"
        
        if not all([results_path.exists(), importance_path.exists(), features_path.exists()]):
            print("❌ Missing result files! Run full pipeline first.")
            return
        
        results_df = pd.read_csv(results_path)
        importance_df = pd.read_csv(importance_path)
        features_df = pd.read_pickle(features_path)
        
        # Generate report
        generate_summary_report(results_df, importance_df, features_df, 0)
        print("✅ Report generated successfully!")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Alpha Research Pipeline')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--report-only', action='store_true', help='Generate report from existing results')
    parser.add_argument('--config', type=str, help='Custom config file')
    
    args = parser.parse_args()
    
    if args.test:
        quick_test()
    elif args.report_only:
        generate_report_only()
    else:
        success = main()
        if success:
            print("\n🎯 Research pipeline completed successfully!")
        else:
            print("\n💥 Research pipeline failed!")
            sys.exit(1)