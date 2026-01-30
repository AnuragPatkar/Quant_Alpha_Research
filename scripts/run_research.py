"""
Run Research Pipeline
====================
Complete ML Alpha Model research pipeline.
Execute end-to-end: Data -> Features -> Model -> Results

Author: Anurag Patkar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings('ignore')

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Import utilities
from scripts.utils import (
    Timer, print_header, print_section, save_results, ensure_dir,
    validate_dataframe, check_file_exists
)

# Import config
try:
    from config import settings, print_welcome
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    print("⚠️ Settings not available")

# Import quant_alpha modules
try:
    from quant_alpha.data import DataLoader
    from quant_alpha.features.registry import FactorRegistry
    from quant_alpha.models import run_walk_forward_validation
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"⚠️ Some modules not available: {e}")


class ResearchPipeline:
    """Complete ML Alpha Research Pipeline."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None
        
        # Results
        self.raw_data = None
        self.features_df = None
        self.feature_names = None
        self.validation_results = None
        self.importance_df = None
        self.trainer = None
        
        # Paths
        if SETTINGS_AVAILABLE:
            self.results_dir = settings.results_dir
            self.processed_dir = settings.data.processed_dir
        else:
            self.results_dir = ROOT / "output" / "results"
            self.processed_dir = ROOT / "data" / "processed"
        
        ensure_dir(self.results_dir)
        ensure_dir(self.processed_dir)
    
    def run(self, save_intermediate: bool = True) -> bool:
        """Execute complete research pipeline."""
        self.start_time = time.time()
        
        try:
            # Welcome
            if self.verbose and SETTINGS_AVAILABLE:
                print_welcome()
                settings.setup()
                if not settings.validate_config():
                    print("❌ Configuration validation failed!")
                    return False
                settings.print_config()
            
            # Step 1: Data Loading
            if not self._step_load_data():
                return False
            
            # Step 2: Feature Engineering
            if not self._step_engineer_features(save_intermediate):
                return False
            
            # Step 3: Model Training
            if not self._step_train_model():
                return False
            
            # Step 4: Results Analysis
            if not self._step_analyze_results(save_intermediate):
                return False
            
            # Step 5: Generate Report
            self._step_generate_report()
            
            # Completion
            self._print_completion()
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _step_load_data(self) -> bool:
        """Step 1: Load and validate data."""
        if self.verbose:
            print_header("STEP 1: DATA LOADING", char="🔄 ")
        
        if not MODULES_AVAILABLE:
            print("❌ Data loader not available")
            return False
        
        with Timer("Loading data", self.verbose):
            loader = DataLoader()
            self.raw_data = loader.data
        
        if self.raw_data is None or len(self.raw_data) == 0:
            print("❌ No data loaded!")
            return False
        
        if self.verbose:
            print(f"✅ Data loaded: {self.raw_data.shape}")
            print(f"   📈 Stocks: {self.raw_data['ticker'].nunique()}")
            print(f"   📅 Period: {self.raw_data['date'].min().date()} to {self.raw_data['date'].max().date()}")
        
        return True
    
    def _step_engineer_features(self, save_intermediate: bool) -> bool:
        """Step 2: Engineer features."""
        if self.verbose:
            print_header("STEP 2: FEATURE ENGINEERING", char="🔄 ")
        
        with Timer("Computing features", self.verbose):
            registry = FactorRegistry()
            self.features_df = registry.compute_features(self.raw_data)
            self.feature_names = registry.get_feature_names()
        
        if self.features_df is None or len(self.features_df) == 0:
            print("❌ Feature engineering failed!")
            return False
        
        if self.verbose:
            print(f"✅ Features computed: {len(self.feature_names)} features")
            print(f"   📊 Dataset: {self.features_df.shape}")
        
        if save_intermediate:
            path = self.processed_dir / "features_dataset.pkl"
            self.features_df.to_pickle(path)
            print(f"   💾 Saved: {path.name}")
        
        return True
    
    def _step_train_model(self) -> bool:
        """Step 3: Train model with walk-forward validation."""
        if self.verbose:
            print_header("STEP 3: MODEL TRAINING & VALIDATION", char="🔄 ")
        
        with Timer("Training model", self.verbose):
            self.validation_results, self.trainer = run_walk_forward_validation(
                self.features_df, self.feature_names
            )
        
        if self.validation_results is None or len(self.validation_results) == 0:
            print("❌ Model training failed!")
            return False
        
        if self.verbose:
            print(f"✅ Training completed: {len(self.validation_results)} folds")
            if 'test_ic' in self.validation_results.columns:
                avg_ic = self.validation_results['test_ic'].mean()
                print(f"   📊 Average IC: {avg_ic:.4f}")
        
        return True
    
    def _step_analyze_results(self, save_intermediate: bool) -> bool:
        """Step 4: Analyze and save results."""
        if self.verbose:
            print_header("STEP 4: RESULTS & ANALYSIS", char="🔄 ")
        
        # Save validation results
        results_path = self.results_dir / "validation_results.csv"
        if self.trainer:
            self.trainer.save_results(str(results_path))
        else:
            self.validation_results.to_csv(results_path, index=False)
        
        # Feature importance
        if self.trainer:
            self.importance_df = self.trainer.get_feature_importance_summary()
            importance_path = self.results_dir / "feature_importance.csv"
            self.importance_df.to_csv(importance_path, index=False)
            
            if self.verbose:
                print(f"✅ Analysis completed!")
                print(f"   📊 Validation: {results_path.name}")
                print(f"   🔧 Importance: {importance_path.name}")
                
                if len(self.importance_df) > 0:
                    print(f"\n   Top 5 Features:")
                    for i, row in self.importance_df.head(5).iterrows():
                        print(f"      {i+1}. {row['feature']}: {row['importance_mean']:.4f}")
        
        return True
    
    def _step_generate_report(self):
        """Step 5: Generate summary report."""
        if self.verbose:
            print_header("STEP 5: SUMMARY REPORT", char="🔄 ")
        
        report_path = self.results_dir / "research_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_report(f)
        
        if self.verbose:
            print(f"   📄 Report: {report_path.name}")
    
    def _write_report(self, f):
        """Write report content."""
        f.write("=" * 80 + "\n")
        f.write("ML ALPHA MODEL - RESEARCH SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution time: {(time.time() - self.start_time)/60:.1f} minutes\n\n")
        
        # Configuration
        if SETTINGS_AVAILABLE:
            f.write("CONFIGURATION\n" + "-" * 40 + "\n")
            f.write(f"Universe: {settings.data.universe}\n")
            f.write(f"Date range: {settings.data.start_date} to {settings.data.end_date}\n")
            f.write(f"Forward return days: {settings.features.forward_return_days}\n")
            f.write(f"Model type: {settings.model.model_type}\n\n")
        
        # Data Summary
        f.write("DATA SUMMARY\n" + "-" * 40 + "\n")
        if self.features_df is not None:
            f.write(f"Total observations: {len(self.features_df):,}\n")
            f.write(f"Unique stocks: {self.features_df['ticker'].nunique()}\n")
            f.write(f"Features created: {len(self.feature_names)}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE\n" + "-" * 40 + "\n")
        if self.validation_results is not None and len(self.validation_results) > 0:
            for metric in ['test_ic', 'test_rank_ic', 'test_hit_rate']:
                if metric in self.validation_results.columns:
                    mean_val = self.validation_results[metric].mean()
                    std_val = self.validation_results[metric].std()
                    f.write(f"{metric:15s}: {mean_val:7.4f} +/- {std_val:6.4f}\n")
            
            avg_ic = self.validation_results['test_ic'].mean()
            f.write(f"\nInterpretation: ")
            if avg_ic > 0.05:
                f.write("EXCELLENT - Strong predictive power\n")
            elif avg_ic > 0.02:
                f.write("GOOD - Moderate predictive power\n")
            elif avg_ic > 0:
                f.write("FAIR - Weak but positive\n")
            else:
                f.write("POOR - No predictive power\n")
        
        # Top Features
        f.write("\nTOP 10 FEATURES\n" + "-" * 40 + "\n")
        if self.importance_df is not None and len(self.importance_df) > 0:
            for idx, row in self.importance_df.head(10).iterrows():
                f.write(f"{row['feature']:25s}: {row['importance_mean']:8.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
    
    def _print_completion(self):
        """Print completion summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "🎉 " * 15)
        print("RESEARCH PIPELINE COMPLETED!")
        print("🎉 " * 15)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results: {self.results_dir}")
        print(f"📊 Files: validation_results.csv, feature_importance.csv, research_summary.txt")


def quick_test():
    """Quick test with small dataset."""
    print("🧪 Running quick test...")
    try:
        loader = DataLoader()
        raw_data = loader.data.head(1000)
        registry = FactorRegistry()
        features_df = registry.compute_features(raw_data)
        print(f"✅ Quick test completed: {features_df.shape}")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def generate_report_only():
    """Generate report from existing results."""
    print("📄 Generating report from existing results...")
    
    if not SETTINGS_AVAILABLE:
        print("❌ Settings not available")
        return False
    
    try:
        if not all([
            (settings.results_dir / "validation_results.csv").exists(),
            (settings.results_dir / "feature_importance.csv").exists(),
            (settings.data.processed_dir / "features_dataset.pkl").exists()
        ]):
            print("❌ Missing files! Run full pipeline first.")
            return False
        
        pipeline = ResearchPipeline()
        pipeline.validation_results = pd.read_csv(settings.results_dir / "validation_results.csv")
        pipeline.importance_df = pd.read_csv(settings.results_dir / "feature_importance.csv")
        pipeline.features_df = pd.read_pickle(settings.data.processed_dir / "features_dataset.pkl")
        pipeline.feature_names = pipeline.importance_df['feature'].tolist() if len(pipeline.importance_df) > 0 else []
        pipeline.start_time = time.time()
        pipeline._step_generate_report()
        
        print("✅ Report generated!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='ML Alpha Research Pipeline')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--no-save', action='store_true', help='Skip saving intermediate')
    
    args = parser.parse_args()
    
    if args.test:
        return 0 if quick_test() else 1
    elif args.report_only:
        return 0 if generate_report_only() else 1
    else:
        pipeline = ResearchPipeline(verbose=not args.quiet)
        return 0 if pipeline.run(save_intermediate=not args.no_save) else 1


if __name__ == "__main__":
    sys.exit(main())