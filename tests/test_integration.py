"""
Integration Test: End-to-End Pipeline
=====================================
Tests the entire workflow:
Data -> Features -> Validation -> Model -> Backtest -> Reporting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# =============================================================================
# TEST UTILITIES
# =============================================================================

class TestResult:
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


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_full_pipeline():
    """Run complete pipeline with minimal data."""
    print("\n" + "="*60)
    print("🚀 INTEGRATION TEST: Full Pipeline")
    print("="*60)
    
    result = TestResult()
    start_time = time.time()
    
    # =========================================================================
    # 0. Import Check
    # =========================================================================
    try:
        print("\n0. Importing Modules...")
        from quant_alpha.data import DataLoader
        from quant_alpha.features import compute_all_features
        from quant_alpha.research import WalkForwardValidator
        from quant_alpha.models.boosting import LightGBMModel
        from quant_alpha.backtest import Backtester
        from config.settings import settings
        result.success("All modules imported")
    except ImportError as e:
        result.fail("Module Import", e)
        return result.summary()
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    try:
        print("\n1. Loading Data...")
        loader = DataLoader()
        data = loader.load()
        
        # Use only 3 stocks for speed
        tickers = data['ticker'].unique()[:3]
        data = data[data['ticker'].isin(tickers)]
        
        print(f"   Using {len(tickers)} stocks: {list(tickers)}")
        print(f"   Rows: {len(data):,}")
        result.success("Data Loaded")
    except Exception as e:
        result.fail("Data Loading", e)
        return result.summary()
    
    # =========================================================================
    # 2. Feature Engineering
    # =========================================================================
    try:
        print("\n2. Computing Features...")
        
        features_df = compute_all_features(
            data,
            normalize=False,
            winsorize=False,
            add_target=True
        )
        
        # Auto-detect feature names
        non_feature_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'forward_return'}
        feature_names = [c for c in features_df.columns if c not in non_feature_cols]
        
        print(f"   Features: {len(feature_names)}")
        print(f"   Shape: {features_df.shape}")
        result.success("Feature Engineering")
    except Exception as e:
        result.fail("Feature Engineering", e)
        return result.summary()
    
    # =========================================================================
    # 3. Validation
    # =========================================================================
    try:
        print("\n3. Walk-Forward Validation...")
        
        original_min_train = settings.validation.min_train_months
        settings.validation.min_train_months = 6
        
        try:
            validator = WalkForwardValidator(feature_names=feature_names)
            _ = validator.train_and_validate(features_df)
            
            fold_results = validator.fold_results
            predictions = validator.predict_out_of_sample(features_df)
            
            print(f"   Folds: {len(fold_results)}")
            print(f"   Predictions: {len(predictions):,}")
            result.success("Validation")
        finally:
            settings.validation.min_train_months = original_min_train
            
    except Exception as e:
        result.fail("Validation", e)
        # Continue even if validation fails
    
    # =========================================================================
    # 4. Model Training
    # =========================================================================
    try:
        print("\n4. Training Final Model...")
        features_clean = features_df.dropna()
        X = features_clean[feature_names]
        y = features_clean['forward_return']
        
        model = LightGBMModel(feature_names)
        model.fit(X, y)
        
        imp = model.get_feature_importance()
        print(f"   Top feature: {imp.iloc[0]['feature']} ({imp.iloc[0]['importance_pct']:.1f}%)")
        result.success("Model Training")
    except Exception as e:
        result.fail("Model Training", e)
        return result.summary()
    
    # =========================================================================
    # 5. Backtesting
    # =========================================================================
    try:
        print("\n5. Backtesting...")
        
        original_top_n = settings.backtest.top_n_long
        settings.backtest.top_n_long = 1
        
        try:
            # Generate predictions
            full_preds = model.predict(features_df[feature_names])
            features_df['prediction'] = full_preds
            
            # Run backtest
            backtester = Backtester()
            backtest_result = backtester.run(features_df)
            
            if backtest_result is None:
                print("   ⚠️ Backtest returned None (no trades)")
                result.success("Backtesting (no trades)")
            else:
                metrics = backtest_result.metrics
                print(f"   Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Return: {metrics.get('total_return', 0):.1%}")
                result.success("Backtesting")
        finally:
            settings.backtest.top_n_long = original_top_n
            
    except Exception as e:
        result.fail("Backtesting", e)
    
    # =========================================================================
    # 6. Reporting (Optional)
    # =========================================================================
    try:
        print("\n6. Generating Report...")
        
        from quant_alpha.visualization.reports import ReportGenerator
        
        output_dir = ROOT / 'test_outputs' / 'integration_test'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if backtest_result is not None:
            report_data = {
                'metrics': backtest_result.metrics,
                'returns': backtest_result.returns,
                'equity_curve': backtest_result.equity_curve,
                'config': backtest_result.config
            }
            
            generator = ReportGenerator(output_dir=str(output_dir))
            generator.export_to_json(report_data, 'integration_test_result')
            print(f"   Report saved to: {output_dir}")
            result.success("Report Generated")
        else:
            result.success("Report Skipped (no backtest data)")
            
    except Exception as e:
        result.fail("Reporting", e)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"⏱️  Total Time: {elapsed:.1f} seconds")
    print("="*60)
    
    return result.summary()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔬 QUANT ALPHA INTEGRATION TEST")
    print("="*60)
    
    success = test_full_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("✅ INTEGRATION TEST PASSED!")
    else:
        print("❌ INTEGRATION TEST FAILED!")
    print("="*60)
    
    sys.exit(0 if success else 1)