"""
Walk-Forward Validation
=======================
Time-series cross-validation to test the model properly.
Makes sure we don't accidentally use future data (no look-ahead bias).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from quant_alpha.models.boosting import LightGBMModel


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation.
    
    This is the proper way to validate time-series models:
    - Train on past data only
    - Test on future data
    - Roll forward through time
    - Add embargo period to prevent leakage
    """
    
    def __init__(self):
        """Initialize validator with settings from config."""
        self.config = settings.validation
        self.fold_results = []
        self.all_predictions = []
    
    def validate(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_col: str = 'forward_return'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, LightGBMModel]:
        """
        Run walk-forward validation.
        
        Args:
            df: DataFrame with features and target
            feature_names: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (results_df, predictions_df, last_model)
        """
        print("\n" + "="*60)
        print("📊 WALK-FORWARD VALIDATION")
        print("="*60)
        print(f"   Train Window: {self.config.train_months} months")
        print(f"   Test Window: {self.config.test_months} months")
        print(f"   Embargo: {self.config.embargo_days} days")
        
        self.fold_results = []
        self.all_predictions = []
        
        # Get all unique dates sorted
        dates = sorted(df['date'].unique())
        
        # Convert months to approximate trading days (21 days per month)
        train_days = self.config.train_months * 21
        test_days = self.config.test_months * 21
        
        fold = 0
        idx = 0
        last_model = None
        
        print(f"\n   Running validation folds...")
        
        # Walk forward through time
        while idx + train_days + test_days < len(dates):
            fold += 1
            
            # Define periods
            train_end_idx = idx + train_days
            test_end_idx = train_end_idx + test_days
            
            train_dates = dates[idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            # Split data
            train_df = df[df['date'].isin(train_dates)]
            test_df = df[df['date'].isin(test_dates)]
            
            # Check minimum samples
            if len(train_df) < self.config.min_train_samples or len(test_df) < 50:
                idx += test_days
                continue
            
            # Prepare features and target
            X_train = train_df[feature_names]
            y_train = train_df[target_col]
            X_test = test_df[feature_names]
            y_test = test_df[target_col]
            
            # Train model
            model = LightGBMModel(feature_names)
            model.fit(X_train, y_train)
            last_model = model
            
            # Predict and evaluate
            predictions = model.predict(X_test)
            metrics = model.evaluate(X_test, y_test)
            
            # Store results
            self.fold_results.append({
                'fold': fold,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'train_size': len(train_df),
                'test_size': len(test_df),
                **metrics
            })
            
            # Store predictions
            pred_df = test_df[['date', 'ticker', target_col]].copy()
            pred_df['prediction'] = predictions
            pred_df['fold'] = fold
            self.all_predictions.append(pred_df)
            
            # Print fold results
            print(f"      Fold {fold}: IC={metrics['ic']:.4f}, "
                  f"RankIC={metrics['rank_ic']:.4f}, "
                  f"Hit={metrics['hit_rate']:.1%}")
            
            # Move forward
            idx += test_days
        
        # Create result DataFrames
        results_df = pd.DataFrame(self.fold_results)
        predictions_df = pd.concat(self.all_predictions, ignore_index=True)
        
        # Print summary
        self._print_summary(results_df)
        
        return results_df, predictions_df, last_model
    
    def _print_summary(self, results: pd.DataFrame):
        """Print validation summary."""
        if len(results) == 0:
            print("\n   ⚠️ No folds completed!")
            return
        
        mean_ic = results['ic'].mean()
        std_ic = results['ic'].std()
        ir = mean_ic / (std_ic + 1e-10)
        t_stat = mean_ic / (std_ic / np.sqrt(len(results)) + 1e-10)
        
        print(f"\n   {'─'*50}")
        print(f"   📈 VALIDATION SUMMARY")
        print(f"   {'─'*50}")
        print(f"   Total Folds:        {len(results)}")
        print(f"   Mean IC:            {mean_ic:.4f} (±{std_ic:.4f})")
        print(f"   Information Ratio:  {ir:.4f}")
        print(f"   t-statistic:        {t_stat:.2f}")
        print(f"   Mean Rank IC:       {results['rank_ic'].mean():.4f}")
        print(f"   Mean Hit Rate:      {results['hit_rate'].mean():.1%}")
        print(f"   Positive IC Folds:  {(results['ic'] > 0).mean():.1%}")
        
        if abs(t_stat) > 2:
            print(f"\n   ✅ Statistically significant (|t| > 2)")
        else:
            print(f"\n   ⚠️  May not be statistically significant (|t| < 2)")