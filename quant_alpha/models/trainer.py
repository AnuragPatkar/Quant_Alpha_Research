"""
Model Trainer
=============
Walk-forward validation for time series ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys
from tqdm import tqdm
import warnings

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from quant_alpha.models.boosting import LightGBMModel

warnings.filterwarnings('ignore')


class WalkForwardTrainer:
    """
    Walk-forward validation trainer for time series data.
    
    Features:
        - Time-aware train/test splits
        - Embargo period to prevent look-ahead bias
        - Model persistence
        - Performance tracking
        - Feature importance analysis
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize trainer.
        
        Args:
            feature_names: List of feature column names
        """
        self.feature_names = feature_names
        self.config = settings.validation
        self.results = []
        self.models = {}
        self.feature_importance_history = []
        
        print(f"🎯 Walk-Forward Trainer initialized")
        print(f"   📊 Features: {len(feature_names)}")
        print(f"   🔧 Train window: {self.config.train_months} months")
        print(f"   📈 Test window: {self.config.test_months} months")
        print(f"   ⏭️  Step size: {self.config.step_months} months")
        print(f"   🚫 Embargo: {self.config.embargo_days} days")
    
    def create_time_splits(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Create time-based train/test splits.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        df['date'] = pd.to_datetime(df['date'])
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        splits = []
        current_date = min_date
        
        while True:
            # Training period
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_months * 30)
            
            # Embargo period
            embargo_end = train_end + timedelta(days=self.config.embargo_days)
            
            # Test period
            test_start = embargo_end
            test_end = test_start + timedelta(days=self.config.test_months * 30)
            
            # Check if we have enough data
            if test_end > max_date:
                break
            
            # Check minimum samples
            train_samples = len(df[(df['date'] >= train_start) & (df['date'] <= train_end)])
            test_samples = len(df[(df['date'] >= test_start) & (df['date'] <= test_end)])
            
            if (train_samples >= self.config.min_train_samples and 
                test_samples >= self.config.min_test_samples):
                splits.append((train_start, train_end, test_start, test_end))
            
            # Move forward
            current_date += timedelta(days=self.config.step_months * 30)
        
        return splits
    
    def train_and_validate(self, df: pd.DataFrame, save_models: bool = True) -> pd.DataFrame:
        """
        Run walk-forward validation.
        
        Args:
            df: Features DataFrame with 'date', 'ticker', features, 'forward_return'
            save_models: Whether to save trained models
            
        Returns:
            DataFrame with validation results
        """
        print("\n" + "="*70)
        print("🎯 WALK-FORWARD VALIDATION")
        print("="*70)
        
        # Validate input
        required_cols = ['date', 'ticker', 'forward_return'] + self.feature_names
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Create time splits
        splits = self.create_time_splits(df)
        print(f"📅 Created {len(splits)} time splits")
        
        if len(splits) == 0:
            raise ValueError("No valid time splits found! Check data range and config.")
        
        # Reset results
        self.results = []
        self.models = {}
        
        # Run validation
        for i, (train_start, train_end, test_start, test_end) in enumerate(tqdm(splits, desc="Training models")):
            
            # Prepare data
            train_data = df[(df['date'] >= train_start) & (df['date'] <= train_end)].copy()
            test_data = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
            
            # Remove NaN targets
            train_data = train_data.dropna(subset=['forward_return'])
            test_data = test_data.dropna(subset=['forward_return'])
            
            if len(train_data) < 100 or len(test_data) < 20:
                continue
            
            # Prepare features and targets
            X_train = train_data[self.feature_names]
            y_train = train_data['forward_return']
            X_test = test_data[self.feature_names]
            y_test = test_data['forward_return']
            
            # Create and train model
            model = LightGBMModel(self.feature_names)
            
            # Use 20% of training data for validation
            val_split = int(0.8 * len(X_train))
            X_train_fit = X_train.iloc[:val_split]
            y_train_fit = y_train.iloc[:val_split]
            X_val = X_train.iloc[val_split:]
            y_val = y_train.iloc[val_split:]
            
            # Train model
            model.fit(X_train_fit, y_train_fit, X_val, y_val, verbose=False)
            
            # Evaluate on test set
            test_metrics = model.evaluate(X_test, y_test, prefix='test_')
            train_metrics = model.evaluate(X_train, y_train, prefix='train_')
            
            # Store results
            result = {
                'fold': i,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                **train_metrics,
                **test_metrics
            }
            
            self.results.append(result)
            
            # Save model if requested
            if save_models:
                model_name = f"model_fold_{i}"
                self.models[model_name] = model
                
                if settings.save_models:
                    model_path = settings.models_dir / f"{model_name}.joblib"
                    model.save(str(model_path))
            
            # Store feature importance
            importance_df = model.get_feature_importance()
            importance_df['fold'] = i
            importance_df['test_start'] = test_start
            self.feature_importance_history.append(importance_df)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Print summary
        self._print_validation_summary(results_df)
        
        return results_df
    
    def _print_validation_summary(self, results_df: pd.DataFrame):
        """Print validation summary statistics."""
        print(f"\n📊 VALIDATION SUMMARY")
        print("="*50)
        
        if len(results_df) == 0:
            print("❌ No validation results!")
            return
        
        # Key metrics
        metrics = ['test_ic', 'test_rank_ic', 'test_hit_rate', 'test_rmse']
        
        print(f"📈 Folds completed: {len(results_df)}")
        print(f"📅 Date range: {results_df['test_start'].min().date()} → {results_df['test_end'].max().date()}")
        
        print(f"\n📊 Performance Metrics:")
        for metric in metrics:
            if metric in results_df.columns:
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                print(f"   {metric:15s}: {mean_val:7.4f} ± {std_val:6.4f}")
        
        # Best and worst folds
        if 'test_ic' in results_df.columns:
            best_fold = results_df.loc[results_df['test_ic'].idxmax()]
            worst_fold = results_df.loc[results_df['test_ic'].idxmin()]
            
            print(f"\n🏆 Best fold: {best_fold['fold']} (IC: {best_fold['test_ic']:.4f})")
            print(f"💥 Worst fold: {worst_fold['fold']} (IC: {worst_fold['test_ic']:.4f})")
        
        print("="*50)
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get aggregated feature importance across all folds."""
        if not self.feature_importance_history:
            return pd.DataFrame()
        
        # Combine all importance data
        all_importance = pd.concat(self.feature_importance_history, ignore_index=True)
        
        # Calculate summary statistics
        summary = all_importance.groupby('feature').agg({
            'importance': ['mean', 'std', 'count'],
            'importance_pct': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        # Sort by mean importance
        summary = summary.sort_values('importance_mean', ascending=False)
        
        return summary.reset_index(drop=True)
    
    def get_best_model(self) -> Optional[LightGBMModel]:
        """Get the model from the best performing fold."""
        if not self.results:
            return None
        
        results_df = pd.DataFrame(self.results)
        if 'test_ic' not in results_df.columns:
            return None
        
        best_fold = results_df.loc[results_df['test_ic'].idxmax(), 'fold']
        model_name = f"model_fold_{best_fold}"
        
        return self.models.get(model_name)
    
    def save_results(self, path: str):
        """Save validation results to file."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(path, index=False)
        print(f"💾 Results saved: {path}")
    
    def get_cross_sectional_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional metrics for each time period.
        
        Args:
            df: DataFrame with predictions and actual returns
            
        Returns:
            DataFrame with cross-sectional metrics by date
        """
        if 'predictions' not in df.columns or 'forward_return' not in df.columns:
            raise ValueError("DataFrame must have 'predictions' and 'forward_return' columns")
        
        def calc_cross_sectional_ic(group):
            pred = group['predictions']
            actual = group['forward_return']
            
            # Remove NaN
            mask = ~(pred.isna() | actual.isna())
            if mask.sum() < 5:  # Need at least 5 stocks
                return pd.Series({
                    'ic': np.nan,
                    'rank_ic': np.nan,
                    'hit_rate': np.nan,
                    'n_stocks': 0
                })
            
            pred_clean = pred[mask]
            actual_clean = actual[mask]
            
            # Calculate metrics
            ic = np.corrcoef(pred_clean, actual_clean)[0, 1]
            rank_ic = pd.Series(pred_clean).corr(pd.Series(actual_clean), method='spearman')
            hit_rate = np.mean(np.sign(pred_clean) == np.sign(actual_clean))
            
            return pd.Series({
                'ic': ic if not np.isnan(ic) else 0,
                'rank_ic': rank_ic if not np.isnan(rank_ic) else 0,
                'hit_rate': hit_rate,
                'n_stocks': len(pred_clean)
            })
        
        return df.groupby('date').apply(calc_cross_sectional_ic).reset_index()
    
    def predict_out_of_sample(self, df: pd.DataFrame, model: LightGBMModel = None) -> pd.DataFrame:
        """
        Generate out-of-sample predictions using the best model.
        
        Args:
            df: DataFrame with features
            model: Trained model (uses best model if None)
            
        Returns:
            DataFrame with predictions added
        """
        if model is None:
            model = self.get_best_model()
            if model is None:
                raise ValueError("No trained model available!")
        
        # Make predictions
        predictions = model.predict(df[self.feature_names])
        
        # Add to DataFrame
        result_df = df.copy()
        result_df['predictions'] = predictions
        
        return result_df


def run_walk_forward_validation(features_df: pd.DataFrame, 
                               feature_names: List[str]) -> Tuple[pd.DataFrame, WalkForwardTrainer]:
    """
    Convenience function to run walk-forward validation.
    
    Args:
        features_df: DataFrame with features and targets
        feature_names: List of feature column names
        
    Returns:
        Tuple of (results_df, trainer)
    """
    trainer = WalkForwardTrainer(feature_names)
    results_df = trainer.train_and_validate(features_df)
    return results_df, trainer


def create_trainer(feature_names: List[str]) -> WalkForwardTrainer:
    """
    Factory function to create WalkForwardTrainer.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Configured WalkForwardTrainer
    """
    return WalkForwardTrainer(feature_names)