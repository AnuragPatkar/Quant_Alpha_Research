"""
Walk-Forward Trainer
====================
Time-series cross-validation for alpha models.

Features:
- Expanding and rolling window support
- Proper embargo period to prevent lookahead bias
- Cross-sectional IC calculation
- Model persistence per fold
- Feature importance tracking

Author: [Your Name]
Last Updated: 2024

IMPORTANT NOTES:
================
1. Each fold trains a FRESH model (no data leakage)
2. Embargo period prevents target variable leakage
3. Cross-sectional IC is the proper evaluation metric
4. Use expanding window for maximum data utilization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import warnings
import json

# Proper imports with fallback
try:
    from config.settings import settings
    from quant_alpha.models.boosting import (
        LightGBMModel, 
        ModelConfig,
        calculate_cross_sectional_ic,
        calculate_information_ratio,
        calculate_all_metrics,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings
    from quant_alpha.models.boosting import (
        LightGBMModel, 
        ModelConfig,
        calculate_cross_sectional_ic,
        calculate_information_ratio,
        calculate_all_metrics,
    )


# Setup logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class FoldResult:
    """
    Results from a single walk-forward fold.
    
    Attributes:
        fold: Fold index
        train_start: Training period start date
        train_end: Training period end date
        test_start: Test period start date
        test_end: Test period end date
        train_samples: Number of training samples
        test_samples: Number of test samples
        metrics: Dictionary of evaluation metrics
        predictions: DataFrame with predictions (optional)
        feature_importance: DataFrame with feature importance (optional)
        model_path: Path to saved model file (optional)
    """
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    metrics: Dict[str, float]
    predictions: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for serialization)."""
        return {
            'fold': self.fold,
            'train_start': self.train_start.isoformat() if self.train_start else None,
            'train_end': self.train_end.isoformat() if self.train_end else None,
            'test_start': self.test_start.isoformat() if self.test_start else None,
            'test_end': self.test_end.isoformat() if self.test_end else None,
            'train_samples': self.train_samples,
            'test_samples': self.test_samples,
            'metrics': self.metrics,
            'model_path': self.model_path,
        }


@dataclass
class WalkForwardResults:
    """
    Complete walk-forward validation results.
    
    Attributes:
        fold_results: List of FoldResult objects
        aggregate_metrics: Aggregated metrics across all folds
        predictions_df: Combined predictions from all folds
        feature_importance_summary: Aggregated feature importance
        config: Configuration used for validation
    """
    fold_results: List[FoldResult]
    aggregate_metrics: Dict[str, float]
    predictions_df: Optional[pd.DataFrame]
    feature_importance_summary: Optional[pd.DataFrame]
    config: Optional[Dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def print_summary(self) -> None:
        """Print formatted summary of results."""
        print("\n" + "=" * 70)
        print("📊 WALK-FORWARD VALIDATION RESULTS")
        print("=" * 70)
        
        # Config info
        if self.config:
            window_type = "Expanding" if self.config.get('use_expanding_window', True) else "Rolling"
            print(f"\n📋 Configuration:")
            print(f"   Window Type: {window_type}")
            print(f"   Test Months: {self.config.get('test_months', 'N/A')}")
            print(f"   Embargo Days: {self.config.get('embargo_days', 'N/A')}")
        
        print(f"   Total Folds: {len(self.fold_results)}")
        
        if self.fold_results:
            print(f"   Test Range: {self.fold_results[0].test_start.date()} → "
                  f"{self.fold_results[-1].test_end.date()}")
        
        # Aggregate metrics
        print(f"\n📈 Aggregate Metrics:")
        
        # Key metrics to display first
        key_metrics = [
            'overall_mean_rank_ic', 'overall_ir', 'test_rank_ic_mean', 
            'test_ic_mean', 'test_hit_rate_mean'
        ]
        
        for metric in key_metrics:
            if metric in self.aggregate_metrics:
                value = self.aggregate_metrics[metric]
                print(f"   {metric}: {value:.4f}")
        
        # Sample counts
        print(f"\n   Total Train Samples: {self.aggregate_metrics.get('total_train_samples', 0):,}")
        print(f"   Total Test Samples: {self.aggregate_metrics.get('total_test_samples', 0):,}")
        
        # Per-fold table
        print(f"\n📋 Per-Fold Results:")
        print("-" * 80)
        print(f"{'Fold':<6} {'Test Period':<24} {'Samples':<10} {'IC':<10} {'Rank IC':<10} {'Hit Rate':<10}")
        print("-" * 80)
        
        for fr in self.fold_results:
            test_period = f"{fr.test_start.strftime('%Y-%m')} to {fr.test_end.strftime('%Y-%m')}"
            ic = fr.metrics.get('test_ic', fr.metrics.get('ic', 0))
            rank_ic = fr.metrics.get('test_rank_ic', fr.metrics.get('rank_ic', 0))
            hit_rate = fr.metrics.get('test_hit_rate', fr.metrics.get('hit_rate', 0))
            
            print(f"{fr.fold:<6} {test_period:<24} {fr.test_samples:<10} "
                  f"{ic:<10.4f} {rank_ic:<10.4f} {hit_rate:<10.4f}")
        
        print("=" * 70)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert fold results to DataFrame."""
        records = []
        for fr in self.fold_results:
            record = {
                'fold': fr.fold,
                'train_start': fr.train_start,
                'train_end': fr.train_end,
                'test_start': fr.test_start,
                'test_end': fr.test_end,
                'train_samples': fr.train_samples,
                'test_samples': fr.test_samples,
                **fr.metrics
            }
            records.append(record)
        return pd.DataFrame(records)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'n_folds': len(self.fold_results),
            'aggregate_metrics': self.aggregate_metrics,
            'timestamp': self.timestamp,
            'config': self.config,
            'fold_results': [fr.to_dict() for fr in self.fold_results],
        }
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save results summary to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Results saved to {path}")
    
    def get_ic_series(self) -> pd.Series:
        """Get time series of test ICs across folds."""
        data = {
            fr.test_start: fr.metrics.get('test_rank_ic', fr.metrics.get('test_ic', np.nan))
            for fr in self.fold_results
        }
        return pd.Series(data).sort_index()
    
    def get_cumulative_returns(self) -> Optional[pd.Series]:
        """
        Calculate cumulative returns from predictions.
        
        Simple long-short strategy: long top quintile, short bottom quintile.
        
        Returns:
            Series of cumulative returns, or None if no predictions
        """
        if self.predictions_df is None or len(self.predictions_df) == 0:
            return None
        
        df = self.predictions_df.copy()
        
        # Calculate quintile ranks per date
        df['quintile'] = df.groupby('date')['prediction'].transform(
            lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        )
        
        # Long top quintile (5), short bottom quintile (1)
        df['position'] = 0.0
        df.loc[df['quintile'] == 5, 'position'] = 1.0
        df.loc[df['quintile'] == 1, 'position'] = -1.0
        
        # Calculate returns
        df['strategy_return'] = df['position'] * df['forward_return']
        
        # Aggregate by date
        daily_returns = df.groupby('date')['strategy_return'].mean()
        
        # Cumulative returns
        cumulative = (1 + daily_returns).cumprod()
        
        return cumulative


# ============================================
# WALK-FORWARD TRAINER
# ============================================

class WalkForwardTrainer:
    """
    Walk-forward validation trainer for time series alpha models.
    
    Key Principles:
    ---------------
    1. Train only on past data (no lookahead bias)
    2. Embargo period between train and test sets
    3. Fresh model for each fold (no scaler/state leakage)
    4. Cross-sectional IC for proper alpha evaluation
    
    Supports two window strategies:
    - Expanding: Training window grows over time (recommended)
    - Rolling: Fixed-size training window
    
    Example:
        >>> trainer = WalkForwardTrainer(feature_names)
        >>> results = trainer.train_and_validate(features_df)
        >>> results.print_summary()
        
        >>> # Get best model
        >>> best_model = trainer.get_best_model()
        
        >>> # Make predictions on new data
        >>> predictions = trainer.predict_out_of_sample(new_data)
    
    Attributes:
        feature_names: List of feature column names
        target_column: Name of target column
        date_column: Name of date column
        ticker_column: Name of ticker column
        fold_results: List of FoldResult objects after training
        models: Dictionary of trained models keyed by fold
    """
    
    def __init__(
        self, 
        feature_names: List[str],
        target_column: str = 'forward_return',
        date_column: str = 'date',
        ticker_column: str = 'ticker',
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize trainer.
        
        Args:
            feature_names: List of feature column names
            target_column: Name of target column
            date_column: Name of date column
            ticker_column: Name of ticker column
            model_config: Custom ModelConfig (optional)
        """
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        
        self.feature_names = list(feature_names)
        self.target_column = target_column
        self.date_column = date_column
        self.ticker_column = ticker_column
        self.model_config = model_config or ModelConfig.from_settings()
        
        # Get validation config from settings
        self.use_expanding_window = getattr(settings.validation, 'use_expanding_window', True)
        self.train_months = getattr(settings.validation, 'train_months', 36)
        self.min_train_months = getattr(settings.validation, 'min_train_months', 24)
        self.test_months = getattr(settings.validation, 'test_months', 3)
        self.step_months = getattr(settings.validation, 'step_months', 3)
        self.embargo_days = getattr(settings.validation, 'embargo_days', 21)
        self.min_train_samples = getattr(settings.validation, 'min_train_samples', 500)
        self.min_test_samples = getattr(settings.validation, 'min_test_samples', 100)
        
        # Results storage
        self.fold_results: List[FoldResult] = []
        self.models: Dict[int, LightGBMModel] = {}
        self.feature_importance_history: List[pd.DataFrame] = []
        
        logger.info(f"WalkForwardTrainer initialized with {len(feature_names)} features")
        self._print_config()
    
    def _print_config(self) -> None:
        """Print trainer configuration."""
        window_type = "Expanding" if self.use_expanding_window else "Rolling"
        
        print(f"\n🎯 Walk-Forward Trainer Initialized")
        print(f"   📊 Features: {len(self.feature_names)}")
        print(f"   📈 Window: {window_type}")
        
        if self.use_expanding_window:
            print(f"   🔧 Min Train: {self.min_train_months} months")
        else:
            print(f"   🔧 Train Window: {self.train_months} months")
        
        print(f"   📈 Test Window: {self.test_months} months")
        print(f"   ⏭️  Step Size: {self.step_months} months")
        print(f"   🚫 Embargo: {self.embargo_days} days")
    
    def create_time_splits(
        self, 
        df: pd.DataFrame
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Create time-based train/test splits.
        
        Uses relativedelta for accurate month calculations.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        dates = pd.to_datetime(df[self.date_column])
        min_date = dates.min()
        max_date = dates.max()
        
        splits = []
        
        if self.use_expanding_window:
            # Expanding window: train_start is always min_date
            train_start = min_date
            
            # First test starts after minimum training period
            first_test_start = min_date + relativedelta(months=self.min_train_months)
            current_test_start = first_test_start
            
            while current_test_start < max_date:
                # Train end is embargo days before test start
                train_end = current_test_start - pd.Timedelta(days=self.embargo_days)
                
                # Test end
                test_end = current_test_start + relativedelta(months=self.test_months)
                test_end = min(pd.Timestamp(test_end), pd.Timestamp(max_date))
                
                # Validate split
                if train_end > train_start:
                    train_mask = (dates >= train_start) & (dates <= train_end)
                    test_mask = (dates >= current_test_start) & (dates <= test_end)
                    
                    train_samples = train_mask.sum()
                    test_samples = test_mask.sum()
                    
                    if (train_samples >= self.min_train_samples and 
                        test_samples >= self.min_test_samples):
                        splits.append((
                            pd.Timestamp(train_start),
                            pd.Timestamp(train_end),
                            pd.Timestamp(current_test_start),
                            pd.Timestamp(test_end)
                        ))
                
                # Move to next test period
                current_test_start = current_test_start + relativedelta(months=self.step_months)
        
        else:
            # Rolling window: fixed training window size
            current_start = min_date
            
            while True:
                train_start = current_start
                train_end = train_start + relativedelta(months=self.train_months)
                
                # Test starts after embargo
                test_start = train_end + pd.Timedelta(days=self.embargo_days)
                test_end = test_start + relativedelta(months=self.test_months)
                
                # Check if we have enough data
                if test_end > max_date:
                    break
                
                # Validate split
                train_mask = (dates >= train_start) & (dates <= train_end)
                test_mask = (dates >= test_start) & (dates <= test_end)
                
                train_samples = train_mask.sum()
                test_samples = test_mask.sum()
                
                if (train_samples >= self.min_train_samples and 
                    test_samples >= self.min_test_samples):
                    splits.append((
                        pd.Timestamp(train_start),
                        pd.Timestamp(train_end),
                        pd.Timestamp(test_start),
                        pd.Timestamp(test_end)
                    ))
                
                # Move forward
                current_start = current_start + relativedelta(months=self.step_months)
        
        logger.info(f"Created {len(splits)} time splits")
        return splits
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = {self.date_column, self.ticker_column, self.target_column}
        required_cols.update(self.feature_names)
        
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(df) == 0:
            raise ValueError("Empty DataFrame provided")
        
        # Check target NaN rate
        nan_rate = df[self.target_column].isna().mean()
        if nan_rate > 0.5:
            logger.warning(f"High NaN rate in target: {nan_rate:.1%}")
    
    def _get_fold_data(
        self,
        df: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data for a single fold."""
        dates = pd.to_datetime(df[self.date_column])
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        train_data = df[train_mask].copy()
        test_data = df[test_mask].copy()
        
        # Drop rows with NaN target
        train_data = train_data.dropna(subset=[self.target_column])
        test_data = test_data.dropna(subset=[self.target_column])
        
        return train_data, test_data
    
    def _train_fold(
        self,
        fold_idx: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        save_predictions: bool = True
    ) -> Tuple[FoldResult, LightGBMModel]:
        """Train and evaluate a single fold."""
        
        # Prepare data
        X_train = train_data[self.feature_names]
        y_train = train_data[self.target_column]
        
        X_test = test_data[self.feature_names]
        y_test = test_data[self.target_column]
        dates_test = test_data[self.date_column]
        
        # Create FRESH model for this fold (prevents any data leakage)
        model = LightGBMModel(self.feature_names, config=self.model_config)
        
        # Split training data for early stopping validation
        # Use last 20% (temporally closest to test set)
        val_size = int(len(X_train) * 0.2)
        
        if val_size >= 50:
            X_train_fit = X_train.iloc[:-val_size]
            y_train_fit = y_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
        else:
            X_train_fit = X_train
            y_train_fit = y_train
            X_val = None
            y_val = None
        
        # Train
        model.fit(X_train_fit, y_train_fit, X_val, y_val)
        
        # Evaluate with dates for cross-sectional IC
        test_metrics = model.evaluate(
            X_test, y_test,
            dates=dates_test,
            prefix='test_'
        )
        
        train_metrics = model.evaluate(
            X_train, y_train,
            dates=train_data[self.date_column],
            prefix='train_'
        )
        
        all_metrics = {**train_metrics, **test_metrics}
        
        # Get predictions
        predictions_df = None
        if save_predictions:
            predictions = model.predict(X_test)
            predictions_df = pd.DataFrame({
                self.date_column: test_data[self.date_column].values,
                self.ticker_column: test_data[self.ticker_column].values,
                'prediction': predictions,
                self.target_column: y_test.values,
                'fold': fold_idx
            })
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Create result
        result = FoldResult(
            fold=fold_idx,
            train_start=train_start.to_pydatetime(),
            train_end=train_end.to_pydatetime(),
            test_start=test_start.to_pydatetime(),
            test_end=test_end.to_pydatetime(),
            train_samples=len(train_data),
            test_samples=len(test_data),
            metrics=all_metrics,
            predictions=predictions_df,
            feature_importance=feature_importance
        )
        
        return result, model
    
    def train_and_validate(
        self, 
        df: pd.DataFrame, 
        save_models: bool = True,
        save_predictions: bool = True,
        verbose: bool = True
    ) -> WalkForwardResults:
        """
        Run walk-forward validation.
        
        Args:
            df: Features DataFrame with date, ticker, features, and target
            save_models: Whether to keep trained models in memory
            save_predictions: Whether to store predictions per fold
            verbose: Whether to print progress
            
        Returns:
            WalkForwardResults object
        """
        print("\n" + "=" * 70)
        print("🎯 WALK-FORWARD VALIDATION")
        print("=" * 70)
        
        # Validate input
        self._validate_input(df)
        
        # Create time splits
        splits = self.create_time_splits(df)
        
        if len(splits) == 0:
            raise ValueError(
                "No valid time splits found! "
                "Check data date range and configuration parameters."
            )
        
        print(f"\n📅 Created {len(splits)} time splits")
        
        # Reset state
        self.fold_results = []
        self.models = {}
        self.feature_importance_history = []
        all_predictions = []
        
        # Progress tracking
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(splits), total=len(splits), desc="Training folds")
        except ImportError:
            iterator = enumerate(splits)
            if verbose:
                print("Training folds...")
        
        # Train each fold
        for fold_idx, (train_start, train_end, test_start, test_end) in iterator:
            
            if verbose and (fold_idx + 1) % 3 == 0:
                logger.debug(
                    f"Fold {fold_idx}: train {train_start.date()} → {train_end.date()}, "
                    f"test {test_start.date()} → {test_end.date()}"
                )
            
            # Get fold data
            train_data, test_data = self._get_fold_data(
                df, train_start, train_end, test_start, test_end
            )
            
            # Skip if insufficient data
            if len(train_data) < self.min_train_samples:
                logger.warning(f"Fold {fold_idx}: insufficient training samples ({len(train_data)})")
                continue
            
            if len(test_data) < self.min_test_samples:
                logger.warning(f"Fold {fold_idx}: insufficient test samples ({len(test_data)})")
                continue
            
            # Train fold
            try:
                fold_result, model = self._train_fold(
                    fold_idx=fold_idx,
                    train_data=train_data,
                    test_data=test_data,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    save_predictions=save_predictions
                )
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                continue
            
            self.fold_results.append(fold_result)
            
            # Store model
            if save_models:
                self.models[fold_idx] = model
            
            # Store feature importance
            if fold_result.feature_importance is not None:
                fi = fold_result.feature_importance.copy()
                fi['fold'] = fold_idx
                fi['test_start'] = test_start
                self.feature_importance_history.append(fi)
            
            # Collect predictions
            if save_predictions and fold_result.predictions is not None:
                all_predictions.append(fold_result.predictions)
        
        # Combine predictions
        predictions_df = None
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(predictions_df)
        
        # Feature importance summary
        fi_summary = self.get_feature_importance_summary()
        
        # Config dictionary
        config_dict = {
            'use_expanding_window': self.use_expanding_window,
            'train_months': self.train_months,
            'min_train_months': self.min_train_months,
            'test_months': self.test_months,
            'step_months': self.step_months,
            'embargo_days': self.embargo_days,
        }
        
        # Create results object
        results = WalkForwardResults(
            fold_results=self.fold_results,
            aggregate_metrics=aggregate_metrics,
            predictions_df=predictions_df,
            feature_importance_summary=fi_summary,
            config=config_dict
        )
        
        # Print summary
        if verbose:
            results.print_summary()
        
        return results
    
    def _calculate_aggregate_metrics(
        self, 
        predictions_df: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across all folds."""
        if not self.fold_results:
            return {}
        
        aggregate = {}
        
        # Collect all metric names
        metric_names = set()
        for fr in self.fold_results:
            metric_names.update(fr.metrics.keys())
        
        # Calculate mean and std for each metric
        for metric in metric_names:
            values = [
                fr.metrics.get(metric, np.nan) 
                for fr in self.fold_results
            ]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregate[f'{metric}_mean'] = float(np.mean(values))
                aggregate[f'{metric}_std'] = float(np.std(values))
        
        # Calculate overall cross-sectional metrics from combined predictions
        if predictions_df is not None and len(predictions_df) > 0:
            try:
                cs_ic = calculate_cross_sectional_ic(
                    predictions_df,
                    pred_col='prediction',
                    actual_col=self.target_column,
                    date_col=self.date_column
                )
                
                aggregate['overall_mean_ic'] = float(cs_ic['ic'].mean())
                aggregate['overall_mean_rank_ic'] = float(cs_ic['rank_ic'].mean())
                aggregate['overall_ir'] = calculate_information_ratio(cs_ic['rank_ic'])
                aggregate['n_test_dates'] = len(cs_ic)
                
            except Exception as e:
                logger.warning(f"Error calculating cross-sectional metrics: {e}")
        
        # Total sample counts
        aggregate['total_train_samples'] = sum(fr.train_samples for fr in self.fold_results)
        aggregate['total_test_samples'] = sum(fr.test_samples for fr in self.fold_results)
        aggregate['n_folds'] = len(self.fold_results)
        
        return aggregate
    
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get aggregated feature importance across all folds.
        
        Returns:
            DataFrame with columns: feature, importance_mean, importance_std, importance_pct_mean
        """
        if not self.feature_importance_history:
            return pd.DataFrame()
        
        # Combine all importance data
        all_importance = pd.concat(self.feature_importance_history, ignore_index=True)
        
        # Calculate summary
        summary = all_importance.groupby('feature').agg({
            'importance': ['mean', 'std'],
            'importance_pct': 'mean'
        }).round(4)
        
        # Flatten columns
        summary.columns = ['importance_mean', 'importance_std', 'importance_pct_mean']
        summary = summary.reset_index()
        
        # Sort by mean importance
        summary = summary.sort_values('importance_mean', ascending=False)
        
        return summary.reset_index(drop=True)
    
    def get_best_model(self) -> Optional[LightGBMModel]:
        """
        Get the model from the best performing fold.
        
        Best is determined by test Rank IC.
        
        Returns:
            Best performing LightGBMModel, or None if no models available
        """
        if not self.fold_results or not self.models:
            return None
        
        # Find fold with best test Rank IC
        best_fold = max(
            self.fold_results,
            key=lambda x: x.metrics.get('test_rank_ic', x.metrics.get('test_ic', 0))
        )
        
        return self.models.get(best_fold.fold)
    
    def get_latest_model(self) -> Optional[LightGBMModel]:
        """
        Get the most recently trained model.
        
        Returns:
            Most recent LightGBMModel, or None if no models available
        """
        if not self.fold_results or not self.models:
            return None
        
        latest_fold = max(self.fold_results, key=lambda x: x.fold)
        return self.models.get(latest_fold.fold)
    
    def get_model(self, fold: int) -> Optional[LightGBMModel]:
        """
        Get model for a specific fold.
        
        Args:
            fold: Fold index
            
        Returns:
            LightGBMModel for specified fold, or None if not found
        """
        return self.models.get(fold)
    
    def predict_out_of_sample(
        self, 
        df: pd.DataFrame, 
        model: Optional[LightGBMModel] = None,
        use_latest: bool = False
    ) -> pd.DataFrame:
        """
        Generate out-of-sample predictions.
        
        Args:
            df: DataFrame with feature columns
            model: Specific model to use (optional)
            use_latest: If True, use latest model; otherwise use best model
            
        Returns:
            DataFrame with predictions added
            
        Raises:
            ValueError: If no trained model available
        """
        if model is None:
            if use_latest:
                model = self.get_latest_model()
            else:
                model = self.get_best_model()
        
        if model is None:
            raise ValueError("No trained model available!")
        
        # Make predictions
        predictions = model.predict(df[self.feature_names])
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['prediction'] = predictions
        
        return result_df
    
    def get_cross_sectional_metrics(
        self, 
        df: pd.DataFrame,
        pred_col: str = 'prediction'
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional metrics for predictions.
        
        Args:
            df: DataFrame with predictions and target
            pred_col: Name of prediction column
            
        Returns:
            DataFrame with IC metrics per date
        """
        if pred_col not in df.columns:
            raise ValueError(f"Column '{pred_col}' not found")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Column '{self.target_column}' not found")
        
        return calculate_cross_sectional_ic(
            df,
            pred_col=pred_col,
            actual_col=self.target_column,
            date_col=self.date_column
        )
    
    def save_models(
        self, 
        directory: Union[str, Path],
        prefix: str = "model"
    ) -> List[str]:
        """
        Save all trained models to directory.
        
        Args:
            directory: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for fold_idx, model in self.models.items():
            path = directory / f"{prefix}_fold_{fold_idx}.pkl"
            model.save(path)
            saved_paths.append(str(path))
            
            # Update fold result with path
            for fr in self.fold_results:
                if fr.fold == fold_idx:
                    fr.model_path = str(path)
        
        logger.info(f"Saved {len(saved_paths)} models to {directory}")
        return saved_paths
    
    def load_models(
        self, 
        directory: Union[str, Path],
        prefix: str = "model"
    ) -> int:
        """
        Load models from directory.
        
        Args:
            directory: Directory containing model files
            prefix: Filename prefix to match
            
        Returns:
            Number of models loaded
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        self.models = {}
        
        for path in directory.glob(f"{prefix}_fold_*.pkl"):
            try:
                # Extract fold index from filename
                fold_idx = int(path.stem.split('_')[-1])
                
                model = LightGBMModel.load(path)
                self.models[fold_idx] = model
                
                logger.debug(f"Loaded model for fold {fold_idx}")
                
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models from {directory}")
        return len(self.models)
    
    def save_results(
        self, 
        path: Union[str, Path],
        include_predictions: bool = True,
        include_feature_importance: bool = True
    ) -> None:
        """
        Save validation results to files.
        
        Creates:
        - {path}.json: Summary metrics
        - {path}_predictions.pkl: Predictions DataFrame
        - {path}_feature_importance.csv: Feature importance
        - {path}_fold_results.csv: Per-fold metrics
        
        Args:
            path: Base path (without extension)
            include_predictions: Whether to save predictions
            include_feature_importance: Whether to save feature importance
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save fold results as CSV
        results_df = pd.DataFrame([
            {
                'fold': fr.fold,
                'train_start': fr.train_start,
                'train_end': fr.train_end,
                'test_start': fr.test_start,
                'test_end': fr.test_end,
                'train_samples': fr.train_samples,
                'test_samples': fr.test_samples,
                **fr.metrics
            }
            for fr in self.fold_results
        ])
        results_df.to_csv(path.with_name(f"{path.stem}_fold_results.csv"), index=False)
        
        # Save aggregate metrics as JSON
        aggregate = self._calculate_aggregate_metrics(None)
        config_dict = {
            'use_expanding_window': self.use_expanding_window,
            'train_months': self.train_months,
            'test_months': self.test_months,
            'embargo_days': self.embargo_days,
            'n_features': len(self.feature_names),
        }
        
        summary = {
            'config': config_dict,
            'aggregate_metrics': aggregate,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save predictions
        if include_predictions:
            all_preds = [
                fr.predictions for fr in self.fold_results 
                if fr.predictions is not None
            ]
            if all_preds:
                predictions_df = pd.concat(all_preds, ignore_index=True)
                predictions_df.to_pickle(path.with_name(f"{path.stem}_predictions.pkl"))
        
        # Save feature importance
        if include_feature_importance:
            fi_summary = self.get_feature_importance_summary()
            if len(fi_summary) > 0:
                fi_summary.to_csv(
                    path.with_name(f"{path.stem}_feature_importance.csv"), 
                    index=False
                )
        
        logger.info(f"Results saved to {path}")
        print(f"💾 Results saved: {path}")
    
    def __repr__(self) -> str:
        """String representation."""
        n_folds = len(self.fold_results)
        status = f"{n_folds} folds trained" if n_folds > 0 else "not trained"
        return f"WalkForwardTrainer(n_features={len(self.feature_names)}, {status})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        if self.fold_results:
            best_ic = max(
                fr.metrics.get('test_rank_ic', 0) 
                for fr in self.fold_results
            )
            return (f"WalkForwardTrainer with {len(self.fold_results)} folds, "
                   f"best Rank IC: {best_ic:.4f}")
        return f"WalkForwardTrainer with {len(self.feature_names)} features (not trained)"


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def run_walk_forward_validation(
    features_df: pd.DataFrame, 
    feature_names: List[str],
    target_column: str = 'forward_return',
    date_column: str = 'date',
    ticker_column: str = 'ticker',
    verbose: bool = True,
    **kwargs
) -> Tuple[WalkForwardResults, WalkForwardTrainer]:
    """
    Convenience function to run walk-forward validation.
    
    Args:
        features_df: DataFrame with features and targets
        feature_names: List of feature column names
        target_column: Name of target column
        date_column: Name of date column
        ticker_column: Name of ticker column
        verbose: Whether to print progress
        **kwargs: Additional arguments for WalkForwardTrainer
        
    Returns:
        Tuple of (WalkForwardResults, WalkForwardTrainer)
        
    Example:
        >>> results, trainer = run_walk_forward_validation(
        ...     features_df,
        ...     feature_names=['mom_21', 'rsi_14'],
        ...     verbose=True
        ... )
        >>> results.print_summary()
        >>> best_model = trainer.get_best_model()
    """
    trainer = WalkForwardTrainer(
        feature_names=feature_names,
        target_column=target_column,
        date_column=date_column,
        ticker_column=ticker_column,
        **kwargs
    )
    
    results = trainer.train_and_validate(
        features_df,
        verbose=verbose
    )
    
    return results, trainer


def create_trainer(
    feature_names: List[str],
    **kwargs
) -> WalkForwardTrainer:
    """
    Factory function to create WalkForwardTrainer.
    
    Args:
        feature_names: List of feature names
        **kwargs: Additional arguments for WalkForwardTrainer
        
    Returns:
        Configured WalkForwardTrainer
    """
    return WalkForwardTrainer(feature_names, **kwargs)


# ============================================
# MODULE TEST
# ============================================

def test_walk_forward_trainer():
    """Test walk-forward trainer with synthetic data."""
    print("\n" + "=" * 70)
    print("🧪 TESTING WALK-FORWARD TRAINER")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create synthetic panel data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    
    feature_names = ['mom_21', 'mom_63', 'rsi_14', 'volatility_21', 'volume_zscore_21']
    
    print(f"\n📊 Creating synthetic data...")
    
    records = []
    for date in dates:
        for ticker in tickers:
            record = {'date': date, 'ticker': ticker}
            
            # Simulated features (cross-sectionally ranked: 0-1)
            for feat in feature_names:
                record[feat] = np.random.uniform(0, 1)
            
            # Signal with some noise
            signal = (
                record['mom_21'] * 0.02 +
                record['mom_63'] * 0.01 -
                record['rsi_14'] * 0.01 +
                np.random.randn() * 0.03
            )
            record['forward_return'] = signal
            
            records.append(record)
    
    df = pd.DataFrame(records)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Tickers: {len(tickers)}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Test 1: Trainer creation
    print("\n1️⃣ Testing trainer creation...")
    trainer = create_trainer(feature_names)
    print(f"   ✅ Created: {trainer}")
    
    # Test 2: Run validation
    print("\n2️⃣ Running walk-forward validation...")
    results = trainer.train_and_validate(df, save_models=True, verbose=True)
    
    # Test 3: Access results
    print("\n3️⃣ Testing results access...")
    print(f"   ✅ Folds: {len(results.fold_results)}")
    print(f"   ✅ Predictions shape: {results.predictions_df.shape if results.predictions_df is not None else 'N/A'}")
    print(f"   ✅ Overall IR: {results.aggregate_metrics.get('overall_ir', 0):.4f}")
    
    # Test 4: Feature importance
    print("\n4️⃣ Testing feature importance...")
    fi_summary = trainer.get_feature_importance_summary()
    if len(fi_summary) > 0:
        print(f"   ✅ Top features:")
        print(fi_summary.head())
    
    # Test 5: Model access
    print("\n5️⃣ Testing model access...")
    best_model = trainer.get_best_model()
    latest_model = trainer.get_latest_model()
    print(f"   ✅ Best model: {best_model}")
    print(f"   ✅ Latest model: {latest_model}")
    
    # Test 6: Out-of-sample prediction
    print("\n6️⃣ Testing out-of-sample prediction...")
    test_df = df[df['date'] == df['date'].max()].copy()
    predictions_df = trainer.predict_out_of_sample(test_df)
    print(f"   ✅ Predictions: {predictions_df['prediction'].describe()}")
    
    # Test 7: Save/Load
    print("\n7️⃣ Testing save/load...")
    save_dir = Path("test_trainer_output")
    
    # Save results
    trainer.save_results(save_dir / "results")
    print(f"   ✅ Results saved")
    
    # Save models
    trainer.save_models(save_dir / "models")
    print(f"   ✅ Models saved")
    
    # Load models
    new_trainer = create_trainer(feature_names)
    n_loaded = new_trainer.load_models(save_dir / "models")
    print(f"   ✅ Loaded {n_loaded} models")
    
    # Cleanup
    import shutil
    shutil.rmtree(save_dir, ignore_errors=True)
    
    # Test 8: Results DataFrame
    print("\n8️⃣ Testing results DataFrame...")
    results_df = results.to_dataframe()
    print(f"   ✅ Shape: {results_df.shape}")
    print(results_df[['fold', 'test_start', 'test_samples', 'test_rank_ic']].head())
    
    # Test 9: IC series
    print("\n9️⃣ Testing IC series...")
    ic_series = results.get_ic_series()
    print(f"   ✅ IC series length: {len(ic_series)}")
    print(f"   ✅ Mean IC: {ic_series.mean():.4f}")
    
    print("\n✅ All walk-forward trainer tests passed!")
    
    return results, trainer


if __name__ == "__main__":
    test_walk_forward_trainer()