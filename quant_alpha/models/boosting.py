"""
Gradient Boosting Models
========================
LightGBM model wrapper for alpha prediction.

Features:
- Proper handling of pre-normalized features (NO double scaling)
- Comprehensive evaluation metrics with cross-sectional IC
- NaN handling strategies
- Model persistence
- Feature importance analysis

Author: [Your Name]
Last Updated: 2024

IMPORTANT NOTES:
================
1. Features should be cross-sectionally normalized BEFORE passing to model
2. This model does NOT apply StandardScaler (features are already 0-1 ranked)
3. Each walk-forward fold should train a FRESH model instance
4. Cross-sectional IC is the proper way to evaluate alpha models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import warnings
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib

# Proper imports with fallback
try:
    from config.settings import settings
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import settings


# Setup logging
logger = logging.getLogger(__name__)

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


# ============================================
# MODEL CONFIGURATION
# ============================================

@dataclass
class ModelConfig:
    """
    Configuration for LightGBM model.
    
    Attributes:
        params: LightGBM hyperparameters
        early_stopping_rounds: Rounds for early stopping (None to disable)
        use_scaling: Whether to scale features (False if pre-normalized)
        handle_nan: How to handle NaN values ('drop', 'fill_zero', 'fill_median')
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
        random_state: Random seed for reproducibility
    """
    params: Dict = field(default_factory=dict)
    early_stopping_rounds: Optional[int] = 50
    use_scaling: bool = False  # Features are already cross-sectionally normalized
    handle_nan: str = 'drop'   # 'drop', 'fill_zero', 'fill_median'
    verbose: int = 0
    random_state: int = 42
    
    def __post_init__(self):
        """Set default LightGBM parameters if not provided."""
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'random_state': self.random_state,
            'verbose': -1,
            'n_jobs': -1,
            'force_col_wise': True,  # Suppress warning
        }
        
        # Merge with provided params (provided params take precedence)
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    @classmethod
    def from_settings(cls) -> 'ModelConfig':
        """Create config from global settings."""
        params = settings.model.lgb_params.copy() if hasattr(settings.model, 'lgb_params') else {}
        return cls(
            params=params,
            random_state=settings.model.random_seed if hasattr(settings.model, 'random_seed') else 42
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'params': self.params,
            'early_stopping_rounds': self.early_stopping_rounds,
            'use_scaling': self.use_scaling,
            'handle_nan': self.handle_nan,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }


# ============================================
# EVALUATION METRICS
# ============================================

def calculate_ic(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    Calculate Information Coefficient (Pearson correlation).
    
    IC measures the linear relationship between predictions and actual returns.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        IC value in range [-1, 1], or 0 if calculation fails
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    
    if mask.sum() < 3:
        return 0.0
    
    try:
        ic = np.corrcoef(predictions[mask], actuals[mask])[0, 1]
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        return 0.0


def calculate_rank_ic(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    Calculate Rank IC (Spearman correlation).
    
    Rank IC is more robust to outliers and measures monotonic relationships.
    This is the preferred metric for alpha model evaluation.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Rank IC value in range [-1, 1], or 0 if calculation fails
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    
    if mask.sum() < 3:
        return 0.0
    
    try:
        rank_ic, _ = stats.spearmanr(predictions[mask], actuals[mask])
        return float(rank_ic) if not np.isnan(rank_ic) else 0.0
    except Exception:
        return 0.0


def calculate_hit_rate(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    Calculate directional accuracy (hit rate).
    
    Measures the percentage of times the prediction got the direction right.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Hit rate in range [0, 1], where 0.5 is random
    """
    # Remove NaN values
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    
    if mask.sum() < 1:
        return 0.5
    
    correct = np.sign(predictions[mask]) == np.sign(actuals[mask])
    return float(np.mean(correct))


def calculate_cross_sectional_ic(
    df: pd.DataFrame,
    pred_col: str = 'prediction',
    actual_col: str = 'forward_return',
    date_col: str = 'date'
) -> pd.DataFrame:
    """
    Calculate IC for each cross-section (date).
    
    This is the CORRECT way to calculate IC for alpha models:
    - IC should be calculated within each date (cross-sectionally)
    - Then aggregated across dates
    
    This avoids inflating IC by mixing time-series and cross-sectional correlations.
    
    Args:
        df: DataFrame with predictions, actuals, and dates
        pred_col: Column name for predictions
        actual_col: Column name for actual returns
        date_col: Column name for dates
        
    Returns:
        DataFrame with IC metrics per date:
        - ic: Pearson correlation
        - rank_ic: Spearman correlation
        - n_stocks: Number of stocks in cross-section
        
    Example:
        >>> cs_ic = calculate_cross_sectional_ic(predictions_df)
        >>> print(f"Mean Rank IC: {cs_ic['rank_ic'].mean():.4f}")
    """
    def _calc_ic_for_date(group: pd.DataFrame) -> pd.Series:
        """Calculate IC for a single date."""
        preds = group[pred_col].values
        actuals = group[actual_col].values
        
        # Remove NaN
        mask = ~(np.isnan(preds) | np.isnan(actuals))
        n_valid = mask.sum()
        
        if n_valid < 3:
            return pd.Series({
                'ic': np.nan,
                'rank_ic': np.nan,
                'n_stocks': n_valid
            })
        
        ic = calculate_ic(preds[mask], actuals[mask])
        rank_ic = calculate_rank_ic(preds[mask], actuals[mask])
        
        return pd.Series({
            'ic': ic,
            'rank_ic': rank_ic,
            'n_stocks': n_valid
        })
    
    # Calculate IC for each date
    result = df.groupby(date_col).apply(_calc_ic_for_date)
    
    return result.reset_index()


def calculate_information_ratio(ic_series: pd.Series) -> float:
    """
    Calculate Information Ratio from IC time series.
    
    IR = mean(IC) / std(IC)
    
    A higher IR indicates more consistent alpha generation.
    Generally, IR > 0.5 is considered good, IR > 1.0 is excellent.
    
    Args:
        ic_series: Series of IC values (one per date)
        
    Returns:
        Information Ratio, or 0 if calculation fails
    """
    # Remove NaN values
    ic_clean = ic_series.dropna()
    
    if len(ic_clean) < 5:
        return 0.0
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    
    if std_ic < 1e-8:
        return 0.0
    
    return float(mean_ic / std_ic)


def calculate_all_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: Optional[np.ndarray] = None,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        dates: Optional date array for cross-sectional IC calculation
        prefix: Prefix for metric names (e.g., 'test_' or 'train_')
        
    Returns:
        Dictionary of metrics:
        - ic: Pearson correlation
        - rank_ic: Spearman correlation
        - hit_rate: Directional accuracy
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - r2: R-squared
        - ir: Information ratio (if dates provided)
        - n_samples: Number of valid samples
    """
    metrics = {}
    
    # Handle NaN
    mask = ~(np.isnan(predictions) | np.isnan(actuals))
    n_valid = mask.sum()
    n_dropped = len(predictions) - n_valid
    
    if n_dropped > 0:
        logger.debug(f"Dropped {n_dropped} rows with NaN for evaluation")
    
    metrics[f'{prefix}n_samples'] = int(n_valid)
    
    if n_valid < 3:
        logger.warning(f"Insufficient valid samples for evaluation: {n_valid}")
        return {
            f'{prefix}ic': 0.0,
            f'{prefix}rank_ic': 0.0,
            f'{prefix}hit_rate': 0.5,
            f'{prefix}rmse': np.nan,
            f'{prefix}mae': np.nan,
            f'{prefix}r2': 0.0,
            f'{prefix}ir': 0.0,
            f'{prefix}n_samples': n_valid,
        }
    
    pred_clean = predictions[mask]
    actual_clean = actuals[mask]
    
    # Core alpha metrics
    metrics[f'{prefix}ic'] = calculate_ic(pred_clean, actual_clean)
    metrics[f'{prefix}rank_ic'] = calculate_rank_ic(pred_clean, actual_clean)
    metrics[f'{prefix}hit_rate'] = calculate_hit_rate(pred_clean, actual_clean)
    
    # Cross-sectional metrics (if dates provided)
    if dates is not None:
        dates_clean = np.asarray(dates)[mask]
        
        # Create temp DataFrame for cross-sectional calculation
        temp_df = pd.DataFrame({
            'date': dates_clean,
            'prediction': pred_clean,
            'forward_return': actual_clean
        })
        
        try:
            cs_ic = calculate_cross_sectional_ic(temp_df)
            metrics[f'{prefix}mean_cs_ic'] = float(cs_ic['ic'].mean())
            metrics[f'{prefix}mean_cs_rank_ic'] = float(cs_ic['rank_ic'].mean())
            metrics[f'{prefix}ir'] = calculate_information_ratio(cs_ic['rank_ic'])
        except Exception as e:
            logger.warning(f"Error calculating cross-sectional IC: {e}")
            metrics[f'{prefix}ir'] = 0.0
    else:
        metrics[f'{prefix}ir'] = 0.0
    
    # Regression metrics
    try:
        metrics[f'{prefix}rmse'] = float(np.sqrt(mean_squared_error(actual_clean, pred_clean)))
        metrics[f'{prefix}mae'] = float(mean_absolute_error(actual_clean, pred_clean))
        metrics[f'{prefix}r2'] = float(r2_score(actual_clean, pred_clean))
    except Exception:
        metrics[f'{prefix}rmse'] = np.nan
        metrics[f'{prefix}mae'] = np.nan
        metrics[f'{prefix}r2'] = 0.0
    
    return metrics


# ============================================
# LIGHTGBM MODEL CLASS
# ============================================

class LightGBMModel:
    """
    LightGBM regressor wrapper for alpha prediction.
    
    Key Design Decisions:
    ---------------------
    1. NO StandardScaler: Features are assumed to be pre-normalized
       (cross-sectionally ranked to 0-1 range)
    
    2. Fresh Model Per Fold: In walk-forward validation, create a new
       LightGBMModel instance for each fold to avoid data leakage
    
    3. Cross-sectional Evaluation: Use calculate_cross_sectional_ic()
       for proper alpha model evaluation
    
    Example:
        >>> # Create and train model
        >>> model = LightGBMModel(feature_names)
        >>> model.fit(X_train, y_train, X_val, y_val)
        
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
        
        >>> # Evaluate with dates for proper IC calculation
        >>> metrics = model.evaluate(X_test, y_test, dates=dates_test)
        
        >>> # Get feature importance
        >>> importance = model.get_feature_importance()
    
    Attributes:
        feature_names: List of feature column names
        config: ModelConfig instance
        model: Trained LightGBM model (None before training)
        is_fitted: Whether model has been trained
        training_info: Dictionary with training metadata
    """
    
    def __init__(
        self, 
        feature_names: List[str], 
        config: Optional[ModelConfig] = None
    ):
        """
        Initialize LightGBM model.
        
        Args:
            feature_names: List of feature column names (must match training data)
            config: ModelConfig instance (uses defaults if None)
        """
        if not feature_names:
            raise ValueError("feature_names cannot be empty")
        
        self.feature_names = list(feature_names)
        self.config = config or ModelConfig.from_settings()
        
        # Model state
        self.model: Optional[lgb.LGBMRegressor] = None
        self.is_fitted: bool = False
        self.training_info: Dict[str, Any] = {}
        
        # Cached feature importance
        self._feature_importance_df: Optional[pd.DataFrame] = None
        
        # Feature statistics (for drift detection)
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"LightGBMModel initialized with {len(feature_names)} features")
    
    def _validate_features(
        self, 
        X: pd.DataFrame, 
        stage: str = 'predict'
    ) -> pd.DataFrame:
        """
        Validate and align feature columns.
        
        Args:
            X: Input features DataFrame
            stage: 'train' or 'predict'
            
        Returns:
            DataFrame with features in correct order
            
        Raises:
            ValueError: If critical features are missing during training
        """
        available = set(X.columns)
        required = set(self.feature_names)
        missing = required - available
        
        if missing:
            if stage == 'train':
                raise ValueError(f"Missing features in training data: {missing}")
            else:
                # For prediction, fill missing with neutral value (0.5 for ranked features)
                logger.warning(f"Missing features in prediction (filling with 0.5): {list(missing)[:5]}...")
                X = X.copy()
                for col in missing:
                    X[col] = 0.5  # Neutral value for 0-1 ranked features
        
        # Select features in correct order (critical for consistency)
        return X[self.feature_names].copy()
    
    def _handle_nan(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], np.ndarray]:
        """
        Handle NaN values in features and target.
        
        Args:
            X: Features DataFrame
            y: Target Series (optional)
            
        Returns:
            Tuple of (cleaned X, cleaned y, boolean mask of valid rows)
        """
        # Create mask for valid rows
        feature_nan_mask = X.isna().any(axis=1)
        
        if y is not None:
            target_nan_mask = y.isna()
            valid_mask = ~(feature_nan_mask | target_nan_mask)
        else:
            valid_mask = ~feature_nan_mask
        
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            pct_invalid = n_invalid / len(X) * 100
            
            if self.config.handle_nan == 'drop':
                logger.debug(f"Dropping {n_invalid} rows ({pct_invalid:.1f}%) with NaN")
                X_clean = X[valid_mask].copy()
                y_clean = y[valid_mask].copy() if y is not None else None
                
            elif self.config.handle_nan == 'fill_zero':
                logger.debug(f"Filling {n_invalid} NaN values with 0")
                X_clean = X.fillna(0.0)
                y_clean = y.fillna(0.0) if y is not None else None
                valid_mask = pd.Series(True, index=X.index)
                
            elif self.config.handle_nan == 'fill_median':
                logger.debug(f"Filling NaN values with column median")
                X_clean = X.fillna(X.median())
                y_clean = y.fillna(y.median()) if y is not None else None
                valid_mask = pd.Series(True, index=X.index)
                
            else:
                raise ValueError(f"Unknown handle_nan method: {self.config.handle_nan}")
        else:
            X_clean = X.copy()
            y_clean = y.copy() if y is not None else None
        
        return X_clean, y_clean, valid_mask.values
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        categorical_features: Optional[List[str]] = None
    ) -> 'LightGBMModel':
        """
        Train the model.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
            sample_weight: Sample weights (optional)
            categorical_features: List of categorical feature names (optional)
            
        Returns:
            self (for method chaining)
            
        Raises:
            ValueError: If features are missing or insufficient data
        """
        start_time = datetime.now()
        
        logger.info(f"Training on {len(X_train):,} samples...")
        
        # Validate and align features
        X_train_aligned = self._validate_features(X_train, stage='train')
        
        # Handle NaN values
        X_train_clean, y_train_clean, train_mask = self._handle_nan(X_train_aligned, y_train)
        
        if len(X_train_clean) < 100:
            raise ValueError(f"Insufficient training samples after NaN removal: {len(X_train_clean)}")
        
        # Adjust sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[train_mask]
        
        # Store feature statistics for drift detection
        self._feature_stats = {
            col: {
                'mean': float(X_train_clean[col].mean()),
                'std': float(X_train_clean[col].std()),
                'min': float(X_train_clean[col].min()),
                'max': float(X_train_clean[col].max()),
            }
            for col in self.feature_names
        }
        
        # Prepare validation data
        eval_set = None
        callbacks = []
        
        if X_val is not None and y_val is not None:
            X_val_aligned = self._validate_features(X_val, stage='predict')
            X_val_clean, y_val_clean, _ = self._handle_nan(X_val_aligned, y_val)
            
            if len(X_val_clean) >= 10:
                eval_set = [(X_val_clean.values, y_val_clean.values)]
                
                # Add early stopping callback
                if self.config.early_stopping_rounds:
                    callbacks.append(
                        lgb.early_stopping(
                            stopping_rounds=self.config.early_stopping_rounds,
                            verbose=self.config.verbose > 0
                        )
                    )
                
                logger.info(f"Using {len(X_val_clean):,} validation samples")
        
        # Suppress LightGBM output if not verbose
        if self.config.verbose <= 0:
            callbacks.append(lgb.log_evaluation(period=0))
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**self.config.params)
        
        # Prepare fit parameters
        fit_params = {
            'X': X_train_clean.values,
            'y': y_train_clean.values,
        }
        
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['eval_names'] = ['validation']
        
        if callbacks:
            fit_params['callbacks'] = callbacks
        
        if categorical_features:
            fit_params['categorical_feature'] = [
                self.feature_names.index(f) for f in categorical_features 
                if f in self.feature_names
            ]
        
        # Train
        self.model.fit(**fit_params)
        
        self.is_fitted = True
        
        # Get best iteration
        best_iteration = getattr(self.model, 'best_iteration_', None)
        if best_iteration is None:
            best_iteration = self.config.params.get('n_estimators', 100)
        
        # Store training info
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self.training_info = {
            'train_samples': len(X_train_clean),
            'val_samples': len(X_val_clean) if X_val is not None else 0,
            'n_features': len(self.feature_names),
            'best_iteration': best_iteration,
            'training_time_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Calculate and cache feature importance
        self._calculate_feature_importance()
        
        logger.info(f"Training complete in {elapsed:.1f}s (best iteration: {best_iteration})")
        
        return self
    
    def predict(
        self, 
        X: pd.DataFrame,
        validate_distribution: bool = False
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            validate_distribution: Whether to check for feature drift
            
        Returns:
            Predictions array (same length as X, NaN for invalid rows)
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted! Call fit() first.")
        
        # Validate and align features
        X_aligned = self._validate_features(X, stage='predict')
        
        # Check for feature drift
        if validate_distribution and self._feature_stats:
            self._check_feature_drift(X_aligned)
        
        # Handle NaN values
        X_clean, _, valid_mask = self._handle_nan(X_aligned)
        
        if len(X_clean) == 0:
            logger.warning("No valid samples for prediction")
            return np.full(len(X), np.nan)
        
        # Make predictions on clean data
        predictions_clean = self.model.predict(X_clean.values)
        
        # Map back to original indices (NaN for dropped rows)
        predictions = np.full(len(X), np.nan)
        predictions[valid_mask] = predictions_clean
        
        return predictions
    
    def _check_feature_drift(self, X: pd.DataFrame) -> None:
        """
        Check if feature distributions have drifted from training.
        
        Logs warnings for significant drift.
        """
        drift_features = []
        
        for col in self.feature_names[:10]:  # Check first 10 features
            if col not in self._feature_stats or col not in X.columns:
                continue
            
            train_stats = self._feature_stats[col]
            pred_mean = X[col].mean()
            
            # Check for significant mean shift
            if train_stats['std'] > 1e-8:
                z_score = abs(pred_mean - train_stats['mean']) / train_stats['std']
                if z_score > 3:
                    drift_features.append(col)
        
        if drift_features:
            logger.warning(f"Feature drift detected in: {drift_features}")
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            X: Features DataFrame
            y: True target values
            dates: Optional date Series for cross-sectional IC calculation
            prefix: Prefix for metric names (e.g., 'test_', 'train_')
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X, validate_distribution=False)
        
        dates_array = dates.values if dates is not None else None
        
        return calculate_all_metrics(
            predictions=predictions,
            actuals=y.values,
            dates=dates_array,
            prefix=prefix
        )
    
    def evaluate_cross_sectional(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate model cross-sectionally (IC per date).
        
        This is the recommended evaluation method for alpha models.
        
        Args:
            X: Features DataFrame
            y: True target values
            dates: Date Series
            
        Returns:
            DataFrame with IC metrics per date
        """
        predictions = self.predict(X, validate_distribution=False)
        
        eval_df = pd.DataFrame({
            'date': dates.values,
            'prediction': predictions,
            'forward_return': y.values
        })
        
        return calculate_cross_sectional_ic(eval_df)
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and cache feature importance."""
        if not self.is_fitted or self.model is None:
            return
        
        importance = self.model.feature_importances_
        
        self._feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Add percentage and cumulative
        total = self._feature_importance_df['importance'].sum()
        if total > 0:
            self._feature_importance_df['importance_pct'] = (
                self._feature_importance_df['importance'] / total * 100
            )
        else:
            self._feature_importance_df['importance_pct'] = 0
            
        self._feature_importance_df['cumulative_pct'] = (
            self._feature_importance_df['importance_pct'].cumsum()
        )
        
        self._feature_importance_df = self._feature_importance_df.reset_index(drop=True)
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance DataFrame.
        
        Args:
            importance_type: 'gain' (default) or 'split'
            
        Returns:
            DataFrame with columns: feature, importance, importance_pct, cumulative_pct
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        
        if importance_type == 'split' and hasattr(self.model, 'booster_'):
            try:
                importance = self.model.booster_.feature_importance(importance_type='split')
                df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                total = df['importance'].sum()
                df['importance_pct'] = df['importance'] / total * 100 if total > 0 else 0
                df['cumulative_pct'] = df['importance_pct'].cumsum()
                
                return df.reset_index(drop=True)
            except Exception:
                pass
        
        if self._feature_importance_df is None:
            self._calculate_feature_importance()
        
        return self._feature_importance_df.copy()
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """
        Get top N most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of feature names
        """
        importance_df = self.get_feature_importance()
        return importance_df.head(n)['feature'].tolist()
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file.
        
        Args:
            path: File path (will add .pkl extension if not present)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model!")
        
        save_path = Path(path)
        if save_path.suffix != '.pkl':
            save_path = save_path.with_suffix('.pkl')
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config,
            'training_info': self.training_info,
            'feature_importance_df': self._feature_importance_df,
            'feature_stats': self._feature_stats,
            'is_fitted': self.is_fitted,
            'version': '2.0',
            'save_timestamp': datetime.now().isoformat(),
        }
        
        joblib.dump(save_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LightGBMModel':
        """
        Load model from file.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded LightGBMModel instance
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        save_data = joblib.load(load_path)
        
        # Handle different versions
        version = save_data.get('version', '1.0')
        
        if version == '2.0':
            instance = cls(
                feature_names=save_data['feature_names'],
                config=save_data['config']
            )
            instance.model = save_data['model']
            instance.training_info = save_data.get('training_info', {})
            instance._feature_importance_df = save_data.get('feature_importance_df')
            instance._feature_stats = save_data.get('feature_stats', {})
            instance.is_fitted = save_data.get('is_fitted', True)
        else:
            # Legacy format compatibility
            instance = cls(
                feature_names=save_data.get('feature_names', []),
                config=ModelConfig(params=save_data.get('params', {}))
            )
            instance.model = save_data.get('model')
            instance.is_fitted = True
            instance._calculate_feature_importance()
        
        logger.info(f"Model loaded from {load_path}")
        
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        info = {
            'status': 'fitted' if self.is_fitted else 'not_fitted',
            'n_features': len(self.feature_names),
            'model_type': 'LightGBM',
            'config': {
                'n_estimators': self.config.params.get('n_estimators'),
                'max_depth': self.config.params.get('max_depth'),
                'learning_rate': self.config.params.get('learning_rate'),
                'early_stopping': self.config.early_stopping_rounds,
            },
        }
        
        if self.is_fitted:
            info['training_info'] = self.training_info
            info['top_10_features'] = self.get_top_features(10)
        
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        status = 'fitted' if self.is_fitted else 'not_fitted'
        return f"LightGBMModel(n_features={len(self.feature_names)}, {status})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        if self.is_fitted:
            return (f"LightGBMModel with {len(self.feature_names)} features, "
                   f"trained on {self.training_info.get('train_samples', '?')} samples")
        return f"LightGBMModel with {len(self.feature_names)} features (not fitted)"


# ============================================
# FACTORY FUNCTIONS
# ============================================

def create_model(
    feature_names: List[str], 
    custom_params: Optional[Dict] = None,
    **config_kwargs
) -> LightGBMModel:
    """
    Factory function to create LightGBM model.
    
    Args:
        feature_names: List of feature names
        custom_params: Custom LightGBM parameters to override defaults
        **config_kwargs: Additional ModelConfig parameters
        
    Returns:
        Configured LightGBMModel
        
    Example:
        >>> model = create_model(
        ...     feature_names=['mom_21', 'rsi_14'],
        ...     custom_params={'n_estimators': 500, 'learning_rate': 0.03}
        ... )
    """
    # Start with settings params
    params = {}
    if hasattr(settings.model, 'lgb_params'):
        params = settings.model.lgb_params.copy()
    
    # Override with custom params
    if custom_params:
        params.update(custom_params)
    
    # Get random seed
    random_state = settings.model.random_seed if hasattr(settings.model, 'random_seed') else 42
    
    # Create config
    config = ModelConfig(
        params=params,
        random_state=random_state,
        **config_kwargs
    )
    
    return LightGBMModel(feature_names, config)


def create_ensemble_model(
    feature_names: List[str],
    n_models: int = 5,
    base_seed: Optional[int] = None
) -> List[LightGBMModel]:
    """
    Create ensemble of models with different random seeds.
    
    Useful for reducing variance in predictions.
    
    Args:
        feature_names: List of feature names
        n_models: Number of models in ensemble
        base_seed: Base random seed (default: from settings)
        
    Returns:
        List of LightGBMModel instances
        
    Example:
        >>> models = create_ensemble_model(feature_names, n_models=5)
        >>> # Train each model
        >>> for model in models:
        ...     model.fit(X_train, y_train)
        >>> # Average predictions
        >>> predictions = np.mean([m.predict(X_test) for m in models], axis=0)
    """
    if base_seed is None:
        base_seed = settings.model.random_seed if hasattr(settings.model, 'random_seed') else 42
    
    models = []
    
    for i in range(n_models):
        seed = base_seed + i * 100  # Different seeds
        
        params = {}
        if hasattr(settings.model, 'lgb_params'):
            params = settings.model.lgb_params.copy()
        
        params['random_state'] = seed
        
        # Add some variation in hyperparameters for diversity
        if i > 0:
            # Vary subsample and colsample
            params['subsample'] = 0.7 + (i % 3) * 0.1
            params['colsample_bytree'] = 0.7 + ((i + 1) % 3) * 0.1
        
        config = ModelConfig(params=params, random_state=seed)
        models.append(LightGBMModel(feature_names, config))
    
    logger.info(f"Created ensemble of {n_models} models")
    
    return models


# ============================================
# MODULE TEST
# ============================================

def test_lightgbm_model():
    """Test LightGBM model with synthetic data."""
    print("\n" + "=" * 70)
    print("🧪 TESTING LIGHTGBM MODEL")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 5000
    n_features = 20
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Features: simulating cross-sectionally ranked data (0-1 range)
    X = pd.DataFrame(
        np.random.uniform(0, 1, (n_samples, n_features)),
        columns=feature_names
    )
    
    # Target with some signal
    true_weights = np.random.randn(n_features) * 0.5
    noise = np.random.randn(n_samples) * 0.02
    y = pd.Series(X.values @ true_weights * 0.01 + noise)
    
    # Dates for cross-sectional evaluation
    n_dates = n_samples // 50
    dates = pd.Series(
        pd.date_range('2023-01-01', periods=n_dates, freq='B').repeat(50)[:n_samples]
    )
    
    # Split data (time-aware)
    train_size = int(n_samples * 0.6)
    val_size = int(n_samples * 0.2)
    
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_val, y_val = X.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
    X_test, y_test = X.iloc[train_size+val_size:], y.iloc[train_size+val_size:]
    dates_test = dates.iloc[train_size+val_size:]
    
    print(f"\n📊 Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Test 1: Model creation
    print("\n1️⃣ Testing model creation...")
    model = create_model(feature_names)
    print(f"   ✅ Created: {model}")
    
    # Test 2: Training
    print("\n2️⃣ Testing training...")
    model.fit(X_train, y_train, X_val, y_val)
    print(f"   ✅ Trained: {model.training_info}")
    
    # Test 3: Prediction
    print("\n3️⃣ Testing prediction...")
    predictions = model.predict(X_test)
    print(f"   ✅ Predictions: shape={predictions.shape}, "
          f"range=[{predictions.min():.4f}, {predictions.max():.4f}]")
    
    # Test 4: Evaluation
    print("\n4️⃣ Testing evaluation...")
    metrics = model.evaluate(X_test, y_test, dates=dates_test, prefix='test_')
    print(f"   ✅ Metrics:")
    for k, v in list(metrics.items())[:6]:
        print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
    
    # Test 5: Feature importance
    print("\n5️⃣ Testing feature importance...")
    importance = model.get_feature_importance()
    print(f"   ✅ Top 5 features:")
    print(importance.head())
    
    # Test 6: Save/Load
    print("\n6️⃣ Testing save/load...")
    save_path = Path("test_model_temp.pkl")
    model.save(save_path)
    
    loaded_model = LightGBMModel.load(save_path)
    loaded_preds = loaded_model.predict(X_test)
    
    assert np.allclose(predictions, loaded_preds, equal_nan=True), "Predictions don't match!"
    print(f"   ✅ Save/load verified")
    
    # Cleanup
    save_path.unlink()
    
    # Test 7: Cross-sectional evaluation
    print("\n7️⃣ Testing cross-sectional IC...")
    cs_ic = model.evaluate_cross_sectional(X_test, y_test, dates_test)
    mean_ic = cs_ic['rank_ic'].mean()
    ir = calculate_information_ratio(cs_ic['rank_ic'])
    print(f"   ✅ Mean Rank IC: {mean_ic:.4f}, IR: {ir:.4f}")
    
    # Test 8: Ensemble
    print("\n8️⃣ Testing ensemble creation...")
    ensemble = create_ensemble_model(feature_names, n_models=3)
    print(f"   ✅ Created {len(ensemble)} models")
    
    print("\n✅ All LightGBM model tests passed!")
    
    return model


if __name__ == "__main__":
    test_lightgbm_model()