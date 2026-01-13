"""
Gradient Boosting Models
========================
LightGBM for alpha prediction with walk-forward validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
from pathlib import Path
import sys
import warnings
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings

warnings.filterwarnings('ignore', category=UserWarning)


class LightGBMModel:
    """
    LightGBM regressor for predicting stock returns.
    
    Features:
        - Walk-forward validation
        - Early stopping
        - Feature importance
        - Comprehensive evaluation
        - Model persistence
    
    Example:
        >>> model = LightGBMModel(feature_names)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """
    
    def __init__(self, feature_names: List[str], params: Dict = None):
        """
        Initialize model.
        
        Args:
            feature_names: List of feature columns
            params: LightGBM parameters (optional)
        """
        self.feature_names = feature_names
        self.params = params or settings.model.lgb_params.copy()
        
        # Remove early_stopping_rounds if present (use callbacks instead)
        # This is needed for newer versions of LightGBM
        self.early_stopping_rounds = self.params.pop('early_stopping_rounds', None)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {}
        
        print(f"🤖 LightGBM Model initialized with {len(feature_names)} features")
    
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            verbose: bool = True) -> None:
        """
        Train the model with optional validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print training progress
        """
        if verbose:
            print(f"   🔧 Training on {len(X_train):,} samples...")
        
        # Validate features
        missing_features = set(self.feature_names) - set(X_train.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and scale features
        X_train_features = X_train[self.feature_names]
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Prepare validation data if provided
        eval_set = None
        callbacks = []
        
        if X_val is not None and y_val is not None:
            X_val_features = X_val[self.feature_names]
            X_val_scaled = self.scaler.transform(X_val_features)
            eval_set = [(X_val_scaled, y_val)]
            
            # Add early stopping callback if validation set provided
            if self.early_stopping_rounds:
                callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False))
            
            if verbose:
                print(f"   📊 Validation on {len(X_val):,} samples...")
        
        # Add log evaluation callback
        if not verbose:
            callbacks.append(lgb.log_evaluation(period=0))
        
        # Create and train model
        self.model = lgb.LGBMRegressor(**self.params)
        
        # Fit with or without validation
        if eval_set and callbacks:
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=eval_set,
                eval_names=['validation'],
                callbacks=callbacks
            )
        elif eval_set:
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=eval_set,
                eval_names=['validation']
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        
        # Store training info
        self.training_history = {
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'features_used': len(self.feature_names),
            'best_iteration': getattr(self.model, 'best_iteration', self.params.get('n_estimators', 100))
        }
        
        if verbose:
            print(f"   ✅ Training complete (best iteration: {self.training_history['best_iteration']})")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        # Select and scale features
        X_features = X[self.feature_names]
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame, return_std: bool = False) -> np.ndarray:
        """
        Predict with uncertainty (for ensemble models).
        Currently returns point predictions.
        """
        predictions = self.predict(X)
        
        if return_std:
            # For single model, return zero std
            return predictions, np.zeros_like(predictions)
        
        return predictions
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            X: Features
            y: True targets
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        # Handle NaN values
        mask = ~(y.isna() | np.isnan(predictions))
        y_clean = y[mask].values
        pred_clean = predictions[mask]
        
        if len(y_clean) == 0:
            return {f'{prefix}ic': 0, f'{prefix}rank_ic': 0, f'{prefix}hit_rate': 0}
        
        # Calculate metrics
        metrics = {}
        
        # Information Coefficient (Pearson correlation)
        ic = np.corrcoef(pred_clean, y_clean)[0, 1]
        metrics[f'{prefix}ic'] = ic if not np.isnan(ic) else 0
        
        # Rank Information Coefficient (Spearman correlation)
        rank_ic = pd.Series(pred_clean).corr(pd.Series(y_clean), method='spearman')
        metrics[f'{prefix}rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0
        
        # Hit Rate (directional accuracy)
        hit_rate = np.mean(np.sign(pred_clean) == np.sign(y_clean))
        metrics[f'{prefix}hit_rate'] = hit_rate
        
        # Regression metrics
        metrics[f'{prefix}rmse'] = np.sqrt(mean_squared_error(y_clean, pred_clean))
        metrics[f'{prefix}mae'] = mean_absolute_error(y_clean, pred_clean)
        
        # R-squared
        r2 = r2_score(y_clean, pred_clean)
        metrics[f'{prefix}r2'] = r2 if not np.isnan(r2) else 0
        
        # Information Ratio (IC / IC_std)
        if len(y_clean) > 10:  # Need sufficient samples
            # Calculate rolling IC for IR
            window = min(50, len(y_clean) // 4)
            if window > 5:
                rolling_ic = []
                for i in range(window, len(y_clean)):
                    y_window = y_clean[i-window:i]
                    pred_window = pred_clean[i-window:i]
                    ic_window = np.corrcoef(pred_window, y_window)[0, 1]
                    if not np.isnan(ic_window):
                        rolling_ic.append(ic_window)
                
                if rolling_ic:
                    ic_std = np.std(rolling_ic)
                    metrics[f'{prefix}ir'] = ic / ic_std if ic_std > 1e-10 else 0
                else:
                    metrics[f'{prefix}ir'] = 0
            else:
                metrics[f'{prefix}ir'] = 0
        else:
            metrics[f'{prefix}ir'] = 0
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance DataFrame.
        
        Args:
            importance_type: 'gain', 'split', or 'gain'
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        
        if importance_type == 'gain':
            importance = self.model.feature_importances_
        elif importance_type == 'split':
            importance = self.model.booster_.feature_importance(importance_type='split')
        else:
            importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        df['cumulative_pct'] = df['importance_pct'].cumsum()
        
        return df.reset_index(drop=True)
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        importance_df = self.get_feature_importance()
        return importance_df.head(n)['feature'].tolist()
    
    def save(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path: File path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model!")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted,
            'save_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(data, save_path)
        print(f"   💾 Model saved: {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """
        Load model from file.
        
        Args:
            path: File path to load model from
            
        Returns:
            Loaded LightGBMModel instance
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        data = joblib.load(load_path)
        
        # Create instance
        instance = cls(data['feature_names'], data['params'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.training_history = data.get('training_history', {})
        instance.is_fitted = data.get('is_fitted', True)
        
        print(f"   📂 Model loaded: {load_path}")
        if 'save_timestamp' in data:
            print(f"      Saved: {data['save_timestamp']}")
        
        return instance
    
    def get_model_info(self) -> Dict:
        """Get model information summary."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        info = {
            'status': 'fitted',
            'n_features': len(self.feature_names),
            'model_type': 'LightGBM',
            'params': self.params,
            'training_history': self.training_history
        }
        
        return info


def create_model(feature_names: List[str], custom_params: Dict = None) -> LightGBMModel:
    """
    Factory function to create LightGBM model.
    
    Args:
        feature_names: List of feature names
        custom_params: Custom parameters to override defaults
        
    Returns:
        Configured LightGBMModel
    """
    params = settings.model.lgb_params.copy()
    
    if custom_params:
        params.update(custom_params)
    
    return LightGBMModel(feature_names, params)