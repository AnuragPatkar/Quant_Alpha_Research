"""
Gradient Boosting Models
========================
LightGBM for alpha prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings


class LightGBMModel:
    """LightGBM regressor for predicting stock returns."""
    
    def __init__(self, feature_names: List[str], params: Dict = None):
        """
        Initialize model.
        
        Args:
            feature_names: List of feature columns
            params: LightGBM parameters (optional)
        """
        self.feature_names = feature_names
        self.params = params or settings.model.lgb_params
        self.model = lgb.LGBMRegressor(**self.params)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model.
        
        Returns:
            Dict with IC, Rank IC, Hit Rate
        """
        pred = self.predict(X)
        
        # Handle NaN
        mask = ~(y.isna() | np.isnan(pred))
        y_clean = y[mask].values
        pred_clean = pred[mask]
        
        if len(y_clean) == 0:
            return {'ic': 0, 'rank_ic': 0, 'hit_rate': 0}
        
        return {
            'ic': np.corrcoef(pred_clean, y_clean)[0, 1],
            'rank_ic': pd.Series(pred_clean).corr(pd.Series(y_clean), method='spearman'),
            'hit_rate': np.mean(np.sign(pred_clean) == np.sign(y_clean))
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance DataFrame."""
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        
        importance = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
        
        return df.reset_index(drop=True)
    
    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'params': self.params
        }
        joblib.dump(data, path)
        print(f"   💾 Model saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """Load model from file."""
        data = joblib.load(path)
        instance = cls(data['feature_names'], data['params'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.is_fitted = True
        return instance