"""
Model Drift Detector
Monitors prediction distribution and concept drift.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ModelDriftDetector:
    """
    Detects Concept Drift (P(y|X) changes) and Label Drift (P(y) changes).
    
    Tracks:
    1. Prediction Mean/Std (Are we becoming too bullish/bearish?)
    2. Prediction Error (MSE) - If actual returns are available
    """
    
    def __init__(self, rolling_window: int = 30):
        self.window = rolling_window
        # Store history as list of dicts
        self.history = deque(maxlen=252) 
        
    def update(self, date: str, predictions: pd.Series, actual_returns: Optional[pd.Series] = None):
        """
        Update with daily predictions and (optional) actual returns
        """
        record = {
            'date': date,
            'pred_mean': predictions.mean(),
            'pred_std': predictions.std(),
            'pred_min': predictions.min(),
            'pred_max': predictions.max()
        }
        
        if actual_returns is not None:
            record['actual_mean'] = actual_returns.mean()
            record['actual_std'] = actual_returns.std()
            
            # Calculate MSE (Mean Squared Error) for this batch
            # Align indices first
            common_idx = predictions.index.intersection(actual_returns.index)
            if len(common_idx) > 0:
                mse = np.mean((predictions[common_idx] - actual_returns[common_idx]) ** 2)
                record['mse'] = mse
                record['mse_count'] = len(common_idx)
            elif not predictions.empty and not actual_returns.empty:
                logger.warning(f"No common indices found between predictions ({len(predictions)}) and actuals ({len(actual_returns)}). Check ticker formats.")
            
        self.history.append(record)
        
    def detect_drift(self) -> Dict:
        """
        Check if recent model behavior deviates from history
        """
        if len(self.history) < self.window * 2:
            return {'status': 'INSUFFICIENT_DATA'}
            
        df = pd.DataFrame(list(self.history))
        
        # Split into reference (older) and current (recent)
        current = df.tail(self.window)
        # Reference is the window before current
        reference = df.iloc[-(self.window * 2):-self.window]
        
        alerts = []
        
        # 1. Check Prediction Mean Shift (Model becoming too bullish/bearish)
        curr_mean = current['pred_mean'].mean()
        ref_mean = reference['pred_mean'].mean()
        ref_std = reference['pred_mean'].std()
        
        # Avoid division by zero
        if ref_std > 1e-6:
            z_score = (curr_mean - ref_mean) / ref_std
            if abs(z_score) > 2.0:
                alerts.append(f"Prediction Mean Shift: Z-Score {z_score:.2f}")
            
        # 2. Check Error Rate Increase (If actuals available)
        if 'mse' in df.columns and not df['mse'].isnull().all():
            current_mse = current['mse'].mean()
            ref_mse = reference['mse'].mean()
            
            if ref_mse > 0 and current_mse > ref_mse * 1.5: # 50% increase in error
                alerts.append(f"MSE Degradation: +{(current_mse/ref_mse - 1):.1%}")
                
        return {
            'drift_detected': len(alerts) > 0,
            'alerts': alerts,
            'current_pred_mean': curr_mean
        }