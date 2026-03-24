"""
Model Drift & Regime Detection System
=====================================
Statistical monitoring engine for detecting Concept Drift and Label Shift in live model predictions.

Purpose
-------
The `ModelDriftDetector` identifies stationarity violations in the model's prediction distribution
($P(\hat{y})$) or the underlying return distribution ($P(y)$). It serves as an early warning system
for "Model Decay," triggering retraining protocols when performance metrics degrade beyond
statistically significant thresholds.

Usage
-----
Intended for integration into the daily post-market reporting loop.

.. code-block:: python

    from quant_alpha.monitoring.model_drift import ModelDriftDetector

    detector = ModelDriftDetector(rolling_window=30)

    # Daily Update
    detector.update(
        date="2023-10-27",
        predictions=daily_preds_series,
        actual_returns=daily_returns_series
    )

    # Check for Drift
    status = detector.detect_drift()
    if status['drift_detected']:
        print(f"Drift Alert: {status['alerts']}")

Importance
----------
- **Alpha Preservation**: Detects when market regimes shift, rendering the trained model obsolete.
- **Risk Mitigation**: Identifies "Silent Failure" modes where the model becomes structurally
  biased (e.g., persistently bullish during a crash).
- **Automated Governance**: Provides quantitative triggers ($Z > 2.0$) for model lifecycle management.

Tools & Frameworks
------------------
- **Pandas**: Time-series alignment and rolling window management.
- **NumPy**: Efficient calculation of statistical moments (mean, variance).
- **Deque**: $O(1)$ memory management for rolling history buffers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ModelDriftDetector:
    """
    Quantifies distributional shifts using rolling-window statistics.

    Monitoring Metrics:
    1.  **Concept Drift**: Changes in the relationship between features and targets ($P(y|X)$),
        proxied by Mean Squared Error ($MSE$) degradation.
    2.  **Prediction Bias**: Shifts in the central tendency of model outputs ($\mu_{\hat{y}}$),
        indicating structural directional bias (too bullish/bearish).
    """
    
    def __init__(self, rolling_window: int = 30):
        """
        Initializes the dynamic probabilistic drift analyzer bounded by structural sequences.
        
        Args:
            rolling_window (int): The trailing lookback horizon length $N$ designated for 
                establishing strict continuous baseline statistics. Defaults to 30.
        """
        self.window = rolling_window
        self.history = deque(maxlen=252) 
        
    def update(self, date: str, predictions: pd.Series, actual_returns: Optional[pd.Series] = None):
        """
        Ingests point-in-time cross-sectional daily inference vectors and mapped future returns.

        Args:
            date (str): Discrete standard ISO-8601 coordinate evaluation metric.
            predictions (pd.Series): Array bounding standardized model alpha predictions ($\hat{y}$).
            actual_returns (Optional[pd.Series]): Array evaluating realized corresponding metrics ($y$).
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
        Evaluates bounds identifying structural statistically significant predictive regime shifts.

        Returns:
            Dict: Configured boundaries detailing distinct drift warnings and standard Z-score deviations.
        """
        if len(self.history) < self.window * 2:
            return {'status': 'INSUFFICIENT_DATA'}
            
        df = pd.DataFrame(list(self.history))
        
        current = df.tail(self.window)
        reference = df.iloc[-(self.window * 2):-self.window]
        
        alerts = []
        
        # Analyzes statistical prediction biases calculating the uniform directional bounds Z-Scores
        # Functionally tracking if internal mapping ($\mu$) strictly isolates significant variance shifts.
        curr_mean = current['pred_mean'].mean()
        ref_mean = reference['pred_mean'].mean()
        ref_std = reference['pred_mean'].std()
        
        if ref_std > 1e-6:
            z_score = (curr_mean - ref_mean) / ref_std
            if abs(z_score) > 2.0:
                alerts.append(f"Prediction Mean Shift: Z-Score {z_score:.2f}")
            
        # Evaluates absolute metric performance structural degradations isolating discrete 
        # percentage deviations strictly measuring historical out-of-sample prediction baselines.
        if 'mse' in df.columns and not df['mse'].isnull().all():
            current_mse = current['mse'].mean()
            ref_mse = reference['mse'].mean()
            
            if ref_mse > 0 and current_mse > ref_mse * 1.5:
                alerts.append(f"MSE Degradation: +{(current_mse/ref_mse - 1):.1%}")
                
        return {
            'drift_detected': len(alerts) > 0,
            'alerts': alerts,
            'current_pred_mean': curr_mean
        }