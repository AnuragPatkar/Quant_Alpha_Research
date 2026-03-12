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
        Args:
            rolling_window (int): The lookback period $N$ for establishing baseline statistics.
        """
        self.window = rolling_window
        # State Management: O(1) append/pop operations for sliding window history.
        self.history = deque(maxlen=252) 
        
    def update(self, date: str, predictions: pd.Series, actual_returns: Optional[pd.Series] = None):
        """
        Ingests daily inference results and (optional) realized returns.

        Args:
            date (str): ISO-8601 date string.
            predictions (pd.Series): Vector of model alpha scores $\hat{y}$.
            actual_returns (Optional[pd.Series]): Vector of realized forward returns $y$.
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
            
            # Error Metric: Mean Squared Error ($MSE$)
            # Data Alignment: Intersection of indices (Inner Join) required for vector subtraction.
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
        Performs statistical tests to identify significant regime shifts.

        Returns:
            Dict: Status report containing drift flags and Z-score metrics.
        """
        if len(self.history) < self.window * 2:
            return {'status': 'INSUFFICIENT_DATA'}
            
        df = pd.DataFrame(list(self.history))
        
        # Partitioning Strategy:
        # 1. Current Window: The most recent $N$ days (Test distribution).
        # 2. Reference Window: The preceding $N$ days (Baseline distribution).
        current = df.tail(self.window)
        reference = df.iloc[-(self.window * 2):-self.window]
        
        alerts = []
        
        # 1. Directional Bias Test: Z-Score of Prediction Means
        # Detects if the model's central tendency ($\mu$) has shifted significantly.
        # .. math::
        #     Z = \frac{\mu_{current} - \mu_{reference}}{\sigma_{reference}}
        curr_mean = current['pred_mean'].mean()
        ref_mean = reference['pred_mean'].mean()
        ref_std = reference['pred_mean'].std()
        
        # Numerical Stability: Epsilon check for division by zero
        if ref_std > 1e-6:
            z_score = (curr_mean - ref_mean) / ref_std
            if abs(z_score) > 2.0:
                alerts.append(f"Prediction Mean Shift: Z-Score {z_score:.2f}")
            
        # 2. Performance Degradation Test
        # Checks for a relative increase in MSE compared to the reference baseline.
        if 'mse' in df.columns and not df['mse'].isnull().all():
            current_mse = current['mse'].mean()
            ref_mse = reference['mse'].mean()
            
            # Threshold: > 50% degradation in Mean Squared Error ($MSE$)
            if ref_mse > 0 and current_mse > ref_mse * 1.5:
                alerts.append(f"MSE Degradation: +{(current_mse/ref_mse - 1):.1%}")
                
        return {
            'drift_detected': len(alerts) > 0,
            'alerts': alerts,
            'current_pred_mean': curr_mean
        }