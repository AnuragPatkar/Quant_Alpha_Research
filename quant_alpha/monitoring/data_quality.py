"""
Data Quality & Integrity Monitor
================================
Real-time validation engine for market data ingestion and feature distribution stability.

Purpose
-------
The `DataQualityMonitor` serves as the first line of defense in the production pipeline.
It enforces strict schema validation and monitors for **Covariate Shift** (Feature Drift)
using statistical divergence metrics. It ensures that the data feeding the inference
models matches the statistical properties of the training data.

Usage
-----
This module is designed for zero-dependency integration into live ingestion loops.

.. code-block:: python

    from quant_alpha.monitoring.data_quality import DataQualityMonitor

    # 1. Initialize with training set reference data
    monitor = DataQualityMonitor(reference_data=train_df)

    # 2. Validate live batch
    report = monitor.check_incoming_data(live_batch, data_type='features')

    if report['status'] == 'FAIL':
        raise DataIntegrityError(report['issues'])

Importance
----------
- **Alpha Preservation**: Prevents "Garbage In, Garbage Out". Models predicting on
  drifted features (e.g., regime change) yield undefined behavior.
- **Operational Safety**: Detects upstream vendor failures (e.g., null prices, zero volume)
  before they propagate to execution algos.
- **Statistical Rigor**: Uses Population Stability Index ($PSI$) to quantify drift magnitude,
  providing a standardized metric for model retraining triggers.

Tools & Frameworks
------------------
- **Pandas**: Efficient columnar validation and null checking.
- **NumPy**: Histogram generation and vector mathematics for PSI calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """
    Production-grade validator for time-series and cross-sectional data batches.

    Capabilities:
    1. **Schema Validation**: Enforces type safety, column existence, and domain constraints.
    2. **Covariate Shift Detection**: Computes Population Stability Index ($PSI$) against a reference baseline.
    3. **Anomaly Detection**: Identifies null spikes and impossible values (e.g., negative prices).
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_data (Optional[pd.DataFrame]): Historical baseline (Training Set).
                Required for calculating PSI drift metrics.
        """
        self.reference_data = reference_data
        
    def set_reference_data(self, data: pd.DataFrame):
        """Updates the baseline distribution (e.g., post-retraining)."""
        self.reference_data = data
        logger.info(f"Reference data updated. Shape: {data.shape}")
        
    def check_incoming_data(self, data: pd.DataFrame, data_type: str = 'prices') -> Dict[str, Any]:
        """
        Executes comprehensive validation logic on ingress data streams.
        
        Args:
            data (pd.DataFrame): The live data batch to validate.
            data_type (str): Context tag ('prices', 'fundamentals', 'features').
            
        Returns:
            Dict[str, Any]: Validation report with status ('PASS'|'WARNING'|'FAIL') and issue logs.
        """
        report = {
            'status': 'PASS', 
            'issues': [], 
            'timestamp': datetime.now().isoformat(),
            'rows': len(data)
        }
        
        if data.empty:
            report['status'] = 'FAIL'
            report['issues'].append("Incoming data is empty")
            return report
        
        # 1. Schema & Domain Validation (Critical Path)
        integrity_issues = self._check_integrity(data, data_type)
        if integrity_issues:
            report['status'] = 'FAIL'
            report['issues'].extend(integrity_issues)
            # Circuit Breaker: Abort immediately if schema is invalid to prevent cascading failures.
            return report

        # 2. Distributional Drift Analysis (Non-Blocking Warning)
        # Only performed if reference baseline exists and data implies feature vectors.
        if self.reference_data is not None:
            drift_report = self._check_feature_drift(data)
            
            if drift_report['drift_detected']:
                if report['status'] == 'PASS':
                    report['status'] = 'WARNING'
                report['issues'].extend(drift_report['drifted_features'])
                report['psi_scores'] = drift_report['psi_scores']
                
        return report

    def _check_integrity(self, data: pd.DataFrame, data_type: str) -> List[str]:
        """Performs deterministic checks on schema validity and domain constraints."""
        issues = []
        
        # Check 1: Column Existence (Schema Validation)
        required_cols = []
        if data_type == 'prices':
            required_cols = ['ticker', 'date', 'close']
            # Liquidity Constraint: Volume is essential for tradeability checks.
            if 'volume' not in data.columns:
                issues.append("Missing recommended column: 'volume'")
        elif data_type == 'features':
            # Feature Consistency: Ensure live feature set matches training schema.
            if self.reference_data is not None:
                # Check for missing columns that are in reference (excluding metadata)
                ref_cols = [c for c in self.reference_data.columns if c not in ['target', 'label']]
                missing = [c for c in ref_cols if c not in data.columns]
                if missing:
                    issues.append(f"Missing feature columns: {missing[:5]}...")
        
        for col in required_cols:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        if issues:
            return issues
            
        # Check 2: Critical Nullity (Sparse Data Detection)
        if data_type == 'prices':
            if data['close'].isnull().any():
                issues.append("Null values detected in 'close' price")
            
            # Check 3: Domain Constraints (Non-Negativity)
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    if (data[col] < 0).any():
                        issues.append(f"Negative values detected in '{col}'")
                
        return issues

    def _check_feature_drift(self, current_data: pd.DataFrame, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Quantifies distributional shift using Population Stability Index ($PSI$).
        
        PSI Interpretation:
        - **PSI < 0.1**: Stable distribution.
        - **0.1 <= PSI <= 0.25**: Minor shift (Monitoring recommended).
        - **PSI > 0.25**: Critical drift (Retraining required).
        """
        drifted_features = []
        psi_scores = {}
        
        # Identify common numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        ref_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        common_cols = list(set(numeric_cols) & set(ref_cols))
        
        # Exclude metadata
        exclude = ['date', 'ticker', 'target', 'label', 'prediction', 'sector', 'industry']
        cols_to_check = [c for c in common_cols if c not in exclude]
        
        for col in cols_to_check:
            try:
                psi = self._calculate_psi(self.reference_data[col], current_data[col])
                psi_scores[col] = psi
                if psi > threshold:
                    drifted_features.append(f"Drift in {col}: PSI={psi:.2f}")
            except Exception:
                continue
                
        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'psi_scores': psi_scores
        }

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """
        Computes the Kullback-Leibler divergence-based PSI metric.
        
        Math:
        .. math::
            PSI = \sum (P_{expected} - P_{actual}) \times \ln(\\frac{P_{expected}}{P_{actual}})
        """
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()
        
        if len(expected_clean) == 0 or len(actual_clean) == 0:
            return 0.0

        # 1. Binning Strategy: Quantile-based discretization on Training Data
        # Ensures bins are robust to outliers and scale-invariant.
        try:
            breakpoints = np.linspace(0, 100, buckets + 1)
            bins = np.percentile(expected_clean, breakpoints)
            bins = np.unique(bins) # Handle cases with many identical values (e.g. 0s)
            
            # Fallback if unique values are too few
            if len(bins) < 2:
                # Degenerate Case: Constant feature. Check if mean shifted significantly.
                if abs(expected_clean.mean() - actual_clean.mean()) > 1e-6:
                    return 1.0 # Max drift
                return 0.0
                
            # Boundary Handling: Extend to +/- infinity to capture Out-of-Sample outliers.
            bins[0] = -np.inf
            bins[-1] = np.inf
            
        except Exception:
            return 0.0
            
        # 2. Frequency Enumeration
        expected_counts, _ = np.histogram(expected_clean, bins)
        actual_counts, _ = np.histogram(actual_clean, bins)

        # 3. Probability Mass Function (PMF) Derivation
        expected_dist = expected_counts / len(expected_clean)
        actual_dist = actual_counts / len(actual_clean)
        
        # 4. Zero-Frequency Smoothing ($\epsilon$ injection)
        # Prevents division by zero in log calculations.
        epsilon = 1e-5
        expected_dist = np.maximum(expected_dist, epsilon)
        actual_dist = np.maximum(actual_dist, epsilon)
        
        # 5. PSI Calculation
        psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
        return np.sum(psi_values)