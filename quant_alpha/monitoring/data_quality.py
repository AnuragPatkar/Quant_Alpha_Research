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

    # Initialize with historical training set reference data bounds
    monitor = DataQualityMonitor(reference_data=train_df)

    # Validate live inbound array batches dynamically
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
    1. Schema Validation: Enforces type safety, column existence, and scalar domain bounds.
    2. Covariate Shift Detection: Computes Population Stability Index (PSI) against reference bases.
    3. Anomaly Detection: Tracks null spikes and systemic impossible states.
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initializes the dynamic data quality bounds checker.
        
        Args:
            reference_data (Optional[pd.DataFrame]): Historical standardized baseline representing 
                the training set. Explicitly required for evaluating PSI drift coefficients.
        """
        self.reference_data = reference_data
        
    def set_reference_data(self, data: pd.DataFrame):
        """
        Overwrites the baseline distribution tracking continuous drift variables.
        
        Employed directly post-retraining to re-anchor statistical benchmarks.
        
        Args:
            data (pd.DataFrame): The new training tensor block limit map.
        """
        self.reference_data = data
        logger.info(f"Reference data updated. Shape: {data.shape}")
        
    def check_incoming_data(self, data: pd.DataFrame, data_type: str = 'prices') -> Dict[str, Any]:
        """
        Executes comprehensive validation logic sequences against live ingress data arrays.
        
        Args:
            data (pd.DataFrame): The live incoming temporal matrix block to validate.
            data_type (str): Topological context map classification 
                (e.g., 'prices', 'fundamentals', 'features').
            
        Returns:
            Dict[str, Any]: Consolidated dictionary bounds mapping output status identifiers 
                ('PASS', 'WARNING', 'FAIL') paired alongside discrete extracted issue logs.
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
        
        # Evaluates core absolute Schema & Domain Verification paths strictly isolating matrices
        integrity_issues = self._check_integrity(data, data_type)
        if integrity_issues:
            report['status'] = 'FAIL'
            report['issues'].extend(integrity_issues)
            
            return report

        if self.reference_data is not None:
            drift_report = self._check_feature_drift(data)
            
            if drift_report['drift_detected']:
                if report['status'] == 'PASS':
                    report['status'] = 'WARNING'
                report['issues'].extend(drift_report['drifted_features'])
                report['psi_scores'] = drift_report['psi_scores']
                
        return report

    def _check_integrity(self, data: pd.DataFrame, data_type: str) -> List[str]:
        """
        Evaluates absolute deterministic states validating schema schemas against target bounds.
        
        Args:
            data (pd.DataFrame): Input dimensional bounds matrix.
            data_type (str): Topological structure label definition.
            
        Returns:
            List[str]: Formatted array defining structural error parameters extracted during analysis.
        """
        issues = []
        
        required_cols = []
        if data_type == 'prices':
            required_cols = ['ticker', 'date', 'close']
            if 'volume' not in data.columns:
                issues.append("Missing recommended column: 'volume'")
        elif data_type == 'features':
            if self.reference_data is not None:
                ref_cols = [c for c in self.reference_data.columns if c not in ['target', 'label']]
                missing = [c for c in ref_cols if c not in data.columns]
                if missing:
                    issues.append(f"Missing feature columns: {missing[:5]}...")
        
        for col in required_cols:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        if issues:
            return issues
            
        if data_type == 'prices':
            if data['close'].isnull().any():
                issues.append("Null values detected in 'close' price")
            
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    if (data[col] < 0).any():
                        issues.append(f"Negative values detected in '{col}'")
                
        return issues

    def _check_feature_drift(self, current_data: pd.DataFrame, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Computes distributional shifts quantitatively via Population Stability Index evaluations.
        
        PSI Interpretation:
        - **PSI < 0.1**: Stable distribution.
        - **0.1 <= PSI <= 0.25**: Minor shift (Monitoring recommended).
        - **PSI > 0.25**: Critical drift (Retraining required).
        
        Args:
            current_data (pd.DataFrame): Incoming continuous out-of-sample data.
            threshold (float): Parametric scalar bounded for maximal limit warnings. Defaults to 0.25.
            
        Returns:
            Dict[str, Any]: Boundary mappings exposing empirical drift evaluations.
        """
        drifted_features = []
        psi_scores = {}
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        ref_cols = self.reference_data.select_dtypes(include=[np.number]).columns
        common_cols = list(set(numeric_cols) & set(ref_cols))
        
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
        Formulates continuous Kullback-Leibler divergence-based metrics evaluating boundary disparities.
        
        Math:
        .. math::
            PSI = \sum (P_{expected} - P_{actual}) \times \ln(\\frac{P_{expected}}{P_{actual}})
            
        Args:
            expected (pd.Series): Bound expected vector arrays generated in-sample.
            actual (pd.Series): Actual discrete out-of-sample observed metrics.
            buckets (int): Grouping density limits utilized for discrete evaluation bounds.
            
        Returns:
            float: Aggregate absolute mathematical PSI magnitude limits.
        """
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()
        
        if len(expected_clean) == 0 or len(actual_clean) == 0:
            return 0.0

        try:
            breakpoints = np.linspace(0, 100, buckets + 1)
            bins = np.percentile(expected_clean, breakpoints)
            bins = np.unique(bins)
            
            if len(bins) < 2:
                if abs(expected_clean.mean() - actual_clean.mean()) > 1e-6:
                    return 1.0 # Max drift
                return 0.0
                
            bins[0] = -np.inf
            bins[-1] = np.inf
            
        except Exception:
            return 0.0
            
        expected_counts, _ = np.histogram(expected_clean, bins)
        actual_counts, _ = np.histogram(actual_clean, bins)

        expected_dist = expected_counts / len(expected_clean)
        actual_dist = actual_counts / len(actual_clean)
        
        epsilon = 1e-5
        expected_dist = np.maximum(expected_dist, epsilon)
        actual_dist = np.maximum(actual_dist, epsilon)
        
        psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
        return np.sum(psi_values)