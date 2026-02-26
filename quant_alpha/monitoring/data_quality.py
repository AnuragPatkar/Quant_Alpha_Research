"""
Data Quality Monitor
Monitors live data integrity and feature distributions.

Designed for production/live deployment.
Self-contained validation to avoid dependencies on backtest libraries.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """
    Production Data Quality Monitor
    
    Checks:
    1. Raw Data Integrity (Schema, NaNs, Types)
    2. Feature Distribution Shifts (PSI - Population Stability Index)
    3. Missing Data Spikes
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Args:
            reference_data: Historical data to compare distributions against (training set).
                            Should contain the features used in the model.
        """
        self.reference_data = reference_data
        
    def set_reference_data(self, data: pd.DataFrame):
        """Update reference data (e.g. after model retraining)"""
        self.reference_data = data
        logger.info(f"Reference data updated. Shape: {data.shape}")
        
    def check_incoming_data(self, data: pd.DataFrame, data_type: str = 'prices') -> Dict[str, Any]:
        """
        Check daily incoming data batch for validity and drift.
        
        Args:
            data: New data batch
            data_type: 'prices', 'fundamentals', or 'features'
            
        Returns:
            Dictionary containing status and list of issues.
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
        
        # 1. Basic Integrity Checks (Self-contained for Production)
        integrity_issues = self._check_integrity(data, data_type)
        if integrity_issues:
            report['status'] = 'FAIL'
            report['issues'].extend(integrity_issues)
            # If critical integrity fails, we might want to stop or continue depending on severity.
            # For now, we return early if schema is broken.
            return report

        # 2. Check for Feature Drift (if reference data exists)
        # We only check drift if we have reference data and the data looks like features (numeric)
        if self.reference_data is not None:
            drift_report = self._check_feature_drift(data)
            
            if drift_report['drift_detected']:
                # Drift is usually a WARNING, not a hard FAIL (unless extreme)
                if report['status'] == 'PASS':
                    report['status'] = 'WARNING'
                report['issues'].extend(drift_report['drifted_features'])
                report['psi_scores'] = drift_report['psi_scores']
                
        return report

    def _check_integrity(self, data: pd.DataFrame, data_type: str) -> List[str]:
        """Perform basic data integrity checks suitable for live production."""
        issues = []
        
        # Check 1: Required Columns
        required_cols = []
        if data_type == 'prices':
            required_cols = ['ticker', 'date', 'close']
            # Volume is usually critical for liquidity filters
            if 'volume' not in data.columns:
                issues.append("Missing recommended column: 'volume'")
        elif data_type == 'features':
            # If we have reference data, ensure columns match
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
            
        # Check 2: Nulls in critical columns
        if data_type == 'prices':
            if data['close'].isnull().any():
                issues.append("Null values detected in 'close' price")
            
            # Check 3: Negative values (Impossible for Price/Volume)
            for col in ['close', 'open', 'high', 'low', 'volume']:
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    if (data[col] < 0).any():
                        issues.append(f"Negative values detected in '{col}'")
                
        return issues

    def _check_feature_drift(self, current_data: pd.DataFrame, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) for features.
        
        PSI Interpretation:
        < 0.1: Stable
        0.1 - 0.25: Minor Shift
        > 0.25: Major Drift (Model likely needs retraining)
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
        """Calculate PSI between two distributions using robust binning."""
        expected_clean = expected.dropna()
        actual_clean = actual.dropna()
        
        if len(expected_clean) == 0 or len(actual_clean) == 0:
            return 0.0

        # 1. Define bins based on Expected distribution (Training Data)
        # We use percentiles to ensure bins are meaningful
        try:
            breakpoints = np.linspace(0, 100, buckets + 1)
            bins = np.percentile(expected_clean, breakpoints)
            bins = np.unique(bins) # Handle cases with many identical values (e.g. 0s)
            
            # Fallback if unique values are too few
            if len(bins) < 2:
                # If essentially constant, check if mean changed significantly
                if abs(expected_clean.mean() - actual_clean.mean()) > 1e-6:
                    return 1.0 # Max drift
                return 0.0
                
            # Extend bins to -inf/inf to capture outliers in actual data
            bins[0] = -np.inf
            bins[-1] = np.inf
            
        except Exception:
            return 0.0
            
        # 2. Calculate frequencies
        expected_counts, _ = np.histogram(expected_clean, bins)
        actual_counts, _ = np.histogram(actual_clean, bins)

        # 3. Normalize to probabilities
        expected_dist = expected_counts / len(expected_clean)
        actual_dist = actual_counts / len(actual_clean)
        
        # 4. Handle zeros (smoothing)
        epsilon = 1e-5
        expected_dist = np.maximum(expected_dist, epsilon)
        actual_dist = np.maximum(actual_dist, epsilon)
        
        # 5. PSI Formula
        psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
        return np.sum(psi_values)