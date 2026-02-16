"""
Earnings Estimate Quality Factors (Production Grade)
Focus: Analyst estimate reliability, consensus strength, and forward expectations.
"""

import pandas as pd
import numpy as np
from ..base import EarningsFactor
from ..registry import FactorRegistry
from .utils import get_events_with_surprise

@FactorRegistry.register()
class ConsensusStrength(EarningsFactor):
    """
    Consensus Strength: Inverse of Average % error.
    Using Median for robustness against outlier quarters.
    """
    def __init__(self):
        super().__init__(name='est_consensus_strength', description='Analyst Consensus Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Using median error over 4Q is safer than mean for consensus
            events['err'] = events['surprise_pct'].abs()
            events['score'] = (100 - events['err'].rolling(4, min_periods=2).median()).clip(0, 100)
            return events['score'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class EstimateSurpriseConsistency(EarningsFactor):
    """
    Estimate Surprise Consistency: Inverse of surprise volatility.
    Uses a decay-style normalization instead of linear subtraction.
    """
    def __init__(self):
        super().__init__(name='est_surprise_consistency', description='Estimate Surprise Consistency')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if len(events) < 3: return pd.Series(np.nan, index=group.index)

            # Volatility of surprises
            vols = events['surprise_pct'].rolling(4, min_periods=3).std()
            
            # Robust Scaling: 1 / (1 + vol/10). 
            # If vol is 0, score is 1.0. If vol is 20, score is 0.33.
            events['consistency'] = 1.0 / (1.0 + (vols / 10.0))
            return events['consistency'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class PositiveEstimateConfidence(EarningsFactor):
    """
    Positive Estimate Confidence: % of recent quarters beating estimates.
    Requires at least 3 quarters for a valid 'confidence' signal.
    """
    def __init__(self):
        super().__init__(name='est_positive_confidence', description='Positive Estimate Confidence')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Handle NaNs: If surprise is missing, don't count as miss, keep as NaN
            beats = pd.Series(np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float)), index=events.index)
            
            # Use a 6-quarter window but require 3 to start signaling
            events['conf'] = beats.rolling(window=6, min_periods=3).mean() * 100
            return events['conf'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class EstimateGuidanceQuality(EarningsFactor):
    """
    Guidance Quality: Accuracy of forward guidance.
    Focuses on the deviation between what analysts thought and what happened.
    """
    def __init__(self):
        super().__init__(name='est_guidance_quality', description='Estimate Guidance Quality')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Relative Error (normalized)
            rel_error = events['surprise_pct'].abs() / 100.0
            avg_err = rel_error.rolling(4, min_periods=2).mean()
            
            events['quality'] = 1.0 / (1.0 + avg_err)
            return events['quality'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)