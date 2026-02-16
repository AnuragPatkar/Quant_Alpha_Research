"""
Earnings Revision Factors (Production Grade)
Focus: Estimate revisions, earnings trajectory, and estimate accuracy trends.
"""

import pandas as pd
import numpy as np
from ..base import EarningsFactor
from ..registry import FactorRegistry
from config.logging_config import logger
from .utils import detect_earnings_events, get_events_with_surprise

@FactorRegistry.register()
class EarningsMomentum(EarningsFactor):
    """
    Earnings Growth Momentum: QoQ EPS Growth Rate
    Formula: (Current EPS - Previous EPS) / max(abs(Previous EPS), 0.01) * 100
    """
    def __init__(self):
        super().__init__(name='earn_eps_momentum', description='EPS Growth Momentum (Q/Q)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_momentum(group):
            is_new_event = detect_earnings_events(group)
            events = group.loc[is_new_event].copy()
            
            if len(events) < 2:
                return pd.Series(0.0, index=group.index)
            
            curr_eps = events['eps_actual']
            prev_eps = events['eps_actual'].shift(1)
            
            # Use 0.01 floor to prevent division by near-zero (common in penny stocks/turnarounds)
            denom = prev_eps.abs().clip(lower=0.01)
            events['eps_growth'] = (curr_eps - prev_eps) / denom * 100
            
            # Clip at 500% to prevent extreme outliers from dominating the cross-section
            return events['eps_growth'].clip(-500, 500).reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)


@FactorRegistry.register()
class RecentPositiveRevisions(EarningsFactor):
    """
    Recent Positive Revisions: % of last 3 quarters with positive surprises.
    """
    def __init__(self):
        super().__init__(name='earn_recent_positive_revisions', description='Recent Positive Revisions %')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc_revisions(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Handle NaNs: If surprise is missing, don't count as negative, keep as NaN
            events['positive'] = np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float))
            events['revision_pct'] = events['positive'].rolling(window=3, min_periods=1).mean() * 100
            
            return events['revision_pct'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_revisions)


@FactorRegistry.register()
class EstimateAccuracyTrend(EarningsFactor):
    """
    Estimate Accuracy Trend: Inverse of recent surprise magnitude.
    Uses Median for robustness against one-off massive surprises.
    """
    def __init__(self):
        super().__init__(name='earn_estimate_accuracy_trend', description='Estimate Accuracy Trend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Check for required columns or ability to compute them
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc_accuracy(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Using median is much more robust for 'predictability' than mean
            events['med_surprise_mag'] = events['surprise_pct'].abs().rolling(window=4, min_periods=2).median()
            
            # Inverse relationship: higher score = lower surprise magnitude
            events['accuracy'] = (100 - events['med_surprise_mag']).clip(lower=0, upper=100)
            
            return events['accuracy'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_accuracy)


@FactorRegistry.register()
class EPSAcceleration(EarningsFactor):
    """
    EPS Acceleration: The second derivative of earnings.
    Formula: Current QoQ Growth - Previous QoQ Growth
    """
    def __init__(self):
        super().__init__(name='earn_eps_acceleration', description='EPS Acceleration (Growth Rate Change)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_acceleration(group):
            is_new_event = detect_earnings_events(group)
            events = group.loc[is_new_event].copy()
            
            if len(events) < 3:
                return pd.Series(np.nan, index=group.index)
            
            # Step 1: Growth
            denom = events['eps_actual'].shift(1).abs().clip(lower=0.01)
            events['eps_growth'] = (events['eps_actual'] - events['eps_actual'].shift(1)) / denom * 100
            
            # Step 2: Acceleration (Change in growth rate)
            events['acceleration'] = events['eps_growth'].diff()
            
            # Clip acceleration to prevent extreme values from distorting the model
            return events['acceleration'].clip(-200, 200).reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_acceleration)