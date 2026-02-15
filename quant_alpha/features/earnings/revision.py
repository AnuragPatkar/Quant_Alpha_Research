"""
Earnings Revision Factors (Production Grade) - TOP 4 MOST IMPORTANT
Focus: Estimate revisions, earnings trajectory, and estimate accuracy trends.

Factors (Ranked by Predictive Power in Academic Literature):
1. Earnings Momentum -> Direction and acceleration of EPS growth (Most Proven)
2. Recent Positive Revisions -> Trend of estimate revisions (beating trend)
3. Estimate Accuracy Trend -> Are estimates improving or deteriorating
4. EPS Acceleration -> Is earnings growth accelerating or slowing
"""

import pandas as pd
import numpy as np
from ..base import EarningsFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class EarningsMomentum(EarningsFactor):
    """
    Earnings Growth Momentum: QoQ EPS Growth Rate
    Formula: (Current EPS - Previous EPS) / Previous EPS * 100
    
    Why: Positive EPS growth is the strongest predictor of stock appreciation.
    High value = Strong earnings growth = Better investment candidate.
    Most consistent cross-sectional predictor.
    """
    def __init__(self):
        super().__init__(name='earn_eps_momentum', description='EPS Growth Momentum (Q/Q)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_momentum(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if len(events) < 2:
                return pd.Series(np.nan, index=group.index)
            
            # Calculate QoQ growth
            events['eps_growth'] = events['eps_actual'].pct_change() * 100
            
            # Forward fill to daily
            return events['eps_growth'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)


@FactorRegistry.register()
class RecentPositiveRevisions(EarningsFactor):
    """
    Recent Positive Revisions: % of last 3 quarters with positive surprises.
    Formula: (# positive surprises in last 3Q) / 3 * 100
    
    Why: Consecutive positive surprises = management consistently raising bar.
    Highest signal of upward estimate revision trajectory.
    Range: 0% (all misses) to 100% (all beats).
    """
    def __init__(self):
        super().__init__(name='earn_recent_positive_revisions', description='Recent Positive Revisions %')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_revisions(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Calculate positive surprises (1 = beat, 0 = miss)
            events['positive'] = (events['surprise_pct'] > 0).astype(int)
            
            # Rolling average of last 3: % positive
            events['revision_pct'] = events['positive'].rolling(window=3, min_periods=1).mean() * 100
            
            # Forward fill to daily
            return events['revision_pct'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_revisions)


@FactorRegistry.register()
class EstimateAccuracyTrend(EarningsFactor):
    """
    Estimate Accuracy Trend: Inverse of recent surprise magnitude.
    Formula: 100 - |Average Surprise %| (last 4 quarters)
    
    Why: Lower surprises = Better estimates = More efficient guidance process.
    High value = More predictable earnings = Lower risk.
    Range: 0 (highly unpredictable) to 100 (perfectly predicted).
    """
    def __init__(self):
        super().__init__(name='earn_estimate_accuracy_trend', description='Estimate Accuracy Trend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_accuracy(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Absolute surprise magnitude
            events['surprise_abs'] = events['surprise_pct'].abs()
            
            # 4-quarter rolling average of surprise magnitude
            events['avg_surprise_mag'] = events['surprise_abs'].rolling(window=4, min_periods=1).mean()
            
            # Convert to accuracy (lower surprise = higher accuracy)
            # Clip at 100 to prevent negative values
            events['accuracy'] = (100 - events['avg_surprise_mag']).clip(lower=0, upper=100)
            
            # Forward fill to daily
            return events['accuracy'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_accuracy)


@FactorRegistry.register()
class EPSAcceleration(EarningsFactor):
    """
    EPS Acceleration: Change in growth rate (2-quarter momentum).
    Formula: (Current Growth Rate) - (Previous Growth Rate)
    
    Why: Accelerating earnings growth is powerful predictor of future appreciation.
    Positive = Growth is speeding up (bullish).
    Negative = Growth is slowing (bearish).
    Most predictive among earnings momentum factors.
    """
    def __init__(self):
        super().__init__(name='earn_eps_acceleration', description='EPS Acceleration (Growth Rate Change)')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_acceleration(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if len(events) < 3:
                return pd.Series(np.nan, index=group.index)
            
            # Calculate QoQ growth rates
            events['eps_growth'] = events['eps_actual'].pct_change() * 100
            
            # Acceleration = change in growth rate (2-quarter momentum)
            events['acceleration'] = events['eps_growth'].diff()
            
            # Forward fill to daily
            return events['acceleration'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_acceleration)
