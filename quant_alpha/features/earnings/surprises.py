"""
Earnings Surprise Factors (Production Grade) - TOP 5 MOST IMPORTANT
Focus: Market reaction to EPS vs Estimates.

Adapts sparse earnings events to daily signals using change detection.

Factors (Ranked by Predictive Power):
1. SUE (Standardized Unexpected Earnings) -> Normalized by Price (Most Proven)
2. Surprise % -> Raw Percentage Surprise
3. Earnings Streak -> Consecutive quarters of beating estimates
4. Last Quarter Magnitude -> Absolute value of most recent surprise (Most Predictive)
5. Beat/Miss Momentum -> % of last 4 quarters beaten (Trend Signal)
"""

import pandas as pd
import numpy as np  
from ..base import EarningsFactor
from ..registry import FactorRegistry
from config.logging_config import logger

@FactorRegistry.register()
class EPSSurprise(EarningsFactor):
    """
    Standardized EPS Surprise (SUE).
    Formula: (Actual - Estimate) / Price
    """

    def __init__(self):
        super().__init__(name='eps_sue_price', description='Surprise standardized by Price')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Try to calculate from actual and estimate first
        if 'eps_actual' in df.columns and 'eps_estimate' in df.columns and 'close' in df.columns:
            surprise = df['eps_actual'] - df['eps_estimate']
            sue = surprise/(df['close'] + 1e-8)
            return sue.clip(-0.5, 0.5)
        
        # Fallback to surprise_pct if available (already includes estimate division)
        if 'surprise_pct' in df.columns and 'close' in df.columns:
            # surprise_pct is already (actual - estimate) / |estimate| * 100
            # Normalize by price (convert from % to absolute)
            sue = (df['surprise_pct'] / 100) * (df['close'] / 100 + 1e-8)
            return sue.clip(-0.5, 0.5)
        
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class EPSSurprisePercentage(EarningsFactor):
    """
    Raw EPS Surprise Percentage.
    Formula: (Actual - Estimate) / |Estimate|
    """

    def __init__(self):
        super().__init__(name='earn_surprise_pct', description='EPS Surprise Percentage')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' in df.columns:
            return df['surprise_pct']
        
        if 'eps_actual' in df.columns and 'eps_estimate' in df.columns:
            actual = df['eps_actual']
            estimate = df['eps_estimate']
            pct = (actual - estimate) / (estimate.abs() + 1e-8) * 100
            return pct.clip(-500, 500)
        
        return pd.Series(np.nan, index=df.index)

@FactorRegistry.register()
class ConsecutiveSurprise(EarningsFactor):
    """
    Count of consecutive positive surprises.
    Updates only on Earnings Announcement Dates.
    """
    def __init__(self):
        super().__init__(name='earn_streak', description='Consecutive Positive Surprises')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_streak(group):
            # Detect new announcement (Value changed from yesterday)
            # Note: This assumes ffill data. If sparse, use notna()
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            
            # Filter to event days only
            events = group.loc[is_new_event].copy()
            if events.empty:
                return pd.Series(0, index=group.index)

            # Vectorized streak calculation
            condition = events['surprise_pct'] > 0
            events['streak'] = condition.groupby((condition != condition.shift()).cumsum()).cumsum()
            
            return events['streak'].reindex(group.index).ffill().fillna(0)
        
        # Apply per ticker to prevent data bleeding
        return df.groupby('ticker', group_keys=False).apply(_calc_streak)
    
@FactorRegistry.register()
class LastQuarterMagnitude(EarningsFactor):
    """
    Magnitude of Last Quarter's Earnings Surprise.
    Formula: |Actual - Estimate| / |Estimate| from most recent earnings announcement
    
    Why: Surprises autocorrelate - last quarter's magnitude predicts this quarter.
    Higher = More volatile earnings (risk) or bigger surprise magnitude.
    Most recent info is most predictive.
    """
    def __init__(self):
        super().__init__(name='earn_last_quarter_magnitude', description='Last Quarter Surprise Magnitude')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_last_quarter(group):
            # Detect earnings announcement dates
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Get absolute value of surprise
            events['magnitude'] = events['surprise_pct'].abs()
            
            # Shift by 1 to get LAST quarter (lag the current surprise)
            events['last_quarter_mag'] = events['magnitude'].shift(1)
            
            # Forward fill to daily values
            return events['last_quarter_mag'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_last_quarter)

@FactorRegistry.register()
class BeatMissMomentum(EarningsFactor):
    """
    Beat/Miss Momentum: % of last 4 quarters beaten.
    Captures trend in earnings quality and guidance accuracy.
    Range: 0 (missed all 4) to 100 (beat all 4)
    Higher = Better guidance quality + positive momentum
    """
    def __init__(self):
        super().__init__(name='earn_beat_miss_momentum', description='% of Last 4Q Beaten')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_momentum(group):
            # Detect earnings announcement dates
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Calculate beat/miss (1 = beat, 0 = miss/beat by 0%)
            events['beat'] = (events['surprise_pct'] > 0).astype(int)
            
            # Rolling average of last 4: % of beats
            events['momentum'] = events['beat'].rolling(window=4, min_periods=1).mean() * 100
            
            # Forward fill to daily
            return events['momentum'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)