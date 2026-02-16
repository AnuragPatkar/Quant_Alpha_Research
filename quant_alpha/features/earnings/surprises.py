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
from .utils import detect_earnings_events, get_events_with_surprise

@FactorRegistry.register()
class EPSSurprise(EarningsFactor):
    """
    Standardized EPS Surprise (SUE).
    Formula: (Actual - Estimate) / Price
    """
    def __init__(self):
        super().__init__(name='eps_sue_price', description='Surprise standardized by Price')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'eps_actual' not in df.columns or 'eps_estimate' not in df.columns or 'close' not in df.columns:
            return pd.Series(np.nan, index=df.index)
            
        def _calc_sue(group):
            is_new_event = detect_earnings_events(group)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
                
            # Calculate SUE at the time of the event
            # Use ffilled close to handle missing price on exact report date (e.g. weekend/holiday reports)
            filled_close = group['close'].ffill()
            events_close = filled_close.loc[events.index]
            valid_close = events_close.where(events_close > 0, np.nan)
            events['sue'] = (events['eps_actual'] - events['eps_estimate']) / valid_close
            
            return events['sue'].reindex(group.index).ffill().clip(-0.5, 0.5)
        
        return df.groupby('ticker', group_keys=False).apply(_calc_sue)


@FactorRegistry.register()
class EPSSurprisePercentage(EarningsFactor):
    """
    Raw EPS Surprise Percentage.
    Formula: (Actual - Estimate) / |Estimate|
    """
    def __init__(self):
        super().__init__(name='earn_surprise_pct', description='EPS Surprise Percentage')

    def compute(self, df: pd.DataFrame) -> pd.Series:
        # If surprise_pct exists, we just need to ensure it's ffilled
        # But to be consistent, we recalculate or ffill based on events
        cols_needed = ['eps_actual', 'eps_estimate']
        if not all(c in df.columns for c in cols_needed) and 'surprise_pct' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_pct(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)
            
            return events['surprise_pct'].reindex(group.index).ffill().clip(-500, 500)

        return df.groupby('ticker', group_keys=False).apply(_calc_pct)


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
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(0, index=group.index)

            condition = events['surprise_pct'] > 0
            # Group by consecutive runs
            run_ids = (condition != condition.shift()).cumsum()
            # Cumsum within each run, then mask out the False runs (misses should be 0)
            events['streak'] = condition.groupby(run_ids).cumsum()
            events['streak'] = events['streak'].where(condition, 0)
            
            return events['streak'].reindex(group.index).ffill().fillna(0)
        
        return df.groupby('ticker', group_keys=False).apply(_calc_streak)
    

@FactorRegistry.register()
class LastQuarterMagnitude(EarningsFactor):
    """
    Magnitude of Last Quarter's Earnings Surprise.
    Formula: |Actual - Estimate| / |Estimate| from most recent earnings announcement
    """
    def __init__(self):
        super().__init__(name='earn_last_quarter_magnitude', description='Last Quarter Surprise Magnitude')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_last_quarter(group):
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            events['magnitude'] = events['surprise_pct'].abs()
            events['last_quarter_mag'] = events['magnitude']
            
            return events['last_quarter_mag'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_last_quarter)


@FactorRegistry.register()
class BeatMissMomentum(EarningsFactor):
    """
    Beat/Miss Momentum: % of last 4 quarters beaten.
    Range: 0 (missed all 4) to 100 (beat all 4)
    """
    def __init__(self):
        super().__init__(name='earn_beat_miss_momentum', description='% of Last 4Q Beaten')
        
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_momentum(group):
            events = get_events_with_surprise(group)
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Handle NaNs: If surprise is missing, don't count as miss (0), keep as NaN
            events['beat'] = np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float))
            
            # FIX: min_periods=2 prevents wild 0/100 swings on a stock's very first earnings report
            events['momentum'] = events['beat'].rolling(window=4, min_periods=2).mean() * 100
            
            return events['momentum'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)