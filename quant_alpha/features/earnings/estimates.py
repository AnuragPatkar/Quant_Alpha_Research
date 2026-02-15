"""
Earnings Estimate Quality Factors (Production Grade) - TOP 4 MOST IMPORTANT
Focus: Analyst estimate reliability, consensus strength, and forward expectations.

Factors (Ranked by Predictive Power):
1. Consensus Strength -> How well estimates predict actuals (Most Important)
2. Estimate Surprise Consistency -> Pattern consistency in surprises
3. Positive Estimate Confidence -> % quarters beating estimates (Uptrend Signal)
4. Estimate Guidance Quality -> Inverse of estimate errors
"""

import pandas as pd
import numpy as np
from ..base import EarningsFactor
from ..registry import FactorRegistry
from config.logging_config import logger


@FactorRegistry.register()
class ConsensusStrength(EarningsFactor):
    """
    Consensus Strength: 100 - Average % error in estimates (4-quarter basis).
    Formula: 100 - |Average %(Actual-Estimate)|
    
    Why: Low estimate errors = Strong analyst consensus.
    Higher value = Better predictability = More reliable analyst forecasts.
    Range: 0 (highly inaccurate) to 100 (perfect predictions).
    Most predictive of analyst quality.
    """
    def __init__(self):
        super().__init__(name='est_consensus_strength', description='Analyst Consensus Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_consensus(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Error magnitude = absolute surprise %
            events['error_pct'] = events['surprise_pct'].abs()
            
            # 4-quarter rolling average of error
            events['avg_error'] = events['error_pct'].rolling(window=4, min_periods=1).mean()
            
            # Consensus Strength = 100 - average error (capped at 100)
            events['consensus'] = (100 - events['avg_error']).clip(lower=0, upper=100)
            
            # Forward fill to daily
            return events['consensus'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_consensus)


@FactorRegistry.register()
class EstimateSurpriseConsistency(EarningsFactor):
    """
    Estimate Surprise Consistency: Inverse of surprise volatility.
    Formula: 100 - StdDev(Surprise %) over last 4 quarters
    
    Why: Low surprise variance = Estimates consistently accurate.
    High consistency = Predictable earnings = Lower execution risk.
    More consistent winners tend to outperform.
    Range: 0 (highly volatile) to 100 (perfectly consistent).
    """
    def __init__(self):
        super().__init__(name='est_surprise_consistency', description='Estimate Surprise Consistency')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_consistency(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if len(events) < 2:
                return pd.Series(np.nan, index=group.index)
            
            # Surprise volatility (std deviation)
            events['surprise_vol'] = events['surprise_pct'].rolling(window=4, min_periods=1).std()
            
            # Consistency = 100 - volatility (normalized to 0-100)
            # Higher std deviation = lower consistency
            events['consistency'] = (100 - (events['surprise_vol'] * 5)).clip(lower=0, upper=100)
            
            # Forward fill to daily
            return events['consistency'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_consistency)


@FactorRegistry.register()
class PositiveEstimateConfidence(EarningsFactor):
    """
    Positive Estimate Confidence: % of recent quarters beating estimates.
    Formula: (# positive surprises / # quarters) * 100 (last 6 quarters)
    
    Why: Companies that consistently beat have strong execution.
    Market attributes value to consistent beat records.
    Captures "surprise momentum" or "execution quality".
    Range: 0% (all misses) to 100% (all beats).
    """
    def __init__(self):
        super().__init__(name='est_positive_confidence', description='Positive Estimate Confidence')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_confidence(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Beat indicator (1 = beat, 0 = miss)
            events['beat'] = (events['surprise_pct'] > 0).astype(int)
            
            # 6-quarter rolling average of beats
            events['confidence'] = events['beat'].rolling(window=6, min_periods=1).mean() * 100
            
            # Forward fill to daily
            return events['confidence'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_confidence)


@FactorRegistry.register()
class EstimateGuidanceQuality(EarningsFactor):
    """
    Estimate Guidance Quality: Accuracy of forward guidance.
    Formula: 1 / (1 + Average Recent Error) - Inverse relationship to error magnitude.
    
    Why: Lower estimate errors = Management provides clear, achievable guidance.
    Companies with strong guidance quality tend to outperform peers.
    Captures management credibility and execution.
    Range: 0 (poor guidance) to 1.0 (perfect guidance).
    """
    def __init__(self):
        super().__init__(name='est_guidance_quality', description='Estimate Guidance Quality')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        if 'surprise_pct' not in df.columns or 'eps_actual' not in df.columns:
            return pd.Series(np.nan, index=df.index)
        
        def _calc_guidance_quality(group):
            # Detect earnings events
            is_new_event = group['eps_actual'] != group['eps_actual'].shift(1)
            events = group.loc[is_new_event].copy()
            
            if events.empty:
                return pd.Series(np.nan, index=group.index)
            
            # Recent error magnitude (absolute surprise)
            events['error'] = (events['surprise_pct'].abs() / 100) + 1e-8  # Normalize
            
            # 4-quarter rolling average of error
            events['avg_error'] = events['error'].rolling(window=4, min_periods=1).mean()
            
            # Guidance quality = 1 / (1 + avg_error)
            # Higher quality = smaller errors = value closer to 1.0
            events['guidance_quality'] = 1.0 / (1.0 + events['avg_error'])
            
            # Forward fill to daily
            return events['guidance_quality'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_guidance_quality)
