"""
Earnings Revision & Momentum Factors
====================================
Quantitative signals derived from the trajectory of earnings updates and surprise history.

Purpose
-------
This module isolates the "second derivative" of corporate fundamentals by analyzing
changes in reported earnings (momentum/acceleration) and the persistence of analyst
surprise trends. Unlike static valuation metrics, these factors capture the
rate of change in fundamental performance, which is often a precursor to price action.

Usage
-----
Factors are automatically registered via the `FactorRegistry` decorator.
Primary execution occurs within the feature engineering pipeline:

.. code-block:: python

    from quant_alpha.features.registry import FactorRegistry
    
    registry = FactorRegistry()
    momentum_factor = registry.get('earn_eps_momentum')
    alpha_signals = momentum_factor.compute(ohlcv_earnings_df)

Importance
----------
- **Alpha Generation**: Earnings acceleration is a classic quantitative signal
  associated with the "Post-Earnings Announcement Drift" (PEAD) anomaly.
- **Regime Detection**: Distinguishes between companies with improving fundamentals
  vs. those in secular decline, even if their raw valuation ratios appear similar.
- **Signal Robustness**: Implements winsorization and robust denominators to
  prevent micro-cap volatility from skewing the cross-sectional distribution.

Tools & Frameworks
------------------
- **Pandas**: Group-apply patterns for ticker-level time-series analysis.
- **NumPy**: Vectorized arithmetic for efficient growth rate calculations.
- **FactorRegistry**: Dynamic discovery and instantiation of factor classes.
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
    Earnings Growth Momentum: Quarter-over-Quarter (QoQ) EPS Growth Rate.
    
    Captures the velocity of fundamental improvement. High momentum often precedes
    positive price action due to the PEAD anomaly.
    
    Formula:
    $$ Momentum_t = \frac{EPS_t - EPS_{t-1}}{\max(|EPS_{t-1}|, 0.01)} \times 100 $$
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
            
            # Numerical Stability: Apply epsilon floor (0.01) to denominator to 
            # prevent division-by-zero or explosion on near-zero earnings.
            denom = prev_eps.abs().clip(lower=0.01)
            events['eps_growth'] = (curr_eps - prev_eps) / denom * 100
            
            # Winsorization: Clip growth at +/- 500% to mitigate the impact 
            # of outliers on the cross-sectional distribution.
            return events['eps_growth'].clip(-500, 500).reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_momentum)


@FactorRegistry.register()
class RecentPositiveRevisions(EarningsFactor):
    """
    Recent Positive Revisions: Frequency of positive earnings surprises.
    
    Quantifies the consistency of "beating the street" over a rolling window.
    
    Formula:
    $$ Score_t = \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{I}(\text{Surprise}_{t-i} > 0) \times 100 $$
    """
    def __init__(self):
        super().__init__(name='earn_recent_positive_revisions', description='Recent Positive Revisions %')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc_revisions(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Data Integrity: Propagate NaN for missing surprises rather than imputing 0 (Miss),
            # preserving the distinction between 'unknown' and 'negative'.
            events['positive'] = np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float))
            events['revision_pct'] = events['positive'].rolling(window=3, min_periods=1).mean() * 100
            
            return events['revision_pct'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_revisions)


@FactorRegistry.register()
class EstimateAccuracyTrend(EarningsFactor):
    """
    Estimate Accuracy Trend: Inverse of the median absolute surprise magnitude.
    
    Measures the predictability of the company's earnings. A high score implies
    analysts can accurately model the business, reducing fundamental uncertainty risk.
    
    Formula:
    $$ Accuracy_t = \text{clip}(100 - \text{Median}(|\text{Surprise}\%|_{t-3...t}), 0, 100) $$
    """
    def __init__(self):
        super().__init__(name='earn_estimate_accuracy_trend', description='Estimate Accuracy Trend')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc_accuracy(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Robust Statistics: Median aggregation reduces sensitivity to singular 
            # outlier events (e.g., one-off COVID charges) compared to the mean.
            events['med_surprise_mag'] = events['surprise_pct'].abs().rolling(window=4, min_periods=2).median()
            
            # Normalization: Inverse relationship where higher score = lower surprise magnitude.
            events['accuracy'] = (100 - events['med_surprise_mag']).clip(lower=0, upper=100)
            
            return events['accuracy'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_accuracy)


@FactorRegistry.register()
class EPSAcceleration(EarningsFactor):
    """
    EPS Acceleration: The second derivative of earnings growth.
    
    Identifies inflection points where the rate of growth is increasing (convexity).
    
    Formula:
    $$ Accel_t = \Delta Growth_t = Growth_t - Growth_{t-1} $$
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
            
            # First Derivative: Calculate QoQ EPS growth rate.
            denom = events['eps_actual'].shift(1).abs().clip(lower=0.01)
            events['eps_growth'] = (events['eps_actual'] - events['eps_actual'].shift(1)) / denom * 100
            
            # Second Derivative: Discrete difference of growth rates (Acceleration).
            events['acceleration'] = events['eps_growth'].diff()
            
            # Outlier Management: Hard clip at +/- 200 to preserve normality in downstream linear models.
            return events['acceleration'].clip(-200, 200).reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc_acceleration)