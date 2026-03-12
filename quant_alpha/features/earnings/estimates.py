"""
Earnings Estimate Quality Factors
=================================
Quantitative factors assessing the reliability and predictive power of analyst estimates.

Purpose
-------
This module constructs alpha factors derived from the relationship between analyst
consensus estimates and actual reported earnings. It focuses on second-order
properties of earnings surprises—such as consistency, volatility, and confidence—
rather than the raw surprise magnitude itself. These metrics help distinguish
high-quality, sustainable earnings beats from noisy, one-off events.

Usage
-----
These factors are automatically registered with the `FactorRegistry` upon import.
They operate on standardized earnings data frames containing `eps_actual`,
`eps_estimate`, and `surprise_pct`.

.. code-block:: python

    registry = FactorRegistry()
    factor = registry.get('est_consensus_strength')
    signals = factor.compute(market_data_df)

Importance
----------
- **Signal Quality**: Filters for companies with predictable earnings patterns,
  improving the Sharpe Ratio of earnings-based strategies.
- **Risk Management**: Identifies firms with high estimate variance ($high \sigma$),
  which often exhibit excessive post-earnings drift volatility.
- **Robustness**: Utilizes median-based aggregation and decay functions to
  minimize the impact of outliers in historical data.

Tools & Frameworks
------------------
- **Pandas**: Efficient time-series grouping and rolling window calculations.
- **NumPy**: Vectorized numerical operations for score normalization.
- **FactorRegistry**: Integration with the central feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from ..base import EarningsFactor
from ..registry import FactorRegistry
from .utils import get_events_with_surprise

@FactorRegistry.register()
class ConsensusStrength(EarningsFactor):
    """
    Consensus Strength: Inverse of the median absolute estimation error.
    
    Measures how accurately analysts have historically predicted the company's EPS.
    Higher scores indicate a "tight" consensus where analysts model the company well.
    
    Formula:
    $$ Score_t = \max(0, 100 - \text{Median}(|\text{Surprise}\%|_{t-3...t})) $$
    """
    def __init__(self):
        super().__init__(name='est_consensus_strength', description='Analyst Consensus Strength')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Robustness: Median aggregation over rolling window mitigates outlier impact.
            events['err'] = events['surprise_pct'].abs()
            events['score'] = (100 - events['err'].rolling(4, min_periods=2).median()).clip(0, 100)
            return events['score'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class EstimateSurpriseConsistency(EarningsFactor):
    """
    Estimate Surprise Consistency: Inverse of earnings surprise volatility.
    
    Penalizes companies with erratic earnings history (volatile surprises).
    Uses a non-linear decay function to normalize volatility into a [0, 1] score.
    
    Formula:
    $$ Consistency_t = \frac{1}{1 + \frac{\sigma(\text{Surprise}\%)_{t-3...t}}{10}} $$
    """
    def __init__(self):
        super().__init__(name='est_surprise_consistency', description='Estimate Surprise Consistency')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if len(events) < 3: return pd.Series(np.nan, index=group.index)

            # Calculate rolling standard deviation ($\sigma$) of surprise percentages
            vols = events['surprise_pct'].rolling(4, min_periods=3).std()
            
            # Normalization: Decay function. $\sigma=0 \to 1.0$, $\sigma=20 \to 0.33$.
            events['consistency'] = 1.0 / (1.0 + (vols / 10.0))
            return events['consistency'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class PositiveEstimateConfidence(EarningsFactor):
    """
    Positive Estimate Confidence: Probability of beating estimates based on recent history.
    
    Quantifies the "beat rate" over a rolling window.
    
    Formula:
    $$ Conf_t = \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{I}(\text{Surprise}_{t-i} > 0) \times 100 $$
    """
    def __init__(self):
        super().__init__(name='est_positive_confidence', description='Positive Estimate Confidence')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Handling Missing Data: Propagate NaN for missing surprises rather than imputing 0 (Miss).
            beats = pd.Series(np.where(events['surprise_pct'].isna(), np.nan, (events['surprise_pct'] > 0).astype(float)), index=events.index)
            
            # Statistical Significance: Require min 3 observations within 6-quarter window.
            events['conf'] = beats.rolling(window=6, min_periods=3).mean() * 100
            return events['conf'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)

@FactorRegistry.register()
class EstimateGuidanceQuality(EarningsFactor):
    """
    Guidance Quality: Inverse of the mean relative estimation error.
    
    Evaluates the accuracy of analyst expectations. High quality implies analysts
    have a clear view of the company's trajectory (low relative error).
    
    Formula:
    $$ Quality_t = \frac{1}{1 + \text{Mean}(|\frac{\text{Surprise}\%}{100}|_{t-3...t})} $$
    """
    def __init__(self):
        super().__init__(name='est_guidance_quality', description='Estimate Guidance Quality')
    
    def compute(self, df: pd.DataFrame) -> pd.Series:
        # Data Validation: Ensure necessary columns exist for computation
        has_surprise = 'surprise_pct' in df.columns
        has_components = 'eps_actual' in df.columns and 'eps_estimate' in df.columns
        
        if not (has_surprise or has_components):
            return pd.Series(np.nan, index=df.index)
        
        def _calc(group):
            events = get_events_with_surprise(group)
            if events.empty: return pd.Series(np.nan, index=group.index)

            # Normalization: Convert percentage surprise to decimal scale (e.g., 5% -> 0.05).
            rel_error = events['surprise_pct'].abs() / 100.0
            avg_err = rel_error.rolling(4, min_periods=2).mean()
            
            events['quality'] = 1.0 / (1.0 + avg_err)
            return events['quality'].reindex(group.index).ffill()
        
        return df.groupby('ticker', group_keys=False).apply(_calc)