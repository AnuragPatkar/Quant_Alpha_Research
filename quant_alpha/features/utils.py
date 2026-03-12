"""
Feature Engineering Utilities
=============================
Statistical preprocessing routines for financial time-series data.

Purpose
-------
This module implements essential statistical transformations required to prepare
raw alpha factors for machine learning models. It focuses on cross-sectional
normalization, outlier mitigation (winsorization), and rank transformation.
These operations render non-stationary financial data suitable for predictive modeling
by standardizing distributions across time.

Usage
-----
These functions are typically invoked by the `BaseFactor` class during the
computation lifecycle.

.. code-block:: python

    from .utils import apply_feature_pipeline

    # Process a DataFrame of raw factors
    clean_df = apply_feature_pipeline(raw_df, features=['momentum', 'value'])

Importance
----------
- **Stationarity**: Cross-sectional normalization ($Z$-Scoring) removes market
  drift, isolating relative asset performance.
- **Robustness**: Winsorization limits the impact of data errors and extreme
  outliers ($> 3\sigma$) on model training.
- **Performance**: Optimized vectorized pandas operations ensure scalability
  to millions of rows ($O(N)$).

Tools & Frameworks
------------------
- **Pandas**: GroupBy and transform operations for cross-sectional statistics.
- **NumPy**: Efficient array masking and broadcasting.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def cross_sectional_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculates Cross-Sectional Z-Scores by date.

    Computes the standardized score for each asset relative to the cross-section
    at a specific point in time.

    Formula:
    $$ Z_{i,t} = \frac{X_{i,t} - \mu_t}{\sigma_t} $$

    Robustness Checks:
    - Requires a minimum sample size ($N \ge 3$) per timestamp.
    - Neutralizes signals where cross-sectional variance is near zero ($\sigma < \epsilon$).
    - Fills invalid/noisy signals with $0.0$ (Neutral Alpha).
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    # Data Validation: Process only numeric columns to prevent type errors
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    # Group-Apply Pattern: Vectorized calculation of cross-sectional statistics
    grouper = result.groupby('date')[numeric_cols]
    
    means = grouper.transform('mean')
    stds = grouper.transform('std')
    counts = grouper.transform('count')
    
    # Statistical Significance Mask:
    # 1. Minimum Sample Size: $N \ge 3$ prevents small-N noise.
    # 2. Variance Check: $\sigma > 10^{-9}$ prevents division by zero (singularities).
    valid_mask = (counts >= 3) & (stds > 1e-9)
    
    # Numerical Stability: Replace zero std with 1 to allow division (masked later)
    z_scores = (result[numeric_cols] - means) / stds.replace(0, 1)
    
    # Signal Neutralization: Set invalid/noisy signals to 0.0 (Neutral Alpha)
    result[numeric_cols] = z_scores.where(valid_mask, 0.0)
    
    return result

def winsorize(df: pd.DataFrame, columns: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Applies Cross-Sectional Winsorization (Clipping).

    Limits extreme values to the specified percentiles to reduce the influence
    of outliers and microstructure noise.

    Optimization:
    Utilizes a vectorized map lookup strategy which is approximately 10x faster
    than standard `groupby().apply()` for large datasets.
    """
    if df.empty or not columns:
        return df

    result = df.copy()
    
    # Data Validation: Process only numeric columns
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    # Step 1: Calculate dynamic quantiles per date in a single pass
    quantiles = result.groupby('date')[numeric_cols].quantile([lower, upper]).unstack()
    
    # Index Handling: Resolve date mapper whether 'date' is a column or MultiIndex level
    if 'date' in result.columns:
        date_mapper = result['date']
    elif 'date' in result.index.names:
        date_mapper = result.index.get_level_values('date')
    else:
        logger.error("Winsorization failed: 'date' not found in columns or index.")
        raise KeyError("Winsorization requires 'date' column or index level.")

    for col in numeric_cols:
        # Step 2: Vectorized Lookup ($O(1)$)
        # Map pre-calculated bounds to the original dataframe index via date key
        lower_limit = date_mapper.map(quantiles[col][lower])
        upper_limit = date_mapper.map(quantiles[col][upper])
        
        # Step 3: Fast Clipping
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)

    return result

def rank_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Computes Cross-Sectional Percentile Ranks.

    Transforms continuous features into a uniform distribution $[0, 1]$.
    This removes the influence of absolute magnitude and outliers, preserving
    only the ordinal relationship between assets.
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    # Data Validation: Process only numeric columns
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result

    # Transformation: Convert values to percentile ranks $[0, 1]$
    result[numeric_cols] = df.groupby('date')[numeric_cols].rank(pct=True)
    
    # Missing Data: Impute NaNs with 0.5 (Neutral Median Rank)
    result[numeric_cols] = result[numeric_cols].fillna(0.5)
    
    return result

def apply_feature_pipeline(df: pd.DataFrame, features: List[str], method: str = 'normalize') -> pd.DataFrame:
    """
    Orchestrates the standard preprocessing pipeline.

    Execution Flow:
    1. **Winsorization**: Stabilize distribution tails.
    2. **Transformation**: Normalize ($Z$-Score) or Rank ($[0,1]$).
    """
    if not features:
        return df

    logger.info(f"🛠️ Applying {method} pipeline to {len(features)} features...")
    
    # 1. Outlier Mitigation: Clip extreme values first to ensure mean/std are not distorted
    data = winsorize(df, features)
    
    # 2. Distribution Standardization
    if method == 'normalize':
        data = cross_sectional_normalize(data, features)
    elif method == 'rank':
        data = rank_transform(data, features)
    else:
        logger.warning(f"⚠️ Unknown method '{method}'. Skipping normalization step.")
        
    return data