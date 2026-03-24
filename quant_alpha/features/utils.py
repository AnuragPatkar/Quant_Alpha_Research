"""
Feature Engineering Utilities
=============================

Strictly vectorized statistical preprocessing routines mapping financial time-series targets.

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
    Calculates uniform Cross-Sectional Z-Scores sequentially bounding temporal distributions.

    Computes the standardized score for each asset relative to the cross-section
    at a specific point in time.

    Formula:
        $Z_{i,t} = \frac{X_{i,t} - \mu_t}{\sigma_t}$

    Robustness Checks:
    - Requires a minimum sample size ($N \ge 3$) per timestamp.
    - Neutralizes signals where cross-sectional variance is near zero ($\sigma < \epsilon$).
    - Fills invalid/noisy signals with $0.0$ (Neutral Alpha).
    
    Args:
        df (pd.DataFrame): Systemic target evaluation structures.
        columns (List[str]): Extracted target metric parameter fields.
        
    Returns:
        pd.DataFrame: Computed zero-mean explicitly normalized structures.
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    grouper = result.groupby('date')[numeric_cols]
    
    means = grouper.transform('mean')
    stds = grouper.transform('std')
    counts = grouper.transform('count')
    
    # Isolates bounds rigorously establishing threshold constraints strictly tracking N >= 3 dependencies 
    # and continuous variance epsilon parameters averting division by zero failures.
    valid_mask = (counts >= 3) & (stds > 1e-9)
    
    z_scores = (result[numeric_cols] - means) / stds.replace(0, 1)
    
    result[numeric_cols] = z_scores.where(valid_mask, 0.0)
    
    return result

def winsorize(df: pd.DataFrame, columns: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Applies optimized Cross-Sectional vector Winsorization bounds minimizing extreme limits.

    Limits extreme values to the specified percentiles to reduce the influence
    of outliers and microstructure noise dynamically per execution sequence.

    Args:
        df (pd.DataFrame): Systemic target matrix extraction targets.
        columns (List[str]): Exact dimensional targets bounding execution loops.
        lower (float): Target numerical explicit probability bounds defining lower floor limits. Defaults to 0.01.
        upper (float): Target numerical explicit probability bounds defining maximum scaling ceiling. Defaults to 0.99.
        
    Returns:
        pd.DataFrame: Validated sequence mappings natively stripped of extreme coordinates.
    """
    if df.empty or not columns:
        return df

    result = df.copy()
    
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    # Computes standard structural parameter derivations sequentially bounding single evaluation blocks
    quantiles = result.groupby('date')[numeric_cols].quantile([lower, upper]).unstack()
    
    if 'date' in result.columns:
        date_mapper = result['date']
    elif 'date' in result.index.names:
        date_mapper = result.index.get_level_values('date')
    else:
        logger.error("Winsorization failed: 'date' not found in columns or index.")
        raise KeyError("Winsorization requires 'date' column or index level.")

    for col in numeric_cols:
        # Binds O(1) mathematically exact lookup maps routing sequential vector parameters
        lower_limit = date_mapper.map(quantiles[col][lower])
        upper_limit = date_mapper.map(quantiles[col][upper])
        
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)

    return result

def rank_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Computes uniform Cross-Sectional continuous target mappings projecting array limits to Percentile Ranks.

    Transforms continuous features into a uniform distribution $[0, 1]$.
    This removes the influence of absolute magnitude and outliers, preserving
    only the ordinal relationship between assets.
    
    Args:
        df (pd.DataFrame): Input execution frame.
        columns (List[str]): Structural targets mathematically extracted.
        
    Returns:
        pd.DataFrame: Computed mappings strictly replacing discrete evaluation ranges.
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result

    result[numeric_cols] = df.groupby('date')[numeric_cols].rank(pct=True)
    
    result[numeric_cols] = result[numeric_cols].fillna(0.5)
    
    return result

def apply_feature_pipeline(df: pd.DataFrame, features: List[str], method: str = 'normalize') -> pd.DataFrame:
    """
    Orchestrates the global evaluation bounding state mappings mathematically.

    Execution Flow:
    1. **Winsorization**: Stabilize distribution tails.
    2. **Transformation**: Normalize ($Z$-Score) or Rank ($[0,1]$).
    
    Args:
        df (pd.DataFrame): Extracted systemic array states.
        features (List[str]): Defined keys determining array paths.
        method (str): Explicit mapping bounding configuration evaluation logic ('normalize' or 'rank'). Defaults to 'normalize'.
        
    Returns:
        pd.DataFrame: A mathematically scaled standardized coordinate dataframe.
    """
    if not features:
        return df

    logger.info(f"🛠️ Applying {method} pipeline to {len(features)} features...")
    
    data = winsorize(df, features)
    
    if method == 'normalize':
        data = cross_sectional_normalize(data, features)
    elif method == 'rank':
        data = rank_transform(data, features)
    else:
        logger.warning(f"⚠️ Unknown method '{method}'. Skipping normalization step.")
        
    return data