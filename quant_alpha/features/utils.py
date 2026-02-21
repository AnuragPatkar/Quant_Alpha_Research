import pandas as pd
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def cross_sectional_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Institutional grade Z-score normalization by date.
    Vectorized logic: Group once, calculate stats, apply significance mask.
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    # Robustness: Filter for numeric columns only
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    # Vectorized Grouping
    grouper = result.groupby('date')[numeric_cols]
    
    means = grouper.transform('mean')
    stds = grouper.transform('std')
    counts = grouper.transform('count')
    
    # Statistical Significance: Avoid normalizing tiny groups or constant values
    # In Quant, normalizing <3 stocks is statistically noisy.
    valid_mask = (counts >= 3) & (stds > 1e-9)
    
    # Safe Z-Score calculation with divide-by-zero protection
    z_scores = (result[numeric_cols] - means) / stds.replace(0, 1)
    
    # Neutralize invalid/noisy signals to 0.0 (The Neutral Alpha position)
    result[numeric_cols] = z_scores.where(valid_mask, 0.0)
    
    return result

def winsorize(df: pd.DataFrame, columns: List[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Highly Optimized Cross-sectional winsorization.
    Vectorized map lookup is ~10x faster than groupby.transform(lambda).
    """
    if df.empty or not columns:
        return df

    result = df.copy()
    
    # Robustness: Filter for numeric columns only
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result
    
    # Step 1: Calculate quantiles in one vectorized pass
    quantiles = result.groupby('date')[numeric_cols].quantile([lower, upper]).unstack()
    
    # Handle 'date' mapping regardless of whether it's a column or index level
    if 'date' in result.columns:
        date_mapper = result['date']
    elif 'date' in result.index.names:
        date_mapper = result.index.get_level_values('date')
    else:
        logger.error("Winsorization failed: 'date' not found in columns or index.")
        raise KeyError("Winsorization requires 'date' column or index level.")

    for col in numeric_cols:
        # Step 2: Vectorized Lookup - Map pre-calculated bounds to the date key
        lower_limit = date_mapper.map(quantiles[col][lower])
        upper_limit = date_mapper.map(quantiles[col][upper])
        
        # Step 3: Fast clipping
        result[col] = result[col].clip(lower=lower_limit, upper=upper_limit)

    return result

def rank_transform(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Percentile Ranking [0, 1] by date.
    Removes outlier influence and normalizes feature scales for GBDT models.
    """
    if df.empty or not columns:
        return df
        
    result = df.copy()
    
    # Robustness: Filter for numeric columns only
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return result

    # pct=True scales ranks to the [0, 1] range
    result[numeric_cols] = df.groupby('date')[numeric_cols].rank(pct=True)
    
    # Handle NaNs: 0.5 is the neutral rank (the median)
    result[numeric_cols] = result[numeric_cols].fillna(0.5)
    
    return result

def apply_feature_pipeline(df: pd.DataFrame, features: List[str], method: str = 'normalize') -> pd.DataFrame:
    """
    Main entry point for feature processing.
    Order: Winsorize (stabilize) -> Normalize/Rank (standardize).
    """
    if not features:
        return df

    logger.info(f"🛠️ Applying {method} pipeline to {len(features)} features...")
    
    # 1. Clip extreme values first to ensure mean/std are not distorted
    data = winsorize(df, features)
    
    # 2. Standardize cross-sectionally
    if method == 'normalize':
        data = cross_sectional_normalize(data, features)
    elif method == 'rank':
        data = rank_transform(data, features)
    else:
        logger.warning(f"⚠️ Unknown method '{method}'. Skipping normalization step.")
        
    return data