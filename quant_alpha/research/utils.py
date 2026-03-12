"""
Research Utilities & Statistical Preprocessing
==============================================
Shared helper functions for data alignment, visualization, and signal transformation.

Purpose
-------
This module provides a unified toolkit for the Research pipeline, handling:
1.  **Data Ingestion**: Standardizing raw DataFrames into aligned MultiIndex formats.
2.  **Statistical Transformation**: Robust outlier mitigation (Winsorization) and normalization (Z-Scoring).
3.  **Visualization**: Standardized plotting routines for factor distributions.

Usage
-----
Intended for internal use by `FactorAnalyzer`, `AlphaDecayAnalyzer`, and other research modules.

.. code-block:: python

    # Data Alignment
    clean_df = prepare_factor_data(raw_df, factor_col='rsi', forward_return_col='ret_5d')

    # Statistical Preprocessing
    clean_series = winsorize_series(clean_df['rsi'], limits=(0.01, 0.01))
    z_score = standardize_series(clean_series)

Importance
----------
-   **Data Integrity**: Enforces strict (Date, Ticker) indexing to prevent alignment errors during cross-sectional analysis.
-   **Numerical Stability**: `standardize_series` includes epsilon handling to prevent DivisionByZero errors in low-variance signals.
-   **Visual Consistency**: Centralized plotting logic ensures uniform reporting standards across notebooks.

Tools & Frameworks
------------------
-   **Pandas**: Time-series alignment and indexing.
-   **SciPy (mstats)**: Winsorization for robust statistics.
-   **Seaborn/Matplotlib**: Distribution visualization (KDE + Histogram).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def prepare_factor_data(data: pd.DataFrame, factor_col: str, forward_return_col: str = 'raw_ret_5d'):
    """
    Ingests and aligns raw data for Factor Analysis.

    Constructs a canonical MultiIndex (Date, Ticker) DataFrame, enforcing strict
    data integrity by removing rows with missing factor or target values.

    Args:
        data (pd.DataFrame): Raw input data.
        factor_col (str): Column name of the alpha signal.
        forward_return_col (str): Column name of the target return.

    Returns:
        pd.DataFrame: Cleaned data indexed by (date, ticker).
    """
    required = ['date', 'ticker', factor_col, forward_return_col]
    
    # Schema Validation
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
        
    # State Isolation: Defensive copy to prevent mutation of source dataframe
    df = data[required].copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Data Cleaning: Drop rows where critical data is NaN
    original_len = len(df)
    df = df.dropna()
    dropped = original_len - len(df)
    if dropped > 0:
        logger.debug(f"Dropped {dropped} rows with NaNs in factor or target.")
        
    # Indexing: Enforce MultiIndex structure for efficient cross-sectional grouping
    return df.set_index(['date', 'ticker']).sort_index()

def get_forward_returns_columns(data: pd.DataFrame):
    """
    Heuristically identifies columns representing forward-looking returns.
    Used for auto-discovery of target variables in analysis workflows.
    """
    return [c for c in data.columns if 'ret' in c and ('fwd' in c or 'future' in c or 'raw' in c)]

def plot_distribution(series: pd.Series, title: str = "Distribution", save_path=None):
    """
    Visualizes the empirical distribution of a factor.
    
    Combines a histogram with Kernel Density Estimation (KDE) to assess
    normality, skewness, and fat tails.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=50)
    plt.title(title)
    plt.axvline(series.mean(), color='r', linestyle='--', label=f'Mean: {series.mean():.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def winsorize_series(s: pd.Series, limits=(0.01, 0.01)):
    """
    Applies Winsorization to mitigate the impact of outliers.

    Clips values at the specified percentiles (default 1% top/bottom).
    Useful for robustifying moments (mean/std) against extreme events.

    Args:
        s (pd.Series): Input data.
        limits (Tuple[float, float]): Lower and upper percentile cuts (e.g., 0.01).
    """
    from scipy.stats.mstats import winsorize
    return pd.Series(winsorize(s, limits=limits), index=s.index)

def standardize_series(s: pd.Series):
    """
    Performs Z-Score Normalization (Standardization).

    .. math:: z = \\frac{x - \\mu}{\\sigma + \\epsilon}

    Includes epsilon ($10^{-8}$) to prevent division-by-zero on constant signals.
    """
    return (s - s.mean()) / (s.std() + 1e-8)
