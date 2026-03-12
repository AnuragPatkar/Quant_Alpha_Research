"""
Feature Preprocessing & Normalization Engine
============================================
High-performance transformation kernels for signal conditioning.

Purpose
-------
This module implements stateful scaling and outlier mitigation strategies essential
for training stable machine learning models. It leverages **Numba JIT compilation**
to accelerate element-wise operations that are typically bottlenecks in Python.

Usage
-----
.. code-block:: python

    # 1. Winsorization (Outlier Clipping)
    w_scaler = WinsorisationScaler(clip_pct=0.01)
    w_scaler.fit(train_df, features=['rsi', 'vol'])
    clean_df = w_scaler.transform(test_df, features=['rsi', 'vol'])

    # 2. Sector-Neutral Z-Scoring
    z_scaler = SectorNeutralScaler(sector_col='sector')
    norm_df = z_scaler.transform(clean_df, features=['rsi', 'vol'])

Importance
----------
-   **Gradient Stability**: Winsorization prevents extreme outliers ($>3\sigma$)
    from exploding gradients during backpropagation in neural networks or boosting.
-   **Regime Alignment**: Sector-neutral scaling isolates idiosyncratic alpha
    from systematic sector bets, ensuring the model learns stock-specific drivers.
-   **Computational Efficiency**: The `winsorize_clip_nb` kernel executes in parallel
    (SIMD-friendly), reducing latency for large datasets ($N > 10^7$).

Tools & Frameworks
------------------
-   **Numba**: LLVM-based JIT compilation for parallelized array operations.
-   **Pandas/NumPy**: Vectorized group-by and broadcasting.
"""

import numpy as np
import pandas as pd
from numba import njit, prange

# ==================== NUMBA KERNELS ====================

@njit(parallel=True, cache=True)
def winsorize_clip_nb(data, lower, upper):
    """
    Numba-accelerated parallel clipping kernel.
    
    Executes element-wise clamping:
    .. math:: x'_{ij} = \\max(L_{ij}, \\min(x_{ij}, U_{ij}))
    
    Args:
        data (np.ndarray): Input feature matrix $(N \\times F)$.
        lower (np.ndarray): Lower bound matrix $(N \\times F)$, aligned to rows.
        upper (np.ndarray): Upper bound matrix $(N \\times F)$, aligned to rows.
        
    Returns:
        np.ndarray: Clipped feature matrix.
    """
    n_rows, n_cols = data.shape
    out = np.empty_like(data)
    for i in prange(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            # Conditional Branching: Explicit min/max logic is often faster 
            # than np.clip within Numba loops due to reduced function call overhead.
            out[i, j] = lower[i, j] if v < lower[i, j] else (upper[i, j] if v > upper[i, j] else v)
    return out

# ==================== SCALERS ====================

class WinsorisationScaler:
    """
    Stateful Outlier Mitigation via Historical Quantiles.
    
    Learns the distribution bounds $[q_{low}, q_{high}]$ per timestamp during fitting,
    and applies these bounds to future data using a "Last Observation Carried Forward" 
    (LOCF) logic for regime stability.
    """
    def __init__(self, clip_pct: float = 0.01):
        self.clip_pct  = clip_pct
        self._lower    = None
        self._upper    = None
        self._date_arr = None

    def fit(self, df: pd.DataFrame, features: list) -> "WinsorisationScaler":
        """
        Computes cross-sectional quantiles per date.
        
        Args:
            df (pd.DataFrame): Training data containing 'date' and feature columns.
            features (list): List of column names to fit.
        """
        grp            = df.groupby("date")[features]
        self._lower    = grp.quantile(self.clip_pct)
        self._upper    = grp.quantile(1.0 - self.clip_pct)
        self._date_arr = self._lower.index.values
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Applies winsorization using the nearest historical bounds.
        
        Uses `searchsorted` to map inference dates to the most recent fit date,
        preventing look-ahead bias while handling gaps in trading calendars.
        """
        dates    = pd.to_datetime(df["date"]).values
        # Temporal Alignment: Map rows to the nearest previous fit date (Forward Fill)
        idx      = np.searchsorted(self._date_arr, dates, side="right") - 1
        idx      = np.clip(idx, 0, len(self._date_arr) - 1)
        mapped   = self._date_arr[idx]
        
        # Data Alignment: Broadcast bounds to the shape of the input data
        low_arr  = self._lower.loc[mapped, features].values.astype(np.float64)
        up_arr   = self._upper.loc[mapped, features].values.astype(np.float64)
        data_arr = df[features].values.astype(np.float64)
        
        clipped  = winsorize_clip_nb(data_arr, low_arr, up_arr)
        out      = df.copy()
        out[features] = clipped
        return out

class SectorNeutralScaler:
    """
    Cross-Sectional Standardization Engine.
    
    Standardizes features by computing the Z-Score relative to the grouping
    (Date, Sector) distribution at that specific timestep.
    
    .. math:: z_{i,t} = \\frac{x_{i,t} - \\mu_{S,t}}{\\sigma_{S,t}}
    
    Note: This is stateless regarding time (calculates stats on the fly), 
    but stateful regarding schema (sector column).
    """
    def __init__(self, sector_col: str = "sector"):
        self.sector_col  = sector_col
        self._features   = []

    def fit(self, df: pd.DataFrame, features: list) -> "SectorNeutralScaler":
        """Validates schema and registers feature set."""
        self._features    = features
        self._has_sector  = self.sector_col in df.columns
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Computes and applies Cross-Sectional Z-Score.
        
        Includes $\\epsilon$ handling to ensure numerical stability in low-variance
        regimes.
        """
        out = df.copy()
        if self._has_sector and self.sector_col in df.columns:
            grp = df.groupby(["date", self.sector_col])[features]
        else:
            # Fallback: Market-wide standardization if sector data is missing
            grp = df.groupby("date")[features]
            
        means = grp.transform("mean")
        # Stability: Fill NaN stds and replace zeros with epsilon to avoid DivByZero
        stds  = grp.transform("std").fillna(1e-8).replace(0, 1e-8)
        
        out[features] = (df[features].values - means.values) / (stds.values + 1e-8)
        return out

    def inference_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Alias for transform, emphasizing that stats are derived from the input batch."""
        return self.transform(df, features)