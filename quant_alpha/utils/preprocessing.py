"""
preprocessing.py
================
Shared preprocessing logic for training, inference, and hyperparameter optimization.
Contains Numba-accelerated kernels and stateful Scaler classes.
"""

import numpy as np
import pandas as pd
from numba import njit, prange

# ==============================================================================
# NUMBA KERNELS
# ==============================================================================
@njit(parallel=True, cache=True)
def winsorize_clip_nb(data, lower, upper):
    """Parallel element-wise clip to [lower, upper]."""
    n_rows, n_cols = data.shape
    out = np.empty_like(data)
    for i in prange(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            out[i, j] = lower[i, j] if v < lower[i, j] else (upper[i, j] if v > upper[i, j] else v)
    return out

# ==============================================================================
# SCALERS
# ==============================================================================
class WinsorisationScaler:
    """
    Fits per-date [q_lo, q_hi] bounds. Transform uses numpy searchsorted.
    """
    def __init__(self, clip_pct: float = 0.01):
        self.clip_pct  = clip_pct
        self._lower    = None
        self._upper    = None
        self._date_arr = None

    def fit(self, df: pd.DataFrame, features: list) -> "WinsorisationScaler":
        grp            = df.groupby("date")[features]
        self._lower    = grp.quantile(self.clip_pct)
        self._upper    = grp.quantile(1.0 - self.clip_pct)
        self._date_arr = self._lower.index.values
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        dates    = pd.to_datetime(df["date"]).values
        # Map rows to nearest fit date
        idx      = np.searchsorted(self._date_arr, dates, side="right") - 1
        idx      = np.clip(idx, 0, len(self._date_arr) - 1)
        mapped   = self._date_arr[idx]
        
        low_arr  = self._lower.loc[mapped, features].values.astype(np.float64)
        up_arr   = self._upper.loc[mapped, features].values.astype(np.float64)
        data_arr = df[features].values.astype(np.float64)
        
        clipped  = winsorize_clip_nb(data_arr, low_arr, up_arr)
        out      = df.copy()
        out[features] = clipped
        return out

class SectorNeutralScaler:
    """
    Sector-neutral z-score scaler.
    """
    def __init__(self, sector_col: str = "sector"):
        self.sector_col  = sector_col
        self._features   = []

    def fit(self, df: pd.DataFrame, features: list) -> "SectorNeutralScaler":
        self._features    = features
        self._has_sector  = self.sector_col in df.columns
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Sector-neutral z-score. Fits stats from df itself."""
        out = df.copy()
        if self._has_sector and self.sector_col in df.columns:
            grp = df.groupby(["date", self.sector_col])[features]
        else:
            grp = df.groupby("date")[features]
            
        means = grp.transform("mean")
        stds  = grp.transform("std").fillna(1e-8).replace(0, 1e-8)
        
        out[features] = (df[features].values - means.values) / (stds.values + 1e-8)
        return out

    def inference_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Alias for transform (cross-sectional z-score on input data)."""
        return self.transform(df, features)