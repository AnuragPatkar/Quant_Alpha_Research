"""
Feature Preprocessing & Normalization Engine
============================================

Provides high-performance continuous transformation kernels for dynamic signal conditioning.

Purpose
-------
This module constructs stateful boundary scalers implementing strict Point-in-Time (PiT) 
feature normalization and winsorization logic. By mathematically locking distribution 
percentiles and neutralizing relative sector drifts, it strictly eliminates look-ahead 
artifacts during dynamic machine-learning cross-validation.

Role in Quantitative Workflow
-----------------------------
Serves as the ultimate mathematical barrier between raw computed alpha tensors and the 
learning ensembles. Isolates pure idiosyncratic alpha by subtracting systemic group 
means and bounding erratic tail anomalies without shrinking historical predictive signals.

Mathematical Dependencies
-------------------------
- **Numba**: Deploys JIT-compiled $O(1)$ loop optimizations replacing native iteration bounds.
- **NumPy/Pandas**: Manages timezone-naive temporal vector mapping and cross-sectional indices.
"""

import numpy as np
import pandas as pd
from numba import njit, prange


def _default_clip_pct() -> float:
    """
    Extracts the centralized lower winsorize quantile parameter.
    
    Forces adherence to a single architectural source of truth via settings injection, 
    gracefully reverting to a standard $1\%$ threshold if the configuration map is inaccessible.
    
    Returns:
        float: The targeted numerical probability boundary (e.g., 0.01).
    """
    try:
        from config.settings import config
        lo, _hi = config.WINSORIZE_QUANTILES
        return float(lo)
    except Exception:
        return 0.01


@njit(parallel=True, cache=True)
def winsorize_clip_nb(data, lower, upper):
    """
    Hardware-accelerated parallel matrix clipping kernel.

    Args:
        data  (np.ndarray): Input feature matrix (N × F).
        lower (np.ndarray): Explicit lower percentile boundary matrix (N × F).
        upper (np.ndarray): Explicit upper percentile boundary matrix (N × F).

    Returns:
        np.ndarray: Clipped feature matrix.
    """
    n_rows, n_cols = data.shape
    out = np.empty_like(data)
    for i in prange(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            out[i, j] = (
                lower[i, j] if v < lower[i, j]
                else (upper[i, j] if v > upper[i, j] else v)
            )
    return out


class WinsorisationScaler:
    """
    Stateful Outlier Mitigation via Historical Quantiles.

    Evaluates the continuous distribution bounds strictly indexed by timestamp during 
    the initial fitting phase. Applies the nearest historical bounding parameters to 
    subsequent out-of-sample data distributions using "Last Observation Carried Forward" 
    (LOCF) mechanics to enforce stability. Time vectors are structurally converted to 
    tz-naive `int64` nanoseconds to strictly guarantee cross-environment synchronization.
    """

    def __init__(self, clip_pct: float | None = None):
        """
        Initializes the dynamic outlier bounding architecture.
        
        Args:
            clip_pct (Optional[float]): The structural trim limit parameter. 
                Defaults to None, dynamically triggering systemic config extraction.
        """
        self.clip_pct  = clip_pct if clip_pct is not None else _default_clip_pct()
        self._lower    = None
        self._upper    = None
        self._date_arr = None  # int64 nanosecond timestamps (tz-naive)

    def fit(self, df: pd.DataFrame, features: list) -> "WinsorisationScaler":
        """
        Computes specific discrete cross-sectional quantiles indexed identically per coordinate date.
        
        Args:
            df (pd.DataFrame): Training temporal panel storing asset prices and boundaries.
            features (list): String definitions mapping explicit feature targets to extract.
            
        Returns:
            WinsorisationScaler: Returns self for continuous pipeline object chaining.
        """
        grp         = df.groupby("date")[features]
        self._lower = grp.quantile(self.clip_pct)
        self._upper = grp.quantile(1.0 - self.clip_pct)

        # Transposes reference objects into standard tz-naive boundary integers
        idx = pd.DatetimeIndex(self._lower.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        self._date_arr = idx.astype(np.int64)
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Executes structural array winsorization via LOCF lookup extrapolation.
        
        Aligns incoming continuous vectors against the nearest established lookback 
        threshold, bypassing computationally heavy grouped distributions out-of-sample.
        
        Args:
            df (pd.DataFrame): Out-of-Sample temporal panel to evaluate.
            features (list): String definitions mapping feature arrays for mutation.
            
        Returns:
            pd.DataFrame: Scaled and stabilized array mappings.
        """
        # Resolves timezone mapping overlaps dynamically via explicit stripping
        raw_dates = pd.to_datetime(df["date"])
        if raw_dates.dt.tz is not None:
            raw_dates = raw_dates.dt.tz_localize(None)
        dates_i64 = raw_dates.values.astype(np.int64)

        # Leverages standard binary search mapping vectors to their trailing boundary index
        idx    = np.searchsorted(self._date_arr, dates_i64, side="right") - 1
        idx    = np.clip(idx, 0, len(self._date_arr) - 1)

        # Re-engineers exact discrete lookup timestamps supporting DataFrame extraction
        mapped_ts = pd.to_datetime(self._date_arr[idx])

        low_arr  = self._lower.loc[mapped_ts, features].values.astype(np.float64)
        up_arr   = self._upper.loc[mapped_ts, features].values.astype(np.float64)
        data_arr = df[features].values.astype(np.float64)

        clipped  = winsorize_clip_nb(data_arr, low_arr, up_arr)
        out      = df.copy()
        out[features] = clipped
        return out


class SectorNeutralScaler:
    """
    Cross-Sectional Standardization Engine.

    Normalizes features continuously by extracting structural Cross-Sectional Z-Scores 
    relative against the precise grouping definitions mapped dynamically per timestep.

    Architecturally, it functions statelessly regarding continuous temporal sequences 
    while maintaining rigorous state configurations regarding explicit grouping schemas.
    """

    def __init__(self, sector_col: str = "sector"):
        """
        Initializes the sector-neutral boundary evaluation module.
        
        Args:
            sector_col (str): The specific metadata column denoting categorical grouping bounds. 
                Defaults to strictly 'sector'.
        """
        self.sector_col = sector_col
        self._features  = []
        self._has_sector = False

    def fit(self, df: pd.DataFrame, features: list) -> "SectorNeutralScaler":
        """
        Verifies available schema keys mapping against internal state configurations.
        
        Args:
            df (pd.DataFrame): Training temporal panel evaluating standard definitions.
            features (list): Explicit feature structures requiring grouped neutralization.
            
        Returns:
            SectorNeutralScaler: Returns self for continuous functional pipelines.
        """
        self._features   = features
        self._has_sector = self.sector_col in df.columns
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Deploys standardized Cross-Sectional Z-Score algorithms strictly within boundary limits.

        Integrates dynamic epsilon padding boundaries systematically mitigating 
        catastrophic numerical instability inherent to flat execution regimes.
        
        Args:
            df (pd.DataFrame): Structural evaluation panel containing localized metadata frames.
            features (list): Mathematical feature subsets targeted for vector standardization.
            
        Returns:
            pd.DataFrame: Extracted and identically mapped Z-score matrices.
        """
        out = df.copy()
        if self._has_sector and self.sector_col in df.columns:
            grp = df.groupby(["date", self.sector_col])[features]
        else:
            # Enforces default system-wide neutralization parameters absent explicit sector schemas
            grp = df.groupby("date")[features]

        means = grp.transform("mean")
        stds  = grp.transform("std").fillna(1e-8).replace(0, 1e-8)

        out[features] = (df[features].values - means.values) / (stds.values + 1e-8)
        return out

    def inference_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Exposes standard structural alias for continuous production sequence evaluation.
        
        Args:
            df (pd.DataFrame): Incoming batch distribution tensor array.
            features (list): Feature keys strictly mapped for group alignment.
            
        Returns:
            pd.DataFrame: Computed transformations natively bounded by continuous data input vectors.
        """
        return self.transform(df, features)