"""
Feature Preprocessing & Normalization Engine
============================================
High-performance transformation kernels for signal conditioning.

FIXES:
  BUG-039: WinsorisationScaler.fit() now normalises its date array to
           timezone-naive int64 (nanoseconds) so that np.searchsorted works
           correctly even when the training DataFrame has tz-aware dates.
           transform() strips tz from df["date"] before comparison.

  BUG-045: WinsorisationScaler now reads clip_pct from config.WINSORIZE_QUANTILES
           instead of a hardcoded default, making config.WINSORIZE_QUANTILES the
           single source of truth (as required by the architectural decision).
           The default fallback of 0.01 is preserved when config is unavailable.
"""

import numpy as np
import pandas as pd
from numba import njit, prange


# ---------------------------------------------------------------------------
# Read clip_pct from config (BUG-045)
# ---------------------------------------------------------------------------

def _default_clip_pct() -> float:
    """Return the lower winsorize quantile from config, or 0.01 as fallback."""
    try:
        from config.settings import config
        lo, _hi = config.WINSORIZE_QUANTILES
        return float(lo)
    except Exception:
        return 0.01


# ==================== NUMBA KERNELS ====================

@njit(parallel=True, cache=True)
def winsorize_clip_nb(data, lower, upper):
    """
    Numba-accelerated parallel clipping kernel.

    Args:
        data  (np.ndarray): Input feature matrix (N × F).
        lower (np.ndarray): Lower bound matrix (N × F), row-aligned.
        upper (np.ndarray): Upper bound matrix (N × F), row-aligned.

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


# ==================== SCALERS ====================

class WinsorisationScaler:
    """
    Stateful Outlier Mitigation via Historical Quantiles.

    Learns the distribution bounds [q_low, q_high] per timestamp during fitting,
    and applies these bounds to future data using "Last Observation Carried
    Forward" (LOCF) logic for regime stability.

    FIX BUG-045: clip_pct defaults to config.WINSORIZE_QUANTILES[0] so that
    config is the single source of truth.

    FIX BUG-039: All date arrays stored and compared as tz-naive int64
    (nanoseconds since epoch) to avoid TypeError / silent miscompare when
    mixing tz-aware and tz-naive datetimes.
    """

    def __init__(self, clip_pct: float | None = None):
        # FIX BUG-045: use config value unless explicitly overridden
        self.clip_pct  = clip_pct if clip_pct is not None else _default_clip_pct()
        self._lower    = None
        self._upper    = None
        self._date_arr = None  # int64 nanosecond timestamps (tz-naive)

    def fit(self, df: pd.DataFrame, features: list) -> "WinsorisationScaler":
        """
        Computes cross-sectional quantiles per date.

        FIX BUG-039: self._date_arr is stored as tz-naive int64 nanoseconds
        so that np.searchsorted works correctly regardless of whether df['date']
        has timezone info at transform time.
        """
        grp         = df.groupby("date")[features]
        self._lower = grp.quantile(self.clip_pct)
        self._upper = grp.quantile(1.0 - self.clip_pct)

        # FIX BUG-039: normalise to tz-naive int64
        idx = pd.DatetimeIndex(self._lower.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        self._date_arr = idx.astype(np.int64)
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Applies winsorization using the nearest historical bounds.

        FIX BUG-039: Converts df["date"] to tz-naive int64 nanoseconds before
        calling np.searchsorted to guarantee a consistent comparison domain.
        """
        # FIX BUG-039: strip tz and convert to int64 nanoseconds
        raw_dates = pd.to_datetime(df["date"])
        if raw_dates.dt.tz is not None:
            raw_dates = raw_dates.dt.tz_localize(None)
        dates_i64 = raw_dates.values.astype(np.int64)

        # Map each row to the most recent fit date (forward-fill semantics)
        idx    = np.searchsorted(self._date_arr, dates_i64, side="right") - 1
        idx    = np.clip(idx, 0, len(self._date_arr) - 1)

        # Reconstruct Timestamps for .loc lookup (use tz-naive)
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

    Standardizes features by computing the Z-Score relative to the grouping
    (Date, Sector) distribution at that specific timestep.

    Note: This is stateless regarding time (calculates stats on the fly),
    but stateful regarding schema (sector column name).
    """

    def __init__(self, sector_col: str = "sector"):
        self.sector_col = sector_col
        self._features  = []
        self._has_sector = False

    def fit(self, df: pd.DataFrame, features: list) -> "SectorNeutralScaler":
        """Validates schema and registers feature set."""
        self._features   = features
        self._has_sector = self.sector_col in df.columns
        return self

    def transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Computes and applies Cross-Sectional Z-Score.

        Includes epsilon handling to ensure numerical stability in low-variance
        regimes.
        """
        out = df.copy()
        if self._has_sector and self.sector_col in df.columns:
            grp = df.groupby(["date", self.sector_col])[features]
        else:
            # Fallback: market-wide standardization if sector data is missing
            grp = df.groupby("date")[features]

        means = grp.transform("mean")
        stds  = grp.transform("std").fillna(1e-8).replace(0, 1e-8)

        out[features] = (df[features].values - means.values) / (stds.values + 1e-8)
        return out

    def inference_transform(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """Alias for transform — stats derived from the input batch."""
        return self.transform(df, features)