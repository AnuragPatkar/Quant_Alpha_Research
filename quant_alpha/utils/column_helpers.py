"""
column_helpers.py
-----------------
Safe column access utilities for factor base classes.
Prevents KeyError / 'Column not found' when OHLCV columns are missing
(e.g. 'high'/'low' shadowed/dropped after alt-data merge in DataManager).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_col(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """Return df[col] if present, else a NaN Series."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype=np.float64)