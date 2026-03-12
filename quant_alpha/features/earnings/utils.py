"""
Earnings Feature Utilities
==========================
Core helper functions for earnings event detection and surprise calculation.

Purpose
-------
This module provides the foundational logic for transforming raw, sparse fundamental
data into event-driven signals. It handles two critical tasks:
1. **Event Detection**: Robustly identifying when new earnings information becomes
   available, even in the absence of explicit `report_date` timestamps, using
   heuristic change detection.
2. **Surprise Imputation**: Calculating Standardized Unexpected Earnings (SUE)
   on-the-fly when pre-computed vendor fields are missing or malformed.

Usage
-----
These functions are primarily consumed by `EarningsFactor` subclasses in the
feature engineering pipeline.

.. code-block:: python

    from .utils import get_events_with_surprise

    # Extract valid earnings events from a raw time-series group
    events_df = get_events_with_surprise(ticker_group_df)

Importance
----------
- **Data Integrity**: Mitigates "ghost events" (duplicate rows) and ensures
  that signals are only generated upon genuine fundamental updates.
- **Signal Coverage**: Significantly increases factor coverage by imputing missing
  `surprise_pct` values directly from raw EPS actuals and estimates.
- **Numerical Stability**: Handles edge cases like zero-denominator estimates
  and type mismatches that often crash production pipelines.

Tools & Frameworks
------------------
- **Pandas**: Advanced indexing and masking for event filtering.
- **NumPy**: Vectorized arithmetic for efficient surprise calculation ($O(n)$).
"""

import pandas as pd
import numpy as np

def detect_earnings_events(group: pd.DataFrame) -> pd.Series:
    """
    Identifies time steps where new earnings information is released.
    
    Implements a dual-strategy detection mechanism:
    1. **Explicit**: Checks for changes in `report_date`.
    2. **Implicit**: Detects value shifts in EPS/Surprise data (fallback).
    """
    # Strategy 1: Explicit Event Detection via 'report_date'
    if 'report_date' in group.columns and group['report_date'].notna().any():
        is_new = group['report_date'].notna() & (group['report_date'] != group['report_date'].shift(1))
    else:
        # Strategy 2: Heuristic Detection via State Changes
        # Infers events by monitoring changes in fundamental values.
        def _safe_change(col):
            if col not in group.columns: 
                return pd.Series(False, index=group.index)
            s = group[col]
            # Logic: Value_t != Value_{t-1}, ignoring NaN->NaN transitions.
            return (s != s.shift(1)) & ~(s.isna() & s.shift(1).isna())

        c1 = _safe_change('eps_actual')      # Primary signal
        c2 = _safe_change('surprise_pct')    # Secondary signal
        c3 = _safe_change('eps_estimate')    # Tertiary signal
            
        is_new = c1 | c2 | c3
    
    # Validity Check: Ensure the detected event has concrete data (not just a transition from NaN)
    return is_new & group['eps_actual'].notna()

def get_events_with_surprise(group: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts earnings events and ensures the existence of a valid surprise metric.
    
    If `surprise_pct` is missing but `eps_actual` and `eps_estimate` are available,
    it computes the surprise on-the-fly to maximize signal coverage.
    """
    is_new = detect_earnings_events(group)
    events = group.loc[is_new].copy()
    
    if events.empty:
        return events
        
    # Schema Enforcement: Guarantee existence of 'surprise_pct' column
    if 'surprise_pct' not in events.columns:
        events['surprise_pct'] = np.nan
    
    # Type Safety: Enforce float precision for downstream numerical operations
    events['surprise_pct'] = events['surprise_pct'].astype(float)
        
    # Imputation Strategy: Calculate missing surprises from raw components
    if 'eps_estimate' in events.columns and 'eps_actual' in events.columns:
        mask = events['surprise_pct'].isna()
        if mask.any():
            actuals = events.loc[mask, 'eps_actual']
            estimates = events.loc[mask, 'eps_estimate']
            
            # Numerical Stability: Apply epsilon floor ($10^{-4}$) to denominator to prevent division-by-zero.
            valid_est = estimates.where(estimates.abs() > 1e-4, np.nan)
            
            # Vectorized Calculation: $$ Surprise\% = \frac{Actual - Estimate}{|Estimate|} \times 100 $$
            calc_surprise = ((actuals - valid_est) / valid_est.abs() * 100).astype(float)
            events.loc[mask, 'surprise_pct'] = calc_surprise
        
    return events