"""
features/earnings/utils.py
===========================
Shared utilities for all earnings factor classes.

Purpose
-------
Provides two helper functions consumed by surprises.py, revisions.py,
and estimates.py:

  detect_earnings_events   — returns a boolean mask identifying the FIRST row
                             on which a NEW earnings event appears for a ticker.

  get_events_with_surprise — returns the subset of rows that are genuine
                             earnings event rows, with a guaranteed 'surprise_pct'
                             column derived from components if not already present.

BUG-028 FIX: This entire file was missing from the project. All earnings factors
import from '.utils' at module load time, so the ImportError crashed the full
feature pipeline before a single factor computed.

Design notes
------------
- An "earnings event" is identified by a change in eps_actual (when the column
  exists), or by a non-NaN eps_estimate row that differs from the previous row.
  This handles both daily-frequency data (where earnings appear as a single
  non-NaN obs among many NaNs) and quarterly-only data (every row is an event).

- Forward-filling is intentionally NOT done inside these helpers; callers that
  need daily signals must ffill the result of the factor computation themselves
  (see EPSSurprise.compute() in surprises.py for the canonical pattern).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


# ---------------------------------------------------------------------------
# detect_earnings_events
# ---------------------------------------------------------------------------

def detect_earnings_events(group: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask indicating which rows are genuine earnings events.

    Detection logic (in priority order):
    1. If 'eps_actual' exists: event = row where eps_actual is non-NaN
       AND differs from the previous non-NaN eps_actual value (new quarter).
    2. If only 'eps_estimate' exists: event = row where eps_estimate is non-NaN
       AND differs from the previous non-NaN eps_estimate (new period).
    3. If neither exists: return all-False mask.

    Parameters
    ----------
    group : pd.DataFrame
        Single-ticker slice, already sorted ascending by date.

    Returns
    -------
    pd.Series (bool), same index as group.
    """
    if 'eps_actual' in group.columns:
        col = group['eps_actual']
        # A new event is a non-NaN row whose value differs from the previous
        # non-NaN row.  Using ffill to propagate the last known value allows
        # us to detect the change reliably even if rows are sparse.
        prev_val = col.ffill().shift(1)
        is_event = col.notna() & (col != prev_val)
        return is_event

    if 'eps_estimate' in group.columns:
        col = group['eps_estimate']
        prev_val = col.ffill().shift(1)
        is_event = col.notna() & (col != prev_val)
        return is_event

    # No earnings columns at all — nothing to detect
    return pd.Series(False, index=group.index)


# ---------------------------------------------------------------------------
# get_events_with_surprise
# ---------------------------------------------------------------------------

def get_events_with_surprise(group: pd.DataFrame) -> pd.DataFrame:
    """
    Return the subset of rows that are earnings events, with a 'surprise_pct'
    column guaranteed to be present and computed.

    Derivation priority for 'surprise_pct':
    1. Use the pre-computed 'surprise_pct' column if already present and
       non-NaN on this row.
    2. Derive from eps_actual and eps_estimate:
           surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate) * 100
       Uses a 0.01 floor on the denominator to avoid explosion near zero EPS.
    3. If neither source is available, return an empty DataFrame.

    Parameters
    ----------
    group : pd.DataFrame
        Single-ticker slice, already sorted ascending by date.

    Returns
    -------
    pd.DataFrame — rows that are earnings events with 'surprise_pct' present.
                   May be empty if no events are detected or data is missing.
    """
    has_surprise_col = 'surprise_pct' in group.columns
    has_components = ('eps_actual' in group.columns
                      and 'eps_estimate' in group.columns)

    if not has_surprise_col and not has_components:
        return pd.DataFrame()

    # Identify event rows
    is_event = detect_earnings_events(group)
    events = group.loc[is_event].copy()

    if events.empty:
        return pd.DataFrame()

    # Ensure surprise_pct is present and filled in
    if has_surprise_col:
        # Fill any NaN surprise_pct from components where possible
        if has_components:
            missing_mask = events['surprise_pct'].isna()
            if missing_mask.any():
                denom = events.loc[missing_mask, 'eps_estimate'].abs().clip(lower=0.01)
                derived = (
                    (events.loc[missing_mask, 'eps_actual']
                     - events.loc[missing_mask, 'eps_estimate'])
                    / (denom + EPS) * 100
                )
                events.loc[missing_mask, 'surprise_pct'] = derived
    elif has_components:
        # Derive surprise_pct entirely from components
        denom = events['eps_estimate'].abs().clip(lower=0.01)
        events['surprise_pct'] = (
            (events['eps_actual'] - events['eps_estimate'])
            / (denom + EPS) * 100
        )
    else:
        return pd.DataFrame()

    return events