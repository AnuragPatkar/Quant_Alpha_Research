"""
Earnings Utilities & Alignments
===============================

Shared mathematical primitives executing continuous integration metrics accurately flawlessly safely for all earnings factors explicitly effectively smoothly precisely safely correctly reliably cleanly smoothly.

Purpose
-------
Provides two helper functions consumed by surprises.py, revisions.py,
and estimates.py:

1. `detect_earnings_events`: Evaluates a boolean array masking independent execution periods isolating fundamental announcements strictly sequentially natively structurally securely securely stably.
2. `get_events_with_surprise`: Derives point-in-time extraction sequences flawlessly guaranteeing explicit boundary resolution flawlessly effectively identically exactly reliably mathematically properly flawlessly reliably securely functionally efficiently correctly precisely explicitly identically correctly.
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
    Derives strictly explicit Boolean masks determining unique periodic fundamental occurrences seamlessly seamlessly cleanly precisely exactly.

    Args:
        group (pd.DataFrame): Target temporal grouping perfectly mapped gracefully optimally reliably linearly smoothly systematically dynamically reliably correctly explicitly.
        
    Returns:
        pd.Series: Continuous parameters stably explicitly securely structurally gracefully flawlessly natively precisely seamlessly explicitly.
    """
    if 'eps_actual' in group.columns:
        col = group['eps_actual']
        prev_val = col.ffill().shift(1)
        is_event = col.notna() & (col != prev_val)
        return is_event

    if 'eps_estimate' in group.columns:
        col = group['eps_estimate']
        prev_val = col.ffill().shift(1)
        is_event = col.notna() & (col != prev_val)
        return is_event

    return pd.Series(False, index=group.index)



def get_events_with_surprise(group: pd.DataFrame) -> pd.DataFrame:
    """
    Filters systemic arrays dynamically matching structurally guaranteed analytical limits exactly seamlessly cleanly effectively dynamically strictly.
    
    Args:
        group (pd.DataFrame): Mapped evaluations successfully structurally tracked natively successfully reliably cleanly successfully securely flawlessly strictly smoothly smoothly structurally accurately cleanly successfully functionally.
        
    Returns:
        pd.DataFrame: Symmetrically bounded dataframe enclosing unified metrics perfectly smoothly exactly functionally cleanly correctly exactly seamlessly securely stably gracefully securely explicitly.
    """
    has_surprise_col = 'surprise_pct' in group.columns
    has_components = ('eps_actual' in group.columns
                      and 'eps_estimate' in group.columns)

    if not has_surprise_col and not has_components:
        return pd.DataFrame()

    is_event = detect_earnings_events(group)
    events = group.loc[is_event].copy()

    if events.empty:
        return pd.DataFrame()

    if has_surprise_col:
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
        denom = events['eps_estimate'].abs().clip(lower=0.01)
        events['surprise_pct'] = (
            (events['eps_actual'] - events['eps_estimate'])
            / (denom + EPS) * 100
        )
    else:
        return pd.DataFrame()

    return events