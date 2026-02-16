import pandas as pd
import numpy as np

def detect_earnings_events(group: pd.DataFrame) -> pd.Series:
    """
    Helper function for robust earnings event detection.
    Prevents bugs where EPS is exactly the same across two quarters.
    """
    # Check if report_date exists and has valid data
    if 'report_date' in group.columns and group['report_date'].notna().any():
        is_new = group['report_date'].notna() & (group['report_date'] != group['report_date'].shift(1))
    else:
        # Fallback: Detect if actuals, estimates, or surprises changed.
        # We check columns safely to avoid shape mismatches or key errors
        def _safe_change(col):
            if col not in group.columns: 
                return pd.Series(False, index=group.index)
            s = group[col]
            # Detect change: Values differ AND it's not just NaN!=NaN
            return (s != s.shift(1)) & ~(s.isna() & s.shift(1).isna())

        c1 = _safe_change('eps_actual')
        c2 = _safe_change('surprise_pct')
        c3 = _safe_change('eps_estimate')
            
        is_new = c1 | c2 | c3
    
    # Ensure it's a valid data point, not just transitioning from NaN
    return is_new & group['eps_actual'].notna()

def get_events_with_surprise(group: pd.DataFrame) -> pd.DataFrame:
    """Helper to get earnings events and ensure surprise_pct exists."""
    is_new = detect_earnings_events(group)
    events = group.loc[is_new].copy()
    
    if events.empty:
        return events
        
    # Ensure column exists and is float type
    if 'surprise_pct' not in events.columns:
        events['surprise_pct'] = np.nan
    
    # Force the column to be float64 or float32 to match the DF
    events['surprise_pct'] = events['surprise_pct'].astype(float)
        
    if 'eps_estimate' in events.columns and 'eps_actual' in events.columns:
        mask = events['surprise_pct'].isna()
        if mask.any():
            actuals = events.loc[mask, 'eps_actual']
            estimates = events.loc[mask, 'eps_estimate']
            
            valid_est = estimates.where(estimates.abs() > 1e-4, np.nan)
            
            # FIXED LINE: Added .astype(float) at the end to ensure scalar compatibility
            calc_surprise = ((actuals - valid_est) / valid_est.abs() * 100).astype(float)
            events.loc[mask, 'surprise_pct'] = calc_surprise
        
    return events