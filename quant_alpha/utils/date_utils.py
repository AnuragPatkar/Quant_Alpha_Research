"""
Market Calendar & Date Alignment Utilities
==========================================

Provides strict time-series alignment and exchange schedule management for rigorous backtesting.

Purpose
-------
This module serves as the temporal source of truth for the algorithmic platform.
It interfaces with execution exchange calendars to differentiate strictly between 
standard business days and actual market trading days. This enforces:
1. **Holiday Handling**: Skipping invalid dates to prevent zero-volume signal artifacts.
2. **Lag Generation**: Accurately computing structural $t-1$ lookback windows.
3. **Data Alignment**: Ensuring independent feature vectors and target labels map 
   to a unified cross-sectional temporal index.

Role in Quantitative Workflow
-----------------------------
Prevents Look-Ahead Bias by ensuring that lag functions never shift data into 
invalid future dates. Employs aggressive memoization ($O(1)$ access) to minimize 
instantiation overhead during high-frequency data loops.

Mathematical Dependencies
-------------------------
- **Pandas Market Calendars**: Instantiates exchange-specific holiday schedules.
- **Pandas**: Vectorized temporal normalization and offset manipulations.
- **Functools**: Structural memoization for optimization.
"""

import pandas as pd
import pandas_market_calendars as mcal
from typing import List, Tuple, Optional
from functools import lru_cache


@lru_cache(maxsize=4)
def get_market_calendar(market: str = 'NYSE'):
    """
    Factory initialization method returning an active exchange calendar object.
    
    Enforces an LRU cache to bypass expensive structural rebuilds during vector loops.

    Args:
        market (str): Exchange MIC identification code. Defaults to 'NYSE'.
        
    Returns:
        mcal.MarketCalendar: The mapped exchange calendar interface.
    """
    return mcal.get_calendar(market)


def get_trading_days(
    start_date: str,
    end_date: str,
    market: str = 'NYSE',
) -> List[pd.Timestamp]:
    """
    Generates a sequence of strictly valid trading days bounded by $[t_{start}, t_{end}]$.

    Aggressively filters structural weekends and exchange-specific holidays to ensure 
    the resulting temporal schedule accurately reflects actionable market liquidity.

    Args:
        start_date (str): The starting temporal boundary.
        end_date (str): The ending temporal boundary.
        market (str): Exchange MIC code. Defaults to 'NYSE'.
        
    Returns:
        List[pd.Timestamp]: A list of tz-naive timestamps normalized for vector compatibility.
    """
    calendar = get_market_calendar(market)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    # Strips localized timezone parameters to strictly avert merge sequence TypeErrors
    return schedule.index.tz_localize(None).tolist()


def align_dates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict inner-join bounding the temporal indices of two matrix datasets.

    Guarantees both datasets share a universally exact time axis prior to initiating 
    correlation tests, residual extraction, or structural regression analysis.

    Args:
        df1 (pd.DataFrame): Primary target dataframe.
        df2 (pd.DataFrame): Secondary target dataframe to align against.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The index-aligned target frames.
    """
    common_dates = df1.index.intersection(df2.index)
    return df1.loc[common_dates], df2.loc[common_dates]


def get_previous_trading_day(
    date: pd.Timestamp,
    market: str = 'NYSE',
) -> pd.Timestamp:
    r"""
    Identifies the nearest active trading session strictly preceding the target temporal coordinate.

    Solves $t_{prev} = \max \{ t \in S \mid t < t_{target} \}$.

    Robustly traverses backwards through weekends and macro holidays to extract the 
    true executing temporal limit (e.g., mapping a post-holiday Tuesday strictly back to Friday).

    Args:
        date (pd.Timestamp): The base coordinate date to evaluate.
        market (str): Exchange MIC code. Defaults to 'NYSE'.
        
    Returns:
        pd.Timestamp: Normalized timezone-naive timestamp representing previous market close.
    """
    calendar = get_market_calendar(market)
    target_date_normalized = date.normalize()

    # Transposes coordinate to localized UTC boundary for external calendar lookup evaluation
    if target_date_normalized.tzinfo is None:
        target_utc = target_date_normalized.tz_localize('UTC')
    else:
        target_utc = target_date_normalized.tz_convert('UTC')

    # Synthesizes an expanded 10-day structural buffer mitigating prolonged holiday overlaps
    schedule = calendar.schedule(
        start_date=target_utc - pd.Timedelta(days=10),
        end_date=target_utc,
    )

    if schedule.empty:
        # Engages standard BusinessDay approximation failover if strict history is inaccessible
        return (date - pd.tseries.offsets.BusinessDay(1)).normalize()

    # Isolates preceding discrete dates prior to systematically stripping timezone bindings
    days_before = schedule.index[schedule.index < target_utc]

    if not days_before.empty:
        return days_before[-1].tz_localize(None)
    else:
        return (date - pd.tseries.offsets.BusinessDay(1)).normalize()


def next_trading_date(
    date: pd.Timestamp,
    market: str = 'NYSE',
) -> pd.Timestamp:
    """
    Locates the initial valid trading session bound strictly ON or AFTER the target coordinate.

    Serves primarily for macro announcement-day lag evaluation, dictating that a 
    discrete event strictly becomes tradeable starting the subsequent open market session.

    Args:
        date (pd.Timestamp): Base coordinate event date.
        market (str): Exchange MIC code. Defaults to 'NYSE'.
        
    Returns:
        pd.Timestamp: A timezone-naive temporal marker mapping the subsequent execution target.
    """
    ts = date.normalize()
    dow = ts.weekday()  # Monday=0 ... Friday=4, Saturday=5, Sunday=6

    # Engages O(1) mathematical fast-path pushing Saturday/Sunday artifacts directly to Monday
    if dow == 5:
        ts = ts + pd.Timedelta(days=2)
    elif dow == 6:
        ts = ts + pd.Timedelta(days=1)

    # Executes comprehensive systemic holiday verification against standard calendar configurations
    try:
        calendar = get_market_calendar(market)
        schedule = calendar.schedule(
            start_date=ts,
            end_date=ts + pd.Timedelta(days=7),
        )
        if not schedule.empty:
            return schedule.index[0].tz_localize(None)
    except Exception:
        pass

    return ts


def date_range(
    start: str,
    end: str,
    freq: str = 'B',
) -> pd.DatetimeIndex:
    """
    Generates a localized continuous date distribution bound by Pandas frequency mechanics.

    Acts as a lightweight wrapper explicitly returning a tz-naive DatetimeIndex layer.

    Args:
        start (str): Boundary 'YYYY-MM-DD' start metric (inclusive).
        end (str): Boundary 'YYYY-MM-DD' termination metric (inclusive).
        freq (str): Target string interval mapping ('B' for business days, 
            'D' calendar days, 'W' weekly, 'M' month-end). Defaults to 'B'.
            
    Returns:
        pd.DatetimeIndex: Linear array of distributed time markers.
    """
    return pd.date_range(start=start, end=end, freq=freq)


def is_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> bool:
    """
    Evaluates Boolean criteria determining if the target date maps to an open market session.

    Optimizes verification by strictly leveraging calendar valid_days bounds, circumventing 
    expensive full-schedule generations for discrete queries.
    
    Args:
        date (pd.Timestamp): The evaluated date boundary.
        market (str): Exchange MIC code. Defaults to 'NYSE'.
        
    Returns:
        bool: True if the target maps to valid executing liquidity, otherwise False.
    """
    calendar = get_market_calendar(market)
    normalized = date.normalize()
    # Re-normalizes explicit tz-aware valid_days extractions to ensure symmetric comparison logic
    valid = calendar.valid_days(start_date=normalized, end_date=normalized)
    if normalized.tzinfo is None:
        return len(valid) > 0
    return normalized in valid