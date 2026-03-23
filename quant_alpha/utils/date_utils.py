"""
Market Calendar & Date Alignment Utilities
==========================================
Time-series alignment and exchange schedule management for rigorous backtesting.

Purpose
-------
The `date_utils` module acts as the temporal source of truth for the platform.
It interfaces with exchange calendars (via pandas_market_calendars) to strictly
differentiate between business days and actual trading days. This is crucial for:
1.  **Holiday Handling**: Skipping invalid dates (e.g., Good Friday, Christmas) to
    prevent zero-volume artifacts in signals.
2.  **Lag Generation**: Accurately computing t-1 for lookback windows.
3.  **Data Alignment**: Ensuring feature vectors and target labels share a common
    temporal index (intersection of features and targets index).

Usage
-----
.. code-block:: python

    # Get valid NYSE trading days
    days = get_trading_days('2023-01-01', '2023-12-31', market='NYSE')

    # Find the previous valid trading session
    prev_day = get_previous_trading_day(pd.Timestamp('2023-01-03'))

Importance
----------
-   **Look-Ahead Bias Prevention**: Naively shifting by BusinessDay(1) fails on
    exchange holidays, potentially aligning t with t-2 or creating missing data gaps.
-   **Performance**: Utilizes lru_cache for calendar instantiation to minimize overhead
    in tight loops (O(1) access after first call).

Tools & Frameworks
------------------
-   **Pandas Market Calendars**: Exchange-specific holiday schedules.
-   **Pandas**: Timestamp normalization and TimeSeries manipulation.
-   **Functools**: Memoization for calendar objects.

FIXES
-----
  BUG-086: Invalid escape sequences in docstrings (\\mathcal, \\max, \\{ etc.)
           trigger SyntaxWarning in Python 3.12+ and will become SyntaxError.
           All LaTeX math moved to raw strings or escape sequences removed.

  BUG-087: get_previous_trading_day() returned a tz-aware (UTC) Timestamp
           from pandas_market_calendars schedule index. Callers throughout
           the pipeline (integration.py, DataManager) use tz-naive dates.
           Comparing tz-aware vs tz-naive raises TypeError in pandas.
           Fix: strip tz with .tz_localize(None) before returning.
           Also applies to next_trading_date (added as companion function).
"""

import pandas as pd
import pandas_market_calendars as mcal
from typing import List, Tuple, Optional
from functools import lru_cache


@lru_cache(maxsize=4)
def get_market_calendar(market: str = 'NYSE'):
    """
    Factory method to retrieve an exchange calendar instance.
    Cached with lru_cache to avoid reconstruction overhead.

    Args:
        market (str): Exchange MIC code (e.g., 'NYSE', 'LSE').
    """
    return mcal.get_calendar(market)


def get_trading_days(
    start_date: str,
    end_date: str,
    market: str = 'NYSE',
) -> List[pd.Timestamp]:
    """
    Generates a sequence of valid trading days [t_start, t_end].

    Filters out weekends and exchange-specific holidays to ensure the schedule
    reflects actionable liquidity.

    Returns
    -------
    List[pd.Timestamp] -- tz-naive timestamps for compatibility with price panels.
    """
    calendar = get_market_calendar(market)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    # FIX BUG-087: strip tz so downstream merges don't raise TypeError
    return schedule.index.tz_localize(None).tolist()


def align_dates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs an inner join on the temporal indices of two DataFrames.

    Ensures both datasets share the exact same time axis before correlation
    or regression analysis (intersection of their indices).
    """
    common_dates = df1.index.intersection(df2.index)
    return df1.loc[common_dates], df2.loc[common_dates]


def get_previous_trading_day(
    date: pd.Timestamp,
    market: str = 'NYSE',
) -> pd.Timestamp:
    r"""
    Identifies the valid trading session strictly preceding the target date.

    t_prev = max { t in S | t < t_target }

    Robustly handles weekends and holidays (e.g., querying Tuesday after
    Memorial Day returns Friday).

    Returns
    -------
    pd.Timestamp -- tz-naive for compatibility with price panel dates.

    FIX BUG-087: Previously returned a tz-aware UTC Timestamp from the
    pandas_market_calendars schedule index. Comparing this against tz-naive
    dates in DataManager / integration.py raises TypeError. Now strips tz
    before returning.
    """
    calendar = get_market_calendar(market)
    target_date_normalized = date.normalize()

    # Normalize input to UTC for calendar query, then strip on return
    if target_date_normalized.tzinfo is None:
        target_utc = target_date_normalized.tz_localize('UTC')
    else:
        target_utc = target_date_normalized.tz_convert('UTC')

    # Query a 10-day buffer to find the preceding open day
    schedule = calendar.schedule(
        start_date=target_utc - pd.Timedelta(days=10),
        end_date=target_utc,
    )

    if schedule.empty:
        # Fallback: BusinessDay approximation when calendar history is unavailable
        return (date - pd.tseries.offsets.BusinessDay(1)).normalize()

    # Select strictly prior dates, then strip tz (FIX BUG-087)
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
    Find the first valid trading session ON OR AFTER the given date.

    Used for announcement-day lag: an event on date d is tradeable starting
    the next open market session.

    Returns
    -------
    pd.Timestamp -- tz-naive.
    """
    ts = date.normalize()
    dow = ts.weekday()  # Monday=0 ... Friday=4, Saturday=5, Sunday=6

    # Fast path: roll Saturday -> Monday, Sunday -> Monday
    if dow == 5:
        ts = ts + pd.Timedelta(days=2)
    elif dow == 6:
        ts = ts + pd.Timedelta(days=1)

    # Check for holidays via calendar
    try:
        calendar = get_market_calendar(market)
        schedule = calendar.schedule(
            start_date=ts,
            end_date=ts + pd.Timedelta(days=7),
        )
        if not schedule.empty:
            return schedule.index[0].tz_localize(None)
    except Exception:
        pass  # fall back to weekday-only logic

    return ts


def date_range(
    start: str,
    end: str,
    freq: str = 'B',
) -> pd.DatetimeIndex:
    """
    Generate a date range using pandas frequency strings.

    Thin wrapper that returns tz-naive DatetimeIndex.

    Parameters
    ----------
    start, end : str  -- 'YYYY-MM-DD' boundaries (inclusive)
    freq       : str  -- 'B' = business days, 'D' = calendar days,
                         'W' = weekly, 'M' = month-end
    """
    return pd.date_range(start=start, end=end, freq=freq)


def is_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> bool:
    """
    Predicate: is the given date an open market session?

    Uses pandas_market_calendars valid_days which is faster than generating
    a full schedule for single-day checks.
    """
    calendar = get_market_calendar(market)
    normalized = date.normalize()
    # valid_days returns tz-aware; compare after normalizing both sides
    valid = calendar.valid_days(start_date=normalized, end_date=normalized)
    if normalized.tzinfo is None:
        return len(valid) > 0
    return normalized in valid