"""
Date handling utilities
"""
import pandas as pd
import pandas_market_calendars as mcal
from typing import List, Tuple
from functools import lru_cache

@lru_cache(maxsize=4) # Increased for future multi-market support
def get_market_calendar(market: str = 'NYSE'):
    """
    Gets an instance of a market calendar, cached for performance.
    Default is NYSE.
    """
    return mcal.get_calendar(market)

def get_trading_days(start_date: str, end_date: str, market: str = 'NYSE') -> List[pd.Timestamp]:
    """
    Get a list of actual trading days between start and end date using a market calendar.
    This correctly handles market holidays.
    """
    calendar = get_market_calendar(market)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    # The index of the schedule DataFrame contains the valid trading days
    return schedule.index.tolist()

def align_dates(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two DataFrames by common dates.
    """
    common_dates = df1.index.intersection(df2.index)
    return df1.loc[common_dates], df2.loc[common_dates]

def get_previous_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> pd.Timestamp:
    """
    Get the previous trading day using the market calendar, correctly handling holidays.
    """
    calendar = get_market_calendar(market)
    target_date_normalized = date.normalize()
    
    # Ensure target is tz-naive for comparison if calendar is tz-naive, or handle conversion
    # mcal schedules are usually UTC. We strip tz for robust comparison if input is naive.
    if target_date_normalized.tzinfo is None:
        target_date_normalized = target_date_normalized.tz_localize('UTC')
    
    # Get trading days in a window before the target date
    schedule = calendar.schedule(start_date=target_date_normalized - pd.Timedelta(days=10), end_date=target_date_normalized)
    
    # Find the last trading day strictly before the target date
    days_before = schedule.index[schedule.index < target_date_normalized]
    
    if not days_before.empty:
        return days_before[-1]
    else:
        # Fallback for very rare edge cases (e.g., date is before calendar starts)
        return date - pd.tseries.offsets.BusinessDay(1)

def is_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> bool:
    """
    Check if a given date is a trading day for the specified market.
    """
    calendar = get_market_calendar(market)
    # Use valid_days for an efficient check without creating a full schedule DataFrame
    return date.normalize() in calendar.valid_days(start_date=date, end_date=date)