"""
Market Calendar & Date Alignment Utilities
==========================================
Time-series alignment and exchange schedule management for rigorous backtesting.

Purpose
-------
The `date_utils` module acts as the temporal source of truth for the platform.
It interfaces with exchange calendars (via `pandas_market_calendars`) to strictly
differentiate between business days and actual trading days. This is crucial for:
1.  **Holiday Handling**: Skipping invalid dates (e.g., Good Friday, Christmas) to
    prevent zero-volume artifacts in signals.
2.  **Lag Generation**: Accurately computing $t-1$ for lookback windows.
3.  **Data Alignment**: Ensuring feature vectors and target labels share a common
    temporal index ($\mathcal{I}_{features} \cap \mathcal{I}_{targets}$).

Usage
-----
.. code-block:: python

    # Get valid NYSE trading days
    days = get_trading_days('2023-01-01', '2023-12-31', market='NYSE')

    # Find the previous valid trading session
    prev_day = get_previous_trading_day(pd.Timestamp('2023-01-03'))

Importance
----------
-   **Look-Ahead Bias Prevention**: Naively shifting by `BusinessDay(1)` fails on
    exchange holidays, potentially aligning $t$ with $t-2$ or creating missing data gaps.
-   **Performance**: Utilizes `lru_cache` for calendar instantiation to minimize overhead
    in tight loops ($O(1)$ access).

Tools & Frameworks
------------------
-   **Pandas Market Calendars**: Exchange-specific holiday schedules.
-   **Pandas**: Timestamp normalization and TimeSeries manipulation.
-   **Functools**: Memoization for calendar objects.
"""

import pandas as pd
import pandas_market_calendars as mcal
from typing import List, Tuple
from functools import lru_cache

@lru_cache(maxsize=4)  # Memoization: Caches calendar instances to avoid reconstruction overhead
def get_market_calendar(market: str = 'NYSE'):
    """
    Factory method to retrieve an exchange calendar instance.
    
    Args:
        market (str): Exchange MIC code (e.g., 'NYSE', 'LSE').
    """
    return mcal.get_calendar(market)

def get_trading_days(start_date: str, end_date: str, market: str = 'NYSE') -> List[pd.Timestamp]:
    """
    Generates a sequence of valid trading days $[t_{start}, t_{end}]$.
    
    Filters out weekends and exchange-specific holidays to ensure the schedule
    reflects actionable liquidity.
    """
    calendar = get_market_calendar(market)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    # The index of the schedule DataFrame represents the canonical trading dates
    return schedule.index.tolist()

def align_dates(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs an Inner Join on the temporal indices of two DataFrames.

    Ensures that both datasets share the exact same time axis $\mathcal{I}_{out} = \mathcal{I}_1 \cap \mathcal{I}_2$
    before correlation or regression analysis.
    """
    common_dates = df1.index.intersection(df2.index)
    return df1.loc[common_dates], df2.loc[common_dates]

def get_previous_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> pd.Timestamp:
    """
    Identifies the valid trading session strictly preceding the target date.
    
    .. math:: t_{prev} = \max \{ t \in \mathcal{S} \mid t < t_{target} \}
    
    Robustly handles weekends and holidays (e.g., querying Tuesday after Memorial Day returns Friday).
    """
    calendar = get_market_calendar(market)
    target_date_normalized = date.normalize()
    
    # Timezone Standardization:
    # Pandas Market Calendars typically return UTC timestamps. If the input is
    # timezone-naive, we localize to UTC to ensure consistency in comparison operations.
    if target_date_normalized.tzinfo is None:
        target_date_normalized = target_date_normalized.tz_localize('UTC')
    
    # Lookback Window: Query a small buffer period to find the preceding open day.
    schedule = calendar.schedule(start_date=target_date_normalized - pd.Timedelta(days=10), end_date=target_date_normalized)
    
    if schedule.empty:
        return date - pd.tseries.offsets.BusinessDay(1)

    # Filter: Select strictly prior dates
    days_before = schedule.index[schedule.index < target_date_normalized]
    
    if not days_before.empty:
        return days_before[-1]
    else:
        # Fallback: Edge case where target is before the start of the calendar's history
        return date - pd.tseries.offsets.BusinessDay(1)

def is_trading_day(date: pd.Timestamp, market: str = 'NYSE') -> bool:
    """
    Predicate to check if a specific timestamp corresponds to an open market session.
    """
    calendar = get_market_calendar(market)
    # Optimization: valid_days is faster than generating full schedules for single-day checks
    return date.normalize() in calendar.valid_days(start_date=date, end_date=date)