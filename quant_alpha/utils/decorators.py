"""
Aspect-Oriented Programming (AOP) Utilities
===========================================
Decorators for cross-cutting concerns such as profiling and fault tolerance.

Purpose
-------
This module provides high-order functions to augment the behavior of core
execution logic without modifying the underlying source code. It handles:
1.  **Performance Profiling**: Instrumentation of execution latency.
2.  **Resilience**: Transient fault handling via exponential backoff retries.

Usage
-----
.. code-block:: python

    @time_execution
    @retry(max_retries=3, backoff=2, exceptions=(ConnectionError,))
    def fetch_market_data(ticker):
        ...

Importance
----------
-   **Observability**: `time_execution` enables identification of O(N^2) bottlenecks
    in the research pipeline, critical for optimizing backtest throughput.
-   **Robustness**: `retry` implements the "Fail-Safe" pattern, ensuring the
    ETL pipeline recovers gracefully from stochastic network failures (e.g., API rate limits).

Tools & Frameworks
------------------
-   **Functools**: Preserves metadata (__name__, __doc__) of decorated functions.
-   **Time**: High-resolution clock access for latency measurement.

FIXES
-----
  BUG-086: Invalid escape sequence \\Delta in inline comment on line 44
           (\"Logs the latency $\\Delta t = ...$\") triggers SyntaxWarning
           in Python 3.12+ and is an error in future versions.
           Fixed by removing LaTeX from plain string context.

  Also: `timer` alias added so quant_alpha/utils/__init__.py can export it
  under the name used by the rest of the codebase (timer, retry).
"""
import time
import functools
import logging
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def time_execution(func):
    """
    Instrumentation decorator for measuring wall-clock execution time.

    Logs the elapsed time (t_end - t_start) to the active logger.
    Execution time is captured in the `finally` block to ensure it is
    recorded even if the wrapped function raises.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function with profiling side-effects.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Use perf_counter for sub-millisecond accuracy
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Guaranteed execution regardless of exceptions to capture failure latency
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[timer] '{func.__name__}' completed in {elapsed:.4f}s"
            )
    return wrapper


# Alias — quant_alpha/utils/__init__.py exports `timer` (matches codebase convention)
timer = time_execution


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Resilience decorator implementing Exponential Backoff.

    Wraps a function to automatically retry upon encountering specified
    transient exceptions. The wait time grows geometrically:
    t_wait = delay * backoff^k

    Args:
        max_retries (int)  : Maximum attempts before propagating the exception.
        delay (float)      : Initial wait time in seconds.
        backoff (float)    : Multiplier for the wait time after each failure.
        exceptions (tuple) : Specific errors to catch (fail-safe).
                             Avoid catching BaseException to allow system interrupts.

    Example
    -------
    .. code-block:: python

        @retry(max_retries=3, delay=1, backoff=2, exceptions=(IOError,))
        def download_prices(ticker: str) -> pd.DataFrame:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts_left = max_retries
            current_delay = delay
            while attempts_left > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        f"[retry] '{func.__name__}' failed "
                        f"({max_retries - attempts_left + 1}/{max_retries}): "
                        f"{e}. Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    attempts_left -= 1
                    current_delay *= backoff
            # Final attempt: propagate exception if this also fails
            return func(*args, **kwargs)
        return wrapper
    return decorator