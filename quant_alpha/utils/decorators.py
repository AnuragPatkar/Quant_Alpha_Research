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
-   **Observability**: `time_execution` enables identification of $O(N^2)$ bottlenecks
    in the research pipeline, critical for optimizing backtest throughput.
-   **Robustness**: `retry` implements the "Fail-Safe" pattern, ensuring the
    ETL pipeline recovers gracefully from stochastic network failures (e.g., API rate limits).

Tools & Frameworks
------------------
-   **Functools**: Preserves metadata (`__name__`, `__doc__`) of decorated functions.
-   **Time**: High-resolution clock access for latency measurement.
"""
import time
import functools
import logging

logger = logging.getLogger(__name__)

def time_execution(func):
    """
    Instrumentation decorator for measuring Wall-Clock execution time.
    
    Logs the latency $\Delta t = t_{end} - t_{start}$ to the active logger.
    
    Args:
        func (Callable): The function to wrap.
        
    Returns:
        Callable: The wrapped function with profiling side-effects.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            # Execute the core logic
            result = func(*args, **kwargs)
            return result
        finally:
            # Guaranteed execution regardless of exceptions to capture failure latency
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
    return wrapper

def retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Resilience decorator implementing Exponential Backoff.

    Wraps a function to automatically retry upon encountering specified transient exceptions.
    The wait time grows geometrically: $t_{wait} = \text{delay} \times \text{backoff}^{k}$.

    Args:
        max_retries (int): Maximum attempts before propagating the exception.
        delay (float): Initial wait time in seconds.
        backoff (float): Multiplier for the wait time after each failure.
        exceptions (Tuple[Exception]): Specific errors to catch (fail-safe). 
                                       Avoid catching `BaseException` to allow system interrupts.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_retries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            # Final attempt: propagate exception if this fails
            return func(*args, **kwargs)
        return wrapper
    return decorator