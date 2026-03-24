"""
Aspect-Oriented Programming (AOP) Utilities
===========================================

Provides high-order decorators managing execution profiling and transient fault tolerance.

Purpose
-------
This module exposes functional wrappers designed to augment the behavioral execution 
of core logic primitives without requiring modifications to the underlying target source. 
It seamlessly orchestrates:
1. **Performance Profiling**: Granular instrumentation mapping execution latency constraints.
2. **Resilience Boundaries**: Transient fault extraction utilizing exponential backoff retries.

Role in Quantitative Workflow
-----------------------------
- **Observability**: Ensures rapid identification of $O(N^2)$ algorithm bottlenecks 
  percolating across the pipeline, strictly prioritizing high-frequency backtest throughput.
- **Robustness**: Enforces systematic "Fail-Safe" recovery procedures, guaranteeing 
  the ingestion ETL gracefully bounds network variance (e.g., HTTP rate limits).

Mathematical Dependencies
-------------------------
- **Functools**: Preserves metadata schemas of wrapped primitives structurally.
- **Time**: Integrates `perf_counter` mapping sub-millisecond execution precision.
"""
import time
import functools
import logging
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def time_execution(func):
    """
    Instrumentation wrapper isolating exact sub-millisecond wall-clock execution durations.

    Implicitly logs the absolute duration delta to the standardized active logger. 
    Evaluates tracking mechanics explicitly inside a `finally` block to ensure 
    terminal state recording even upon severe structural exceptions.

    Args:
        func (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function with profiling side-effects.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Guarantees extraction execution capturing latency regardless of stack exceptions
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[timer] '{func.__name__}' completed in {elapsed:.4f}s"
            )
    return wrapper


# Exports global standardized alias bridging backwards compatibility targets
timer = time_execution


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Resilience wrapper implementing geometrical Exponential Backoff recovery loops.

    Automatically intercepts specific transient runtime exceptions and re-initiates 
    execution, compounding delays structurally by: $t_{wait} = delay \times backoff^k$.

    Args:
        max_retries (int): Maximum bounded attempts before strictly propagating the fault.
        delay (float): Initial execution suspension interval evaluated in seconds.
        backoff (float): Mathematical multiplier expanding the wait horizon on failure.
        exceptions (tuple): Strict class tuple defining target error states. Excludes 
            `BaseException` to safely map keyboard or system interrupts.
            
    Returns:
        Callable: The fault-tolerant wrapped function.
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
            # Initiates terminal attempt boundary allowing strict propagation if state fails
            return func(*args, **kwargs)
        return wrapper
    return decorator