"""
Quantitative Financial Mathematics Library
==========================================
Core primitives for return calculation, risk attribution, and performance metrics.

Purpose
-------
The `math_utils` module provides a vectorized, numerically stable toolkit for
transforming price series into actionable risk/reward metrics. It implements
industry-standard formulas for Sharpe, Sortino, and Drawdown analysis,
handling edge cases such as zero-volatility periods and sparse data.

Usage
-----
Intended for use in both Research (Factor Analysis) and Production (Reporting).

.. code-block:: python

    # Calculate rolling Sharpe Ratio
    sharpe = calculate_sharpe(strategy_returns, risk_free_rate=0.04)

    # Compute Max Drawdown for risk management
    mdd = calculate_max_drawdown(equity_curve)

Importance
----------
-   **Standardization**: Ensures consistent performance reporting across all strategies,
    preventing "methodology arbitrage" where different formulas yield varying results.
-   **Numerical Stability**: Explicit handling of epsilon (machine epsilon) and
    infinite values prevents pipeline crashes during regime shifts (e.g., flat markets).
-   **Vectorization**: Utilizes NumPy/Pandas primitives for O(N) performance on
    high-frequency time series.

Tools & Frameworks
------------------
-   **NumPy**: Efficient array operations for log-returns and deviation calculations.
-   **Pandas**: Time-series alignment and rolling window operations.
"""
import numpy as np
import pandas as pd


def calculate_returns(prices: pd.Series) -> pd.Series:
    r"""
    Computes Discrete (Simple) Returns.

    .. math:: R_t = \frac{P_t}{P_{t-1}} - 1

    Args:
        prices (pd.Series): A time series of asset prices.

    Returns:
        pd.Series: The discrete period-to-period returns.
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    r"""
    Computes Continuous (Log) Returns.

    .. math:: r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)

    Preferred for time-series aggregation due to additivity.

    Args:
        prices (pd.Series): A time series of asset prices.

    Returns:
        pd.Series: The continuous logarithmic returns.
    """
    return np.log(prices / prices.shift(1))


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    r"""
    Calculates the Annualized Sharpe Ratio.

    .. math:: SR = \frac{E[R_p - R_f]}{\sigma_{excess}} \times \sqrt{252}

    Args:
        returns (pd.Series): Daily strategy returns.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).

    Returns:
        float: The annualized Sharpe ratio. Returns 0.0 if there is zero volatility
            or insufficient data.
    """
    if len(returns) < 2:
        return 0.0

    # Temporal scaling: De-annualize risk-free rate to match the daily return frequency
    excess_returns = returns - risk_free_rate / 252

    # Denominator: Calculates the standard deviation of excess returns
    std_dev = excess_returns.std()

    # Stability Guard: Prevents DivisionByZero exceptions during flat market regimes
    if std_dev < 1e-9 or np.isnan(std_dev) or std_dev == 0:
        return 0.0

    # Annualization: Scales the daily Sharpe ratio by the square root of trading days
    sharpe = (excess_returns.mean() / std_dev) * np.sqrt(252)
    return float(sharpe) if not np.isnan(sharpe) else 0.0


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    r"""
    Computes the Drawdown Time Series.

    .. math:: DD_t = \frac{NAV_t - HWM_t}{HWM_t}

    Where HWM_t is the High Water Mark (running maximum) up to time t.

    Args:
        equity_curve (pd.Series): The cumulative portfolio value or asset price time series.

    Returns:
        pd.Series: A time series representing the percentage drawdown from the peak at each point in time.
    """
    running_max = equity_curve.cummax()

    # Stability Guard: Replaces 0 with NaN in the denominator to avoid Infinity values,
    # then safely fills resulting NaNs with 0.0 to maintain structural integrity.
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return drawdown.fillna(0.0)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    r"""
    Calculates Maximum Drawdown (MaxDD).

    .. math:: MaxDD = \min_t (DD_t)

    Args:
        equity_curve (pd.Series): The cumulative portfolio value or asset price time series.

    Returns:
        float: The absolute value of the maximum peak-to-trough drawdown percentage.
    """
    drawdown = calculate_drawdown(equity_curve)
    return float(abs(drawdown.min()))


def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    r"""
    Calculates the Annualized Sortino Ratio.

    Unlike Sharpe, Sortino only penalizes downside volatility (returns < MAR).

    .. math:: Sortino = \frac{E[R_p - R_f]}{\sigma_{down}} \times \sqrt{252}

    Where sigma_down is the Target Downside Deviation (LPM of degree 2).

    Args:
        returns (pd.Series): Daily strategy returns.
        risk_free_rate (float): Annualized risk-free rate or Minimum Acceptable Return (MAR).

    Returns:
        float: The annualized Sortino ratio. Returns positive infinity if there is
            zero downside variance.
    """
    if len(returns) < 2:
        return 0.0

    # Temporal scaling: De-annualize risk-free rate to match the daily frequency
    daily_rf = risk_free_rate / 252

    # Isolates excess returns against the specified MAR hurdle
    excess_returns = returns - daily_rf

    # Asymmetric Risk Profile: Discards positive variance to isolate downside deviations
    downside_returns = excess_returns[excess_returns < 0]

    # Stability Guard: If the strategy exhibits strictly positive returns relative to MAR,
    # the denominator is zero. Mathematically, Sortino approaches infinity.
    if len(downside_returns) < 1:
        return np.inf

    # Denominator: Calculates the annualized Lower Partial Moment (Degree 2)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

    # Stability Guard: Prevents DivisionByZero exceptions during micro-variance regimes
    if downside_deviation < 1e-9:
        return np.inf

    # Annualization: Scales the daily expected return and divides by downside risk
    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation
    return float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0