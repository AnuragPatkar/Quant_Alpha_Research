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

FIXES
-----
  BUG-086: Invalid escape sequences in docstrings and inline comments
           (e.g. \\epsilon, \\sigma) trigger SyntaxWarning in Python 3.12+
           and will become SyntaxError in future versions. All LaTeX math
           in non-raw strings must use either raw strings (r\"...\") or
           double backslashes. Converted all affected docstrings to raw
           strings and removed invalid escapes from inline comments.
"""
import numpy as np
import pandas as pd


def calculate_returns(prices: pd.Series) -> pd.Series:
    r"""
    Computes Discrete (Simple) Returns.

    .. math:: R_t = \frac{P_t}{P_{t-1}} - 1
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    r"""
    Computes Continuous (Log) Returns.

    .. math:: r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)

    Preferred for time-series aggregation due to additivity.
    """
    return np.log(prices / prices.shift(1))


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    r"""
    Calculates the Annualized Sharpe Ratio.

    .. math:: SR = \frac{E[R_p - R_f]}{\sigma_{excess}} \times \sqrt{252}

    Args:
        returns (pd.Series): Daily strategy returns.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).
    """
    if len(returns) < 2:
        return 0.0

    # Adjust annualized Risk-Free Rate to daily frequency
    excess_returns = returns - risk_free_rate / 252

    # Volatility Calculation: use std of excess returns (strict Information Ratio definition)
    std_dev = excess_returns.std()

    # Numerical Stability: Guard against DivisionByZero in zero-volatility regimes (e.g., cash)
    if std_dev < 1e-9 or np.isnan(std_dev) or std_dev == 0:
        return 0.0

    sharpe = (excess_returns.mean() / std_dev) * np.sqrt(252)
    return float(sharpe) if not np.isnan(sharpe) else 0.0


def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    r"""
    Computes the Drawdown Time Series.

    .. math:: DD_t = \frac{NAV_t - HWM_t}{HWM_t}

    Where HWM_t is the High Water Mark (running maximum) up to time t.
    """
    running_max = equity_curve.cummax()

    # Stability: Replace 0 with NaN in denominator to avoid Inf, then fill resulting NaNs.
    drawdown = (equity_curve - running_max) / running_max.replace(0, np.nan)
    return drawdown.fillna(0.0)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    r"""
    Calculates Maximum Drawdown (MaxDD).

    .. math:: MaxDD = \min_t (DD_t)
    """
    drawdown = calculate_drawdown(equity_curve)
    return float(abs(drawdown.min()))


def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    r"""
    Calculates the Annualized Sortino Ratio.

    Unlike Sharpe, Sortino only penalizes downside volatility (returns < MAR).

    .. math:: Sortino = \frac{E[R_p - R_f]}{\sigma_{down}} \times \sqrt{252}

    Where sigma_down is the Target Downside Deviation (LPM of degree 2).

    Formula verification:
        Ann_numerator   = mean_daily_excess * 252
        Ann_denominator = downside_std_daily * sqrt(252)
        Sortino         = (mean * 252) / (std * sqrt(252)) = mean * sqrt(252) / std
        Code:  downside_deviation = sqrt(mean(d^2)) * sqrt(252)   <- annualised
               sortino = mean * 252 / downside_deviation
                       = mean * 252 / (std * sqrt(252))           <- correct
    """
    if len(returns) < 2:
        return 0.0

    # De-annualize risk-free rate
    daily_rf = risk_free_rate / 252

    # Excess Returns relative to MAR (Minimum Acceptable Return)
    excess_returns = returns - daily_rf

    # Downside Variance Isolation: Filter for returns below the target
    downside_returns = excess_returns[excess_returns < 0]

    # Edge Case: Portfolio has no downside variance (Pure Alpha).
    # Mathematically, Sortino approaches infinity.
    if len(downside_returns) < 1:
        return np.inf

    # Lower Partial Moment (Degree 2) — annualised downside deviation
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

    if downside_deviation < 1e-9:
        return np.inf

    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation
    return float(sortino_ratio) if not np.isnan(sortino_ratio) else 0.0