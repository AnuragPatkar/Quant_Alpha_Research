"""
Math and financial calculation utilities
"""
import numpy as np
import pandas as pd

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns.
    """
    return prices.pct_change()

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns.
    """
    return np.log(prices / prices.shift(1))

def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    
    # Use standard deviation of excess returns for theoretical correctness
    std_dev = excess_returns.std()
    
    # Guard against division by zero
    if std_dev == 0:
        return 0.0
        
    sharpe = (excess_returns.mean() / std_dev) * np.sqrt(252)
    return sharpe if not np.isnan(sharpe) else 0.0

def calculate_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    """
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    # Handle division by zero if equity curve starts at 0, resulting in NaN
    return drawdown.fillna(0.0)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    """
    drawdown = calculate_drawdown(equity_curve)
    return abs(drawdown.min())

def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sortino ratio, which only penalizes downside volatility.
    """
    if len(returns) < 2:
        return 0.0

    # Daily risk free rate
    daily_rf = risk_free_rate / 252

    # Calculate excess returns
    excess_returns = returns - daily_rf

    # Target Downside Deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1:
        return np.inf  # No downside returns, Sortino is technically infinite

    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)

    if downside_deviation == 0:
        return np.inf

    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation
    return sortino_ratio if not np.isnan(sortino_ratio) else 0.0

def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sortino ratio, which only penalizes downside volatility.
    """
    if len(returns) < 2:
        return 0.0

    # Daily risk free rate
    daily_rf = risk_free_rate / 252

    # Calculate excess returns
    excess_returns = returns - daily_rf

    # Target Downside Deviation
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1:
        return np.inf  # No downside returns, Sortino is technically infinite

    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)

    if downside_deviation == 0:
        return np.inf

    sortino_ratio = (excess_returns.mean() * 252) / downside_deviation
    return sortino_ratio if not np.isnan(sortino_ratio) else 0.0