"""
Factor Analysis Visualization Suite
===================================
Standardized plotting routines for alpha factor performance diagnostics.

Purpose
-------
This module provides a set of canonical visualizations used throughout the
quantitative research process to evaluate the efficacy of alpha factors. It
focuses on two key diagnostics:
1.  **Predictive Power Stability**: Visualizing the Information Coefficient (IC) over time.
2.  **Signal Monotonicity**: Assessing the relationship between factor quantiles and returns.

Usage
-----
These functions are typically called from a `FactorAnalyzer` instance or directly
within a research notebook.

.. code-block:: python

    from quant_alpha.research import FactorAnalyzer
    from quant_alpha.visualization import plot_ic_time_series, plot_quantile_returns

    analyzer = FactorAnalyzer(data, factor_col='rsi', target_col='ret_5d')
    
    # Plot IC stability
    ic_series = analyzer.calculate_ic()
    plot_ic_time_series(ic_series, window=22)

    # Plot signal monotonicity
    q_returns, _ = analyzer.calculate_quantile_returns()
    plot_quantile_returns(q_returns)

Importance
----------
-   **IC Time Series**: A stable, positive IC is the hallmark of a predictive factor.
    This plot quickly reveals IC decay, regime sensitivity, or structural breaks.
-   **Quantile Returns**: A monotonic increase in returns from the lowest quantile (Q1)
    to the highest (QN) confirms the factor correctly ranks assets. A non-monotonic
    "humped" shape can indicate a flawed signal.

Tools & Frameworks
------------------
-   **Matplotlib/Seaborn**: Core libraries for static plot generation.
-   **Pandas**: Data manipulation for time-series and quantile data.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from .utils import set_style

def plot_ic_time_series(ic_series: pd.Series, window: int = 20, save_path: Optional[str] = None):
    """
    Visualizes the Information Coefficient (IC) time series and its rolling average.

    This plot helps assess the stability and consistency of a factor's predictive power.
    A consistently positive IC with low volatility is desirable.

    Args:
        ic_series (pd.Series): Time-series of daily Information Coefficients.
        window (int): The rolling window period for the moving average trend line.
        save_path (Optional[str]): If provided, saves the plot to this file path.
    """
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Render daily IC as bars to show daily noise and variance.
    plt.bar(ic_series.index, ic_series, color='gray', alpha=0.3, label='Daily IC', width=1.0)
    
    # A rolling average trendline reveals the underlying signal stability.
    rolling_mean = ic_series.sort_index().rolling(window).mean()
    plt.plot(rolling_mean.index, rolling_mean, color='blue', label=f'{window}-day Moving Avg', linewidth=2)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Information Coefficient (IC) Over Time: {ic_series.name}')
    plt.xlabel('Date')
    plt.ylabel('IC')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_quantile_returns(quantile_returns: pd.Series, save_path: Optional[str] = None):
    """
    Generates a bar chart of mean forward returns for each factor quantile.

    This is a critical test for signal monotonicity. An effective ranking factor
    should exhibit a clear, monotonic trend in returns from the lowest quantile
    (Q1) to the highest (QN).

    Args:
        quantile_returns (pd.Series): A Series where the index represents the
                                      quantile and values are the mean forward returns.
        save_path (Optional[str]): If provided, saves the plot to this file path.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Use a bar chart to clearly visualize the spread between quantiles.
    # The ax=plt.gca() ensures the plot respects the figure's pre-defined size.
    quantile_returns.plot(kind='bar', color='skyblue', edgecolor='black', ax=plt.gca())
    
    # Calculate spread for title
    spread = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
    plt.title(f'Mean Return by Factor Quantile (Spread: {spread:.4f})')
    plt.xlabel('Quantile')
    plt.ylabel('Mean Forward Return')
    plt.axhline(0, color='black', linewidth=0.8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()