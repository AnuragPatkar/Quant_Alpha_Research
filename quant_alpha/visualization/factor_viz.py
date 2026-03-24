"""
Factor Analysis Visualization Suite
===================================

Provides standardized rendering routines for alpha factor performance diagnostics 
and predictive power evaluation.

Purpose
-------
This module generates canonical visualizations used throughout the quantitative 
research lifecycle to evaluate the out-of-sample efficacy of alpha factors. 
It focuses on two critical diagnostic pillars:
1. **Predictive Power Stability**: Tracking the Information Coefficient (IC) trajectory.
2. **Signal Monotonicity**: Assessing the structural relationship between factor 
   quantiles and forward asset returns.

Role in Quantitative Workflow
-----------------------------
Serves as the primary graphical interface for the factor validation engines. 
These plots are essential for researchers to visually identify alpha decay, 
regime-conditional factor inversion, and non-linear target mapping prior to 
ensemble model inclusion.

Mathematical Dependencies
-------------------------
- **Pandas**: Vectorized time-series manipulations, expanding window alignments, 
  and cross-sectional quantile grouping.
- **Matplotlib**: Primary rendering backend for static plot generation and 
  multi-axis assembly.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from .utils import set_style

def plot_ic_time_series(ic_series: pd.Series, window: int = 20, save_path: Optional[str] = None):
    """
    Visualizes the Information Coefficient (IC) time series and its rolling average.

    Constructs a dual-layer plot mapping granular daily predictive power against 
    its structural moving average trend. The IC represents the cross-sectional 
    Spearman Rank Correlation between the factor values and forward asset returns. 
    A consistently positive IC with low variance indicates a robust alpha signal.

    Args:
        ic_series (pd.Series): A daily time-series vector mapping Information Coefficients.
        window (int, optional): The rolling window lookback period (in days) for the 
            moving average trendline extraction. Defaults to 20.
        save_path (Optional[str], optional): The filepath destination for the rendered 
            figure. If None, the plot is rendered interactively. Defaults to None.
            
    Returns:
        None: Renders the plot via matplotlib backend or saves directly to disk.
    """
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Renders discrete daily IC observations as a bar distribution to highlight 
    # granular high-frequency noise and empirical variance.
    plt.bar(ic_series.index, ic_series, color='gray', alpha=0.3, label='Daily IC', width=1.0)
    
    # Extracts the underlying structural signal stability by smoothing the vector 
    # via a rolling mean aggregation window.
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
    Generates a discrete bar chart of mean forward returns mapped by factor quantile.

    Validates strict signal monotonicity. An optimal cross-sectional ranking factor 
    should exhibit a structurally monotonic increase in returns traversing from the 
    lowest quantile (Q1) to the highest quantile (QN). Non-monotonic "humped" shapes 
    typically indicate a flawed signal or non-linear target relationship.

    Args:
        quantile_returns (pd.Series): A cross-sectionally grouped series where the 
            index represents the sorted quantile bins and the values map the mean 
            forward asset returns.
        save_path (Optional[str], optional): The filepath destination for the rendered 
            figure. If None, the plot is rendered interactively. Defaults to None.
            
    Returns:
        None: Renders the plot via matplotlib backend or saves directly to disk.
    """
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Projects the relative spread mapping across rank buckets to visually assess 
    # the linearity of the alpha distribution.
    quantile_returns.plot(kind='bar', color='skyblue', edgecolor='black', ax=plt.gca())
    
    # Computes the empirical Long/Short spread differential (Top Quantile - Bottom Quantile) 
    # to quantify the theoretical maximal alpha extraction boundary.
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