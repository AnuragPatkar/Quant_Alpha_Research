"""
Visualization tools for Alpha Factors.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from .utils import set_style

def plot_ic_time_series(ic_series: pd.Series, window: int = 20, save_path: Optional[str] = None):
    """Plot Information Coefficient (IC) over time with moving average."""
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Plot daily IC as bars
    plt.bar(ic_series.index, ic_series, color='gray', alpha=0.3, label='Daily IC', width=1.0)
    
    # Plot moving average
    rolling_mean = ic_series.sort_index().rolling(window).mean()
    plt.plot(rolling_mean.index, rolling_mean, color='blue', label=f'{window}-day Moving Avg', linewidth=2)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title('Information Coefficient (IC) Over Time')
    plt.xlabel('Date')
    plt.ylabel('IC')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_quantile_returns(quantile_returns: pd.Series, save_path: Optional[str] = None):
    """Plot mean returns by factor quantile."""
    set_style()
    plt.figure(figsize=(10, 6))
    
    # Explicitly use current axes to respect figsize
    quantile_returns.plot(kind='bar', color='skyblue', edgecolor='black', ax=plt.gca())
    
    plt.title('Mean Return by Factor Quantile')
    plt.xlabel('Quantile')
    plt.ylabel('Mean Forward Return')
    plt.axhline(0, color='black', linewidth=0.8)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()