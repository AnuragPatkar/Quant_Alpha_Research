"""
Standard static plots using Matplotlib and Seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .utils import set_style

def plot_equity_curve(equity_df, benchmark_df=None, save_path=None):
    """Plot portfolio equity curve vs benchmark."""
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Ensure date is datetime
    dates = pd.to_datetime(equity_df['date'])
    
    plt.plot(dates, equity_df['total_value'], label='Portfolio', linewidth=2)
    
    if benchmark_df is not None:
        # Normalize benchmark to start at portfolio initial value
        initial_value = equity_df['total_value'].iloc[0]
        bench_dates = pd.to_datetime(benchmark_df['date'])
        
        # Align dates if needed, simple normalization here
        bench_norm = benchmark_df['close'] / benchmark_df['close'].iloc[0] * initial_value
        plt.plot(bench_dates, bench_norm, label='Benchmark', alpha=0.7, linestyle='--')
        
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_drawdown(equity_df, save_path=None):
    """Plot underwater drawdown curve."""
    set_style()
    equity = equity_df['total_value']
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    dates = pd.to_datetime(equity_df['date'])
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
    plt.plot(dates, drawdown, color='red', linewidth=1)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_monthly_heatmap(returns_series, save_path=None):
    """Plot monthly returns heatmap."""
    # Implementation handled in reports.py or can be added here if needed standalone
    # For now, we rely on the implementation inside reports for PDF generation
    # or interactive dashboards.
    pass