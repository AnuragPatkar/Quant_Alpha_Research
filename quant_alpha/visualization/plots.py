"""
Standard static plots using Matplotlib and Seaborn.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from .utils import set_style, format_currency

def plot_equity_curve(equity_df, benchmark_df=None, save_path=None):
    """Plot portfolio equity curve vs benchmark."""
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Ensure date is datetime
    equity_df = equity_df.copy()
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    plt.plot(equity_df['date'], equity_df['total_value'], label='Portfolio', linewidth=2)
    
    if benchmark_df is not None:
        # Merge and normalize benchmark
        initial_value = equity_df['total_value'].iloc[0]
        
        # Align benchmark to portfolio dates
        bench_df = benchmark_df.copy()
        bench_df['date'] = pd.to_datetime(bench_df['date'])
        merged = pd.merge(equity_df[['date']], bench_df[['date', 'close']], on='date', how='left')
        
        # Normalize
        if not merged['close'].empty:
            start_price = merged['close'].iloc[0] if not pd.isna(merged['close'].iloc[0]) else 1.0
            bench_norm = merged['close'] / start_price * initial_value
            plt.plot(merged['date'], bench_norm, label='Benchmark', alpha=0.7, linestyle='--')
        
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Format Y-axis as Currency
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
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
    set_style()
    
    # Prepare data
    monthly_ret = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    plt.figure(figsize=(10, 6))
    
    if not monthly_ret.empty:
        monthly_ret = monthly_ret.to_frame(name='return')
        monthly_ret['year'] = monthly_ret.index.year
        monthly_ret['month'] = monthly_ret.index.month
        pivot = monthly_ret.pivot(index='year', columns='month', values='return')
        
        if not pivot.empty and pivot.notna().any().any():
            sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
            plt.title('Monthly Returns')
            plt.ylabel('Year')
            plt.xlabel('Month')
        else:
            plt.text(0.5, 0.5, 'Insufficient Data for Heatmap', ha='center', va='center')
            plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()