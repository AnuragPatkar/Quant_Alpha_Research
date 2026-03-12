"""
Static Performance Visualization Library
========================================
Standardized plotting routines for reporting portfolio metrics.

Purpose
-------
This module generates high-quality static charts for the core components of a
financial tearsheet: Equity Curves, Drawdown Profiles, and Monthly Return Heatmaps.
Unlike interactive plots, these are optimized for PDF reports and fixed-layout
presentations.

Usage
-----
Intended for use within the `generate_tearsheet` workflow or for ad-hoc
analysis in research notebooks.

.. code-block:: python

    # Compare Strategy vs Benchmark
    plot_equity_curve(
        equity_df=portfolio_nav,
        benchmark_df=spy_data,
        save_path='equity_curve.png'
    )

    # Analyze Risk Profile
    plot_drawdown(equity_df=portfolio_nav)

Importance
----------
-   **Performance Attribution**: Overlays the strategy against a benchmark to visually
    assess Alpha generation ($\alpha$) and Beta exposure ($\beta$).
-   **Risk Visualization**: The "Underwater" drawdown plot highlights the depth and
    duration of recovery periods, critical for psychological tolerance assessment.
-   **Seasonality Detection**: Monthly heatmaps reveal calendar effects or regime
    dependencies (e.g., poor performance in volatile months).

Tools & Frameworks
------------------
-   **Matplotlib**: Low-level engine for low-latency plotting.
-   **Seaborn**: High-level API for generating the monthly return matrix heatmap.
-   **Pandas**: Time-series resampling (compounding) and index alignment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from .utils import set_style, format_currency

def plot_equity_curve(equity_df, benchmark_df=None, save_path=None):
    """
    Visualizes the Portfolio Net Asset Value (NAV) trajectory against a Benchmark.
    
    Performs time-series alignment and normalization to ensure both series start
    at the same capital base ($V_{t=0}$), allowing for direct relative comparison.

    Args:
        equity_df (pd.DataFrame): Portfolio time-series containing 'date' and 'total_value'.
        benchmark_df (Optional[pd.DataFrame]): Benchmark time-series containing 'date' and 'close'.
        save_path (Optional[str]): File path to persist the chart image.
    """
    set_style()
    plt.figure(figsize=(12, 6))
    
    # Type Enforcement: Coerce dates to datetime objects for correct axis plotting
    equity_df = equity_df.copy()
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    plt.plot(equity_df['date'], equity_df['total_value'], label='Portfolio', linewidth=2)
    
    if benchmark_df is not None:
        # Benchmark Alignment: Normalize benchmark series to match portfolio starting capital.
        # V_bench_norm(t) = (P_bench(t) / P_bench(0)) * V_port(0)
        initial_value = equity_df['total_value'].iloc[0]
        
        # Data Synchronization: Inner join on dates to align time-series
        bench_df = benchmark_df.copy()
        bench_df['date'] = pd.to_datetime(bench_df['date'])
        merged = pd.merge(equity_df[['date']], bench_df[['date', 'close']], on='date', how='left')
        
        # Normalization Logic
        if not merged['close'].empty:
            # Handle edge case where first price is NaN or Zero
            start_price = merged['close'].iloc[0] if not pd.isna(merged['close'].iloc[0]) else 1.0
            bench_norm = merged['close'] / start_price * initial_value
            plt.plot(merged['date'], bench_norm, label='Benchmark', alpha=0.7, linestyle='--')
        
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    
    # Visualization: Apply currency formatter ($1M, $1B) to Y-axis
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_drawdown(equity_df, save_path=None):
    """
    Generates an "Underwater" plot representing the Drawdown profile.
    
    .. math:: DD_t = \\frac{NAV_t - HWM_t}{HWM_t}
    
    Where $HWM_t$ is the High Water Mark (running maximum) up to time $t$.
    Shaded red areas indicate periods where the portfolio is below its peak.
    """
    set_style()
    equity = equity_df['total_value']
    
    # Vectorized Drawdown Calculation
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
    """
    Visualizes the Calendar Performance Matrix.
    
    Aggregates daily returns into monthly geometric returns and displays them
    in a Year x Month grid. This aids in identifying seasonality or structural
    issues in specific calendar months.
    
    .. math:: R_{month} = \\prod_{t \\in month} (1 + r_t) - 1
    """
    set_style()
    
    # Resampling: Convert daily returns to Monthly Geometric Returns (Compounded)
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