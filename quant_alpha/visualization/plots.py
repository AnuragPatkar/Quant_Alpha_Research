"""
Standardized Backtest Visualization Routines
==========================================

Provides institutional-grade static plotting functions for strategy 
performance evaluation and tearsheet generation.

Purpose
-------
This module generates publication-ready visualizations of key quantitative 
metrics, including continuous equity trajectories, underwater drawdown 
profiles, and aggregate regime performance (monthly heatmaps). It enforces 
consistent styling and normalizes benchmark comparisons.

Role in Quantitative Workflow
-----------------------------
Acts as the primary graphical rendering engine for the backtesting pipeline. 
It translates raw portfolio execution data (time-series NAV and returns) 
into diagnostic charts used by researchers for alpha evaluation.

Mathematical Dependencies
-------------------------
- **Matplotlib/Seaborn**: Core rendering backends for static plot generation.
- **NumPy/Pandas**: Vectorized time-series compounding, alignment, and maximum 
  drawdown boundary calculations.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from typing import Optional

from .utils import set_style, format_currency


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots the portfolio Net Asset Value (NAV) trajectory, optionally normalized against a benchmark.

    Normalizes the benchmark to strictly align with the portfolio's initial 
    starting capital, providing a direct comparative analysis of relative 
    wealth generation over the backtest horizon.

    Args:
        equity_df (pd.DataFrame): Time-series dataframe containing at least 
            'date' and 'total_value' columns mapping capital levels.
        benchmark_df (Optional[pd.DataFrame]): Optional time-series dataframe 
            containing 'date' and 'close' price columns for the benchmark asset.
        save_path (Optional[str]): Filepath to save the generated figure. 
            If None, the plot is rendered interactively.
            
    Returns:
        None: Renders the plot via matplotlib backend or saves directly to disk.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])

    ax.plot(eq["date"], eq["total_value"], label="Portfolio", linewidth=2)

    if benchmark_df is not None:
        initial_value = float(eq["total_value"].iloc[0])
        bench = benchmark_df.copy()
        
        # Maps time-series indices to standard datetime structures for continuous alignment
        bench["date"] = pd.to_datetime(bench["date"])
        merged = pd.merge(eq[["date"]], bench[["date", "close"]], on="date", how="left")

        if not merged["close"].empty:
            start_price = (
                float(merged["close"].dropna().iloc[0])
                if not merged["close"].dropna().empty
                else 1.0
            )
            if start_price > 0:
                # Dynamically rebases the benchmark trajectory to the portfolio's inception capital
                # to ensure strictly proportional geometric scaling on the comparative Y-axis.
                bench_norm = merged["close"] / start_price * initial_value
                ax.plot(merged["date"], bench_norm,
                        label="Benchmark", alpha=0.7, linestyle="--")

    ax.set_title("Portfolio Equity Curve", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_drawdown(
    equity_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Generates an underwater drawdown profile tracking capital peak-to-trough regressions.

    Calculates continuous drawdown from the High-Water Mark (HWM). 
    Employs a safe-division scalar to prevent zero-division instability.

    Args:
        equity_df (pd.DataFrame): Time-series dataframe containing 'date' 
            and 'total_value' columns.
        save_path (Optional[str]): Filepath to save the generated figure. 
            If None, the plot is rendered interactively.

    Returns:
        None: Renders the plot via matplotlib backend or saves directly to disk.
    """
    set_style()
    nav    = equity_df["total_value"].values.astype(float)
    # Computes the High-Water Mark (HWM) via an expanding maximum array calculation
    hwm    = np.maximum.accumulate(nav)
    # Injects a 1e-12 boundary floor to guarantee numerical stability during fraction division
    safe   = np.where(hwm > 1e-12, hwm, 1e-12)
    # Solves empirical drawdown: $DD_t = \frac{NAV_t - HWM_t}{HWM_t}$
    dd     = (nav - hwm) / safe
    dates  = pd.to_datetime(equity_df["date"])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, dd, 0, color="red", alpha=0.3)
    ax.plot(dates, dd, color="red", linewidth=1)
    ax.set_title("Portfolio Drawdown", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_monthly_heatmap(
    returns_series: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """
    Constructs a calendar-aligned heatmap mapping compounded monthly geometric returns.

    Translates granular daily return vectors into structural monthly buckets, 
    highlighting cyclicality, seasonality, or regime-specific strategy decay.

    Args:
        returns_series (pd.Series): A daily geometric returns series indexed by standard dates.
        save_path (Optional[str]): Filepath to save the generated figure. 
            If None, the plot is rendered interactively.
            
    Returns:
        None: Renders the plot via matplotlib backend or saves directly to disk.
    """
    set_style()
    # Compounds daily geometric returns ($R_{daily}$) into discrete monthly buckets
    # Formula: $R_{month} = \prod (1 + R_{daily}) - 1$
    monthly = returns_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    if not monthly.empty:
        mdf          = monthly.to_frame(name="return")
        mdf["year"]  = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot(index="year", columns="month", values="return")

        if not pivot.empty and pivot.notna().any().any():
            sns.heatmap(
                pivot, annot=True, fmt=".1%", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5,
            )
            ax.set_title("Monthly Returns", fontweight="bold")
            ax.set_ylabel("Year")
            ax.set_xlabel("Month")
        else:
            ax.text(0.5, 0.5, "Insufficient data for heatmap",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()