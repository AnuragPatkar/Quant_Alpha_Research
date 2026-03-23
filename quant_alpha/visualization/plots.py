"""
quant_alpha/visualization/plots.py
=====================================
Standard static plotting routines for the backtest tearsheet.

FIXED: `from .utils import set_style, format_currency` now correctly
resolves within the quant_alpha/visualization/ package.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from typing import Optional

# Correct relative import — utils.py is in the same visualization/ package
from .utils import set_style, format_currency


def plot_equity_curve(
    equity_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the portfolio NAV trajectory, optionally vs a benchmark.

    Parameters
    ----------
    equity_df    : DataFrame with columns ['date', 'total_value'].
    benchmark_df : Optional DataFrame with columns ['date', 'close'].
    save_path    : Path to save PNG; if None, displays interactively.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])

    ax.plot(eq["date"], eq["total_value"], label="Portfolio", linewidth=2)

    if benchmark_df is not None:
        initial_value = float(eq["total_value"].iloc[0])
        bench = benchmark_df.copy()
        bench["date"] = pd.to_datetime(bench["date"])
        merged = pd.merge(eq[["date"]], bench[["date", "close"]], on="date", how="left")

        if not merged["close"].empty:
            start_price = (
                float(merged["close"].dropna().iloc[0])
                if not merged["close"].dropna().empty
                else 1.0
            )
            if start_price > 0:
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
    Generate an underwater drawdown chart.

    DD_t = (NAV_t - HWM_t) / HWM_t
    """
    set_style()
    nav    = equity_df["total_value"].values.astype(float)
    hwm    = np.maximum.accumulate(nav)
    safe   = np.where(hwm > 1e-12, hwm, 1e-12)
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
    Plot a calendar monthly return heatmap.

    Compounds daily returns to monthly geometric returns:
        R_month = prod(1 + R_daily) - 1
    """
    set_style()
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