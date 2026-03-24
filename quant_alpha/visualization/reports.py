"""
Backtest Reporting & Tearsheet Generation.
=========================================

Provides institutional-grade PDF tearsheet compilation for quantitative backtest results.

Purpose
-------
This module aggregates continuous equity trajectories, rolling drawdown profiles, 
monthly regime heatmaps, and discrete performance metrics into a standardized, 
publication-ready multi-page PDF report.

Role in Quantitative Workflow
-----------------------------
Serves as the primary ex-post diagnostic output for the research and simulation 
pipeline. Standardized tearsheets ensure objective evaluation and comparison 
across diverse alpha strategies and parameter configurations.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Vectorized time-series manipulations, expanding maximum arrays, 
  and geometric return compounding.
- **Matplotlib/Seaborn**: Static backend rendering for multi-axis PDF page assembly.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from .utils import set_style, format_currency


def generate_tearsheet(
    results: Dict[str, Any],
    save_path: str = "tearsheet.pdf",
) -> None:
    """
    Compiles a multi-page PDF performance tearsheet from simulation outputs.

    Constructs four standardized diagnostic visualizations:
    1. Continuous Equity Curve tracking the geometric accumulation of capital.
    2. Underwater Drawdown Profile mapping capital peak-to-trough regressions.
    3. Calendar Heatmap mapping compounded monthly geometric returns.
    4. Text-based Performance Metrics Summary table.

    Args:
        results (Dict[str, Any]): The primary simulation output dictionary. Must contain:
            - 'equity_curve': A pd.DataFrame with 'date' and 'total_value' columns.
            - 'metrics': A dictionary mapping metric names to calculated scalar values.
        save_path (str, optional): The filepath destination for the rendered PDF report. 
            Defaults to "tearsheet.pdf".
            
    Returns:
        None: Directly saves the generated PDF report to disk via the Matplotlib backend.
        
    Raises:
        KeyError: If the 'equity_curve' matrix is missing from the results mapping.
    """
    set_style()

    equity_df = results["equity_curve"].copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    if "return" not in equity_df.columns:
        equity_df["return"] = equity_df["total_value"].pct_change()

    returns_series = equity_df.set_index("date")["return"]

    with PdfPages(save_path) as pdf:

        # --- Page 1: Equity Curve ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(equity_df["date"], equity_df["total_value"], linewidth=1.5)
        ax.set_title("Strategy Equity Curve", fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Drawdown ---
        fig, ax = plt.subplots(figsize=(10, 4))
        nav     = equity_df["total_value"]
        # Computes the expanding High-Water Mark (HWM) boundary
        hwm     = nav.cummax()
        # Resolves empirical drawdown fraction, injecting NaN substitution to strictly 
        # prevent zero-division instability during potential initialization states
        dd      = (nav - hwm) / hwm.replace(0, np.nan)
        ax.fill_between(equity_df["date"], dd, 0, color="red", alpha=0.3)
        ax.plot(equity_df["date"], dd, color="red", linewidth=1)
        ax.set_title("Drawdown Profile", fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 3: Monthly Heatmap ---
        fig, ax = plt.subplots(figsize=(12, 6))
        # Compounds granular daily geometric returns into discrete monthly buckets
        # Formula: $R_{month} = \prod (1 + R_{daily}) - 1$
        monthly = returns_series.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        if not monthly.empty:
            mdf            = monthly.to_frame(name="return")
            mdf["year"]    = mdf.index.year
            mdf["month"]   = mdf.index.month
            pivot = mdf.pivot(index="year", columns="month", values="return")
            if not pivot.empty and pivot.notna().any().any():
                sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn",
                            center=0, ax=ax, linewidths=0.5)
                ax.set_title("Monthly Returns", fontweight="bold")
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.axis("off")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 4: Metrics Table ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        metrics = results.get("metrics", {})
        if metrics:
            lines = ["Performance Metrics\n"]
            for k, v in metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"  {k:<35} {val_str}")
            ax.text(0.05, 0.95, "\n".join(lines), fontsize=11,
                    verticalalignment="top", fontfamily="monospace",
                    transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No metrics available", ha="center", va="center",
                    transform=ax.transAxes)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)