"""
quant_alpha/visualization/reports.py
=======================================
Backtest Reporting & Tearsheet Generation.

This is the CORRECT location for reports.py per the project structure.
The file at the project root (reports.py) was placed in the wrong directory
and had a broken `from .utils import ...` relative import.

Provides:
  - generate_tearsheet()    : multi-page PDF tearsheet
  - print_metrics_report()  : re-exported from backtest.metrics for convenience
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

# FIX: correct relative import — utils.py lives in the same visualization/ package
from .utils import set_style, format_currency


def generate_tearsheet(
    results: Dict[str, Any],
    save_path: str = "tearsheet.pdf",
) -> None:
    """
    Compile a multi-page PDF tearsheet from backtest results.

    Pages:
    1. Equity Curve
    2. Drawdown Profile
    3. Monthly Returns Heatmap
    4. Performance Metrics Summary

    Parameters
    ----------
    results   : Dict returned by BacktestEngine.run().
                Must contain 'equity_curve' and 'metrics'.
    save_path : Output path (.pdf or .png).
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
        hwm     = nav.cummax()
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