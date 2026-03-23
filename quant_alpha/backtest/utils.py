"""
quant_alpha/backtest/utils.py
==============================
Shared styling and formatting helpers used by backtest.metrics, backtest.attribution,
visualization.plots, and visualization.reports.

Keep this module import-side-effect free — it only defines functions.
"""

import matplotlib.pyplot as plt
from typing import Optional


def set_style() -> None:
    """Apply a clean, publication-ready matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            pass   # fall back to matplotlib default silently


def format_currency(x, pos: Optional[int] = None) -> str:
    """
    Format a numeric axis tick as a dollar amount with commas.

    Used as a FuncFormatter callback:
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
    """
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"