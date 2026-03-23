"""
quant_alpha/visualization/utils.py
=====================================
Shared styling and formatting helpers for all visualization modules
(plots.py, reports.py, factor_viz.py, interactive.py).

Kept identical to quant_alpha/backtest/utils.py — both resolve the same
`from .utils import set_style, format_currency` relative import inside
their respective subpackages.
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
            pass


def format_currency(x, pos: Optional[int] = None) -> str:
    """
    Format a numeric axis tick as a dollar amount.

    Used as FuncFormatter callback:
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))
    """
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"