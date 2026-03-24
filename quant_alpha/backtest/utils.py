"""
Backtest Formatting Utilities
=============================

Provides shared styling and formatting heuristics for performance metrics 
and backtest reporting.

Purpose
-------
This module establishes canonical formatting rules to ensure strict 
aesthetic consistency across textual and graphical backtest outputs.

Mathematical Dependencies
-------------------------
- **Matplotlib**: Primary rendering backend configuration.
"""

import matplotlib.pyplot as plt
from typing import Optional


def set_style() -> None:
    """
    Applies a clean, publication-ready configuration to the Matplotlib backend.
    
    Returns:
        None: Mutates global matplotlib runtime configuration.
    """
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            pass


def format_currency(x, pos: Optional[int] = None) -> str:
    """
    Formats absolute numeric scalars into human-readable currency strings.
    
    Args:
        x (float): The raw continuous numeric axis value to be formatted.
        pos (Optional[int]): The axis tick position mapping passed implicitly 
            by the Matplotlib formatting engine. Defaults to None.
            
    Returns:
        str: A discrete formatted string abbreviation mapped to standard 
            financial prefixes (e.g., $1.5M, $500K).
    """
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:.1f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:.0f}"