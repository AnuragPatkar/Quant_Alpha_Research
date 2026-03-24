"""
Visualization Utilities
=====================

Provides shared styling and formatting heuristics for the quantitative 
visualization and reporting suite.

Purpose
-------
This module establishes canonical formatting rules, ensuring strict 
aesthetic consistency across interactive dashboards, static tearsheets, 
and factor analysis plots. 

Role in Quantitative Workflow
-----------------------------
Acts as the foundational styling configuration for all graphical rendering 
engines, abstracting away backend-specific boilerplate to maintain a clean 
and professional institutional presentation layer.

Mathematical Dependencies
-------------------------
- **Matplotlib**: Primary rendering backend for static plot generation and 
  aesthetic configuration.
"""

import matplotlib.pyplot as plt
from typing import Optional


def set_style() -> None:
    """
    Applies a clean, publication-ready configuration to the Matplotlib backend.
    
    Attempts to establish a standardized Seaborn-derived darkgrid style. 
    Incorporates nested failovers to guarantee visual consistency across 
    varying execution environments and Matplotlib versions.
    
    Returns:
        None: Globally mutates the current Matplotlib state map.
    """
    # Employs nested resolution to capture distinct internal naming conventions 
    # established in Matplotlib >= 3.6 vs older legacy environments.
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

    Designed strictly as a callback interface for Matplotlib's `FuncFormatter` 
    to structurally align axis ticks mapping gross portfolio wealth or 
    capital allocation metrics.

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