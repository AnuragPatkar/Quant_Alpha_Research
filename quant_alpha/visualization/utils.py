"""
Visualization utilities and styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Set default plotting style for the project."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5

def format_currency(x, pos):
    """Format axis labels as currency (K, M, B)."""
    sign = '-' if x < 0 else ''
    x = abs(x)
    
    if x >= 1e9:
        return f'{sign}${x*1e-9:1.1f}B'
    elif x >= 1e6:
        return f'{sign}${x*1e-6:1.1f}M'
    elif x >= 1e3:
        return f'{sign}${x*1e-3:1.0f}K'
    return f'{sign}${x:1.0f}'