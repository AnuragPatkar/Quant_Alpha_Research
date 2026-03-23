"""
quant_alpha.visualization
===========================
Exports all public visualization symbols so run_backtest.py can use:

    from quant_alpha.visualization import (
        plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
        plot_ic_time_series, generate_tearsheet
    )
"""

from .plots   import plot_equity_curve, plot_drawdown, plot_monthly_heatmap
from .reports import generate_tearsheet
from .utils   import set_style, format_currency


def plot_ic_time_series(rolling_ic, save_path=None):
    """
    Plot a rolling IC time series.  Thin wrapper so run_backtest.py can
    import this from quant_alpha.visualization without knowing the module.

    Parameters
    ----------
    rolling_ic : pd.Series indexed by date.
    save_path  : Optional path to save PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_ic.index, rolling_ic.values, linewidth=1.5, label="Rolling IC")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling Information Coefficient", fontweight="bold")
    ax.set_ylabel("Spearman IC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_ic_time_series",
    "generate_tearsheet",
    "set_style",
    "format_currency",
]