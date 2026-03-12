"""
Backtest Reporting & Tearsheet Generation
=========================================
Automated generation of institutional-grade performance reports.

Purpose
-------
The `reports` module aggregates the results of a backtest simulation into a
cohesive, portable document (PDF). It functions as the final step in the
research pipeline, transforming raw time-series data into actionable intelligence
for Investment Committee review.

Usage
-----
Intended to be called at the end of a backtest execution via the `BacktestEngine`.

.. code-block:: python

    # Run Backtest
    results = engine.run(predictions, prices)

    # Generate Tearsheet
    generate_tearsheet(
        results=results,
        save_path='reports/strategy_v1_tearsheet.pdf'
    )

Importance
----------
-   **Standardization**: Enforces a consistent reporting format across all strategies,
    facilitating apples-to-apples comparison of Sharpe ratios, Drawdowns, and Alpha.
-   **Portability**: The PDF output serves as an immutable record of a strategy's
    performance at a specific point in time (Snapshot), crucial for audit trails.

Tools & Frameworks
------------------
-   **Matplotlib (PdfPages)**: Backend for multi-page PDF composition.
-   **Seaborn**: Heatmap visualization for monthly return attribution.
-   **Pandas**: Time-series resampling and pivot table construction.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from .utils import set_style, format_currency

def generate_tearsheet(results: Dict[str, Any], save_path: str = 'tearsheet.pdf'):
    """
    Compiles a Multi-Page PDF Tearsheet from backtest results.

    Report Structure:
    1.  **Equity Curve**: Visualizes $NAV_t$ trajectory.
    2.  **Drawdown Profile**: "Underwater" plot highlighting risk depth/duration.
    3.  **Monthly Returns**: Heatmap of calendar performance.
    4.  **Key Metrics**: Tabular summary of CAGR, Sharpe, Sortino, etc.

    Args:
        results (Dict[str, Any]): Output dictionary from `BacktestEngine.run()`.
                                  Must contain keys: `'equity_curve'`, `'metrics'`.
        save_path (str): Filesystem path for the output PDF.
    """
    set_style()
    equity_df = results['equity_curve'].copy()
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    # Data Preparation: Derive daily returns if not pre-calculated
    # R_t = (NAV_t / NAV_{t-1}) - 1
    if 'return' not in equity_df.columns:
        equity_df['return'] = equity_df['total_value'].pct_change()
    
    returns_series = equity_df.set_index('date')['return']
    
    with PdfPages(save_path) as pdf:
        # --- Page 1: Equity Curve (NAV Trajectory) ---
        plt.figure(figsize=(10, 6))
        plt.plot(equity_df['date'], equity_df['total_value'])
        plt.title('Equity Curve')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()

        # --- Page 2: Drawdown Profile (Underwater Plot) ---
        plt.figure(figsize=(10, 6))
        equity = equity_df['total_value']
        
        # Vectorized Max Drawdown Calculation
        # DD_t = (NAV_t - HWM_t) / HWM_t
        dd = (equity - equity.cummax()) / equity.cummax()
        
        plt.fill_between(equity_df['date'], dd, 0, color='red', alpha=0.3)
        plt.plot(equity_df['date'], dd, color='red', linewidth=1)
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # --- Page 3: Monthly Return Attribution (Heatmap) ---
        # Resample daily returns to monthly geometric returns: R_m = \prod(1+r_d) - 1
        monthly_ret = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        plt.figure(figsize=(10, 6))
        if not monthly_ret.empty:
            # Pivot Transformation: Index=Year, Columns=Month
            monthly_ret = monthly_ret.to_frame(name='return')
            monthly_ret['year'] = monthly_ret.index.year
            monthly_ret['month'] = monthly_ret.index.month
            pivot = monthly_ret.pivot(index='year', columns='month', values='return')
            
            if not pivot.empty and pivot.notna().any().any():
                sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
                plt.title('Monthly Returns')
            else:
                plt.text(0.5, 0.5, 'Insufficient Data for Heatmap', ha='center', va='center')
                plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            plt.axis('off')
            
        pdf.savefig()
        plt.close()
        
        # --- Page 4: Performance Metrics Summary ---
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        metrics = results.get('metrics', {})
        if metrics:
            text = "Performance Metrics\n\n"
            # Formatting: Render floats with 4 decimals, others as strings
            for k, v in metrics.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                text += f"{k}: {val_str}\n"
            plt.text(0.1, 0.9, text, fontsize=12, verticalalignment='top', fontfamily='monospace')
        else:
             plt.text(0.5, 0.5, 'No Metrics Available', ha='center', va='center')
             
        pdf.savefig()
        plt.close()