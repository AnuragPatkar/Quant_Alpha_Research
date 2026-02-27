"""
Generate PDF/HTML reports for backtest results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from .utils import set_style

def generate_tearsheet(results, save_path='tearsheet.pdf'):
    """Generate a PDF tearsheet with key metrics and plots."""
    set_style()
    equity_df = results['equity_curve']
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    
    # Calculate returns if not present
    if 'return' not in equity_df.columns:
        equity_df['return'] = equity_df['total_value'].pct_change()
    
    returns_series = equity_df.set_index('date')['return']
    
    with PdfPages(save_path) as pdf:
        # Page 1: Equity Curve
        plt.figure(figsize=(10, 6))
        plt.plot(equity_df['date'], equity_df['total_value'])
        plt.title('Equity Curve')
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # Page 2: Drawdown
        plt.figure(figsize=(10, 6))
        equity = equity_df['total_value']
        dd = (equity - equity.cummax()) / equity.cummax()
        plt.fill_between(equity_df['date'], dd, 0, color='red', alpha=0.3)
        plt.plot(equity_df['date'], dd, color='red', linewidth=1)
        plt.title('Drawdown')
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # Page 3: Monthly Heatmap
        monthly_ret = returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_ret = monthly_ret.to_frame(name='return')
        monthly_ret['year'] = monthly_ret.index.year
        monthly_ret['month'] = monthly_ret.index.month
        pivot = monthly_ret.pivot(index='year', columns='month', values='return')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
        plt.title('Monthly Returns')
        pdf.savefig()
        plt.close()
        
        # Page 4: Metrics Text
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        metrics = results.get('metrics', {})
        text = "Performance Metrics\n\n"
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            text += f"{k}: {val_str}\n"
        plt.text(0.1, 0.9, text, fontsize=12, verticalalignment='top', fontfamily='monospace')
        pdf.savefig()
        plt.close()