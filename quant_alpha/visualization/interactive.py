"""
Interactive Performance Dashboards
==================================
Dynamic visualization suite for portfolio analysis using Plotly.

Purpose
-------
This module provides interactive HTML-based charts that allow for zooming,
panning, and granular inspection of portfolio performance. The primary
component is the `plot_interactive_equity` function, which overlays discrete
trade execution markers onto the continuous NAV time series, enabling
visual correlation of specific trades with equity inflections.

Usage
-----
Intended for use in Jupyter Notebooks or reporting dashboards.

.. code-block:: python

    # Create interactive plot with trade annotations
    fig = plot_interactive_equity(
        equity_df=portfolio.get_equity_curve_df(),
        trades_df=portfolio.get_tx_history_df()
    )
    fig.show()

Importance
----------
-   **Trade Attribution**: By marking Buy/Sell points directly on the curve,
    researchers can visually inspect if trades were executed near local minima/maxima.
-   **Data Granularity**: Hover tooltips reveal precise execution prices and
    portfolio values that static plots often obscure.

Tools & Frameworks
------------------
-   **Plotly Graph Objects**: Low-level interface for composing complex, multi-layered charts.
-   **Pandas**: Time-series alignment and vectorized data filtering.
"""

import plotly.graph_objects as go
import pandas as pd

def plot_interactive_equity(equity_df, trades_df=None):
    """
    Generates an interactive Plotly figure displaying the portfolio Equity Curve
    annotated with trade execution markers.
    
    Args:
        equity_df (pd.DataFrame): Time-series of portfolio NAV.
            Must contain columns: `['date', 'total_value']`.
        trades_df (Optional[pd.DataFrame]): Ledger of executed trades.
            Must contain columns: `['date', 'side', 'ticker']`.
            
    Returns:
        go.Figure: Interactive Plotly chart object.
    """
    fig = go.Figure()
    
    # Layer 1: Continuous Equity Curve (NAV)
    fig.add_trace(go.Scatter(
        x=equity_df['date'], 
        y=equity_df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='$%{y:,.2f}'
    ))
    
    # Layer 2: Discrete Trade Markers (if ledger provided)
    if trades_df is not None and not trades_df.empty:
        # Schema Validation
        if not all(col in trades_df.columns for col in ['date', 'side', 'ticker']):
            return fig
            
        # Vectorized Filtering: Segregate Buys and Sells for distinct styling
        buys = trades_df[trades_df['side'].astype(str).str.lower() == 'buy']
        sells = trades_df[trades_df['side'].astype(str).str.lower() == 'sell']
        
        # Optimization: O(1) Lookup Map Construction
        # We need to plot trade markers at the correct Y-axis height (Equity Value).
        # Instead of repeated DataFrame lookups (O(N*M)), we build a hash map.
        
        # 1. Normalize equity dates to midnight (remove time component)
        equity_dates = pd.to_datetime(equity_df['date']).dt.normalize()
        
        # 2. Map Date -> Portfolio Value
        date_val_map = dict(zip(equity_dates, equity_df['total_value']))
        
        if not buys.empty:
            # Normalize trade dates to align with the lookup map
            buy_dates = pd.to_datetime(buys['date']).dt.normalize()
            
            # Map trade dates to Y-values (Portfolio Value on that day)
            y_values = [date_val_map.get(d, None) for d in buy_dates]
            
            fig.add_trace(go.Scatter(
                x=buys['date'],
                y=y_values,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy',
                text=buys['ticker']
            ))
        
        if not sells.empty:
            sell_dates = pd.to_datetime(sells['date']).dt.normalize()
            y_values = [date_val_map.get(d, None) for d in sell_dates]
            
            fig.add_trace(go.Scatter(
                x=sells['date'],
                y=y_values,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell',
                text=sells['ticker']
            ))

    fig.update_layout(
        title='Interactive Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig