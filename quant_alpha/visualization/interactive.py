"""
Interactive Performance Dashboards
==================================

Provides dynamic, interactive visualization suites for portfolio analysis 
and trade execution attribution.

Purpose
-------
This module generates interactive HTML-based charts enabling granular 
inspection of portfolio performance over discrete time horizons. It overlays 
discrete trade execution markers onto continuous Net Asset Value (NAV) 
time-series to visually correlate execution timing with equity inflections.

Role in Quantitative Workflow
-----------------------------
Serves as an ex-post trade attribution and diagnostic tool. By visualizing 
exact Buy/Sell coordinates against the capital curve, researchers can audit 
execution efficiency, identify sub-optimal entries/exits (e.g., trading at 
local maxima/minima), and assess slippage impacts interactively.

Mathematical Dependencies
-------------------------
- **Plotly Graph Objects**: Low-level interface for composing complex, 
  multi-layered interactive vector graphics.
- **Pandas**: Time-series alignment, vectorized temporal normalization, 
  and $O(1)$ hash map construction for cross-sectional overlay.
"""

import plotly.graph_objects as go
import pandas as pd

def plot_interactive_equity(equity_df, trades_df=None):
    """
    Generates an interactive Plotly figure mapping trade executions to capital trajectory.
    
    Args:
        equity_df (pd.DataFrame): Time-series matrix of portfolio NAV. 
            Strictly requires 'date' and 'total_value' columns.
        trades_df (Optional[pd.DataFrame]): Ledger of discrete executed trades. 
            Strictly requires 'date', 'side', and 'ticker' columns. Defaults to None.
            
    Returns:
        go.Figure: An interactive, multi-layered Plotly chart object.
    """
    fig = go.Figure()
    
    # Renders the primary continuous geometric wealth trajectory
    fig.add_trace(go.Scatter(
        x=equity_df['date'], 
        y=equity_df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='$%{y:,.2f}'
    ))
    
    if trades_df is not None and not trades_df.empty:
        # Validates strict dependency schema for execution overlays
        if not all(col in trades_df.columns for col in ['date', 'side', 'ticker']):
            return fig
            
        # Isolates execution directionality via vectorized string matching
        buys = trades_df[trades_df['side'].astype(str).str.lower() == 'buy']
        sells = trades_df[trades_df['side'].astype(str).str.lower() == 'sell']
        
        # Constructs an O(1) temporal hash map strictly mapping execution dates to discrete NAV values, 
        # bypassing computationally expensive O(N*M) continuous matrix lookups during rendering.
        equity_dates = pd.to_datetime(equity_df['date']).dt.normalize()
        date_val_map = dict(zip(equity_dates, equity_df['total_value']))
        
        if not buys.empty:
            buy_dates = pd.to_datetime(buys['date']).dt.normalize()
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