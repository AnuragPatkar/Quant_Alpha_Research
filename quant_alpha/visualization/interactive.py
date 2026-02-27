"""
Interactive dashboards using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_interactive_equity(equity_df, trades_df=None):
    """
    Create an interactive equity curve with trade markers.
    
    Args:
        equity_df: DataFrame with 'date' and 'total_value'
        trades_df: Optional DataFrame with 'date', 'side' ('buy'/'sell'), 'ticker'
    """
    fig = go.Figure()
    
    # Equity line
    fig.add_trace(go.Scatter(
        x=equity_df['date'], 
        y=equity_df['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add trades if available
    if trades_df is not None and not trades_df.empty:
        buys = trades_df[trades_df['side'].str.lower() == 'buy']
        sells = trades_df[trades_df['side'].str.lower() == 'sell']
        
        # Map trade dates to equity values for y-axis placement
        # Note: This assumes trade dates exist in equity_df. 
        # In production, might need merge_asof or interpolation.
        
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['date'],
                y=[equity_df.loc[equity_df['date'] == d, 'total_value'].iloc[0] if d in equity_df['date'].values else None for d in buys['date']],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy',
                text=buys['ticker']
            ))
        
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['date'],
                y=[equity_df.loc[equity_df['date'] == d, 'total_value'].iloc[0] if d in equity_df['date'].values else None for d in sells['date']],
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