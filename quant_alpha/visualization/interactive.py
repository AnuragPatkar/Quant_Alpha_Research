"""
Interactive dashboards using Plotly.
"""

import plotly.graph_objects as go
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
        line=dict(color='#1f77b4', width=2),
        hovertemplate='$%{y:,.2f}'
    ))
    
    # Add trades if available
    if trades_df is not None and not trades_df.empty:
        # Validate required columns
        if not all(col in trades_df.columns for col in ['date', 'side', 'ticker']):
            return fig
            
        # Safe string conversion
        buys = trades_df[trades_df['side'].astype(str).str.lower() == 'buy']
        sells = trades_df[trades_df['side'].astype(str).str.lower() == 'sell']
        
        # Create a fast lookup map: Date -> Equity Value
        # This avoids repeated .loc lookups inside the loop
        # FIX: Normalize to pandas Timestamp to ensure type matching
        equity_dates = pd.to_datetime(equity_df['date']).dt.normalize()
        date_val_map = dict(zip(equity_dates, equity_df['total_value']))
        
        if not buys.empty:
            buy_dates = pd.to_datetime(buys['date']).dt.normalize()
            fig.add_trace(go.Scatter(
                x=buys['date'],
                y=[date_val_map.get(d, None) for d in buy_dates],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy',
                text=buys['ticker']
            ))
        
        if not sells.empty:
            sell_dates = pd.to_datetime(sells['date']).dt.normalize()
            fig.add_trace(go.Scatter(
                x=sells['date'],
                y=[date_val_map.get(d, None) for d in sell_dates],
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