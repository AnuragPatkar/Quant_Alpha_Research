"""
Interactive Dashboard - ML Alpha Model
=======================================
Streamlit-based dashboard for backtest analysis.

Usage:
    streamlit run quant_alpha/visualization/dashboards.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import sys

# Streamlit & Plotly imports
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Install: pip install streamlit")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Install: pip install plotly")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Dashboard configuration."""
    
    COLORS = {
        'primary': '#00D4AA',
        'negative': '#FF6B6B',
        'warning': '#FFD93D',
        'info': '#4ECDC4',
        'background': '#0E1117',
        'card_bg': '#1E2130',
        'border': '#2D3748'
    }
    
    PAGE_CONFIG = {
        'page_title': '📊 ML Alpha Dashboard',
        'page_icon': '📊',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    CHART_HEIGHT = 450
    CHART_HEIGHT_SMALL = 350
    CHART_TEMPLATE = 'plotly_dark'
    
    PAGES = [
        ("📈", "Overview", "overview"),
        ("📊", "Performance", "performance"),
        ("🔬", "Factors", "factors"),
        ("💹", "Trades", "trades"),
        ("⚠️", "Risk", "risk"),
        ("📋", "Data", "data")
    ]


# =============================================================================
# DATA LOADING & METRIC CALCULATION UTILITIES
# =============================================================================

def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate missing metrics."""
    metrics = data.get('metrics', {})
    returns = data.get('returns', pd.Series())
    equity = data.get('equity_curve', pd.Series())
    trades = data.get('trades', pd.DataFrame())
    
    # Initialize all metrics to 0 or appropriate default if returns are empty
    if returns.empty:
        metrics.setdefault('cagr', 0.0)
        metrics.setdefault('sharpe_ratio', 0.0)
        metrics.setdefault('max_drawdown', 0.0)
        metrics.setdefault('win_rate', 0.0)
        metrics.setdefault('sortino_ratio', 0.0)
        metrics.setdefault('calmar_ratio', 0.0)
        metrics.setdefault('profit_factor', 0.0)
        metrics.setdefault('total_return', 0.0)
        metrics.setdefault('var_95', 0.0)
        metrics.setdefault('cvar_95', 0.0)
        metrics.setdefault('skewness', 0.0)
        metrics.setdefault('kurtosis', 0.0)
        metrics.setdefault('best_day', 0.0)
        metrics.setdefault('worst_day', 0.0)
        data['metrics'] = metrics
        return data
    
    # Normalize existing metrics (e.g., if percentage was string or > 1)
    metrics_mapping = {
        'total_return': ['total_return', 'Total Return', 'cumulative_return'],
        'cagr': ['cagr', 'CAGR', 'annualized_return', 'Annualized Return'],
        'sharpe_ratio': ['sharpe_ratio', 'Sharpe Ratio', 'sharpe'],
        'max_drawdown': ['max_drawdown', 'Max Drawdown', 'maximum_drawdown'],
        'volatility': ['volatility', 'Volatility', 'annual_volatility'],
        'win_rate': ['win_rate', 'Win Rate', 'hit_rate']
    }
    
    for standard_key, possible_keys in metrics_mapping.items():
        for key_in_dict in list(metrics.keys()): # Iterate over copy to allow modification
            if key_in_dict in possible_keys:
                value = metrics[key_in_dict]
                if isinstance(value, str) and '%' in value:
                    metrics[standard_key] = float(value.replace('%', '')) / 100
                elif isinstance(value, (int, float)):
                    # Assume metrics like CAGR, MaxDD, WinRate are stored as 0.XX for internal calcs
                    # If they come as e.g. 21.32 for 21.32%, convert to 0.2132
                    if standard_key in ['cagr', 'max_drawdown', 'total_return', 'win_rate'] and abs(value) > 1 and value > 0:
                        metrics[standard_key] = value / 100
                    else:
                        metrics[standard_key] = value
                break
    
    # Calculate Total Return if missing
    if metrics.get('total_return', 0) == 0 and not equity.empty and len(equity) > 1:
        metrics['total_return'] = equity.iloc[-1] / equity.iloc[0] - 1
    
    # Calculate CAGR if missing or 0
    if metrics.get('cagr', 0) == 0 and not equity.empty and len(equity) > 1 and metrics['total_return'] != 0:
        n_years = (equity.index.max() - equity.index.min()).days / 365.25
        if n_years > 0:
            metrics['cagr'] = (1 + metrics['total_return']) ** (1 / n_years) - 1
        else:
            metrics['cagr'] = 0.0
            
    # Calculate Max Drawdown if missing or 0
    if metrics.get('max_drawdown', 0) == 0 and not equity.empty and len(equity) > 1:
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        metrics['max_drawdown'] = drawdown.min() # This will be negative
    
    # Calculate Sharpe Ratio if missing or 0
    if metrics.get('sharpe_ratio', 0) == 0:
        volatility = returns.std() * np.sqrt(252)
        if volatility > 0:
            metrics['sharpe_ratio'] = (returns.mean() * 252) / volatility
        else:
            metrics['sharpe_ratio'] = 0.0
            
    # Calculate Win Rate if missing or 0
    if metrics.get('win_rate', 0) == 0:
        if not trades.empty and 'pnl' in trades.columns:
            wins = (trades['pnl'] > 0).sum()
            total_trades = len(trades)
            if total_trades > 0:
                metrics['win_rate'] = wins / total_trades
            else:
                metrics['win_rate'] = 0.0
        else: # Fallback to daily returns win rate
            positive_days = (returns > 0).sum()
            total_days = len(returns)
            if total_days > 0:
                metrics['win_rate'] = positive_days / total_days
            else:
                metrics['win_rate'] = 0.0


    # Sortino Ratio
    if metrics.get('sortino_ratio', 0) == 0:
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0:
            downside_std = neg_returns.std() * np.sqrt(252)
            if downside_std > 0:
                metrics['sortino_ratio'] = (returns.mean() * 252) / downside_std
            else:
                metrics['sortino_ratio'] = 0.0 # Avoid division by zero
        else:
            metrics['sortino_ratio'] = np.inf if returns.mean() > 0 else 0.0 # All positive returns

    # Calmar Ratio
    if metrics.get('calmar_ratio', 0) == 0:
        cagr = metrics.get('cagr', 0)
        max_dd = abs(metrics.get('max_drawdown', 0)) # MaxDD is usually negative, take absolute for Calmar
        
        if max_dd > 0:
            metrics['calmar_ratio'] = cagr / max_dd
        else:
            metrics['calmar_ratio'] = np.inf if cagr > 0 else 0.0 # Avoid division by zero
    
    # Profit Factor
    if metrics.get('profit_factor', 0) == 0:
        if not trades.empty and 'pnl' in trades.columns:
            wins_pnl = trades[trades['pnl'] > 0]['pnl'].sum()
            losses_pnl = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            if losses_pnl > 0:
                metrics['profit_factor'] = wins_pnl / losses_pnl
            else:
                metrics['profit_factor'] = np.inf if wins_pnl > 0 else 0.0 # Handle case with no losses
        else: # Fallback using returns
            pos_ret = returns[returns > 0].sum()
            neg_ret = abs(returns[returns < 0].sum())
            if neg_ret > 0:
                metrics['profit_factor'] = pos_ret / neg_ret
            else:
                metrics['profit_factor'] = np.inf if pos_ret > 0 else 0.0 # All positive returns

    # VaR & CVaR (if not already calculated)
    if metrics.get('var_95', None) is None:
        metrics['var_95'] = np.percentile(returns, 5) if len(returns) >= 2 else 0.0
    
    if metrics.get('cvar_95', None) is None:
        var_95_val = metrics.get('var_95', np.percentile(returns, 5) if len(returns) >= 2 else 0.0)
        filtered_returns = returns[returns <= var_95_val]
        metrics['cvar_95'] = filtered_returns.mean() if len(filtered_returns) > 0 else 0.0
    
    if metrics.get('var_99', None) is None: # Added for var_99
        metrics['var_99'] = np.percentile(returns, 1) if len(returns) >= 2 else 0.0

    if metrics.get('cvar_99', None) is None: # Added for cvar_99
        var_99_val = metrics.get('var_99', np.percentile(returns, 1) if len(returns) >= 2 else 0.0)
        filtered_returns = returns[returns <= var_99_val]
        metrics['cvar_99'] = filtered_returns.mean() if len(filtered_returns) > 0 else 0.0

    # Skewness & Kurtosis
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    metrics['best_day'] = returns.max()
    metrics['worst_day'] = returns.min()
    
    data['metrics'] = metrics
    return data

def load_results(results_dir: str = "results") -> Optional[Dict[str, Any]]:
    """Load backtest results from folder."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        st.error(f"Results directory '{results_dir}' not found.")
        return None
    
    data = {
        'metrics': {},
        'equity_curve': pd.Series(dtype=float),
        'returns': pd.Series(dtype=float),
        'trades': pd.DataFrame(),
        'feature_importance': pd.DataFrame(columns=['feature', 'importance']), # Pre-initialize columns
        'validation': pd.DataFrame(),
        'config': {}
    }
    
    # Load backtest metrics
    metrics_file = results_path / "backtest_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                data['metrics'] = json.load(f)
        except json.JSONDecodeError as e:
            st.error(f"Error loading backtest_metrics.json: {e}")
            return None
    
    # Load backtest results (equity curve, returns)
    backtest_file = results_path / "backtest_results.csv"
    if backtest_file.exists():
        try:
            df = pd.read_csv(backtest_file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Find equity curve column
            for col in ['portfolio_value', 'equity', 'value', 'cumulative_value']:
                if col in df.columns:
                    data['equity_curve'] = df[col]
                    break
            
            # Find returns column
            for col in ['returns', 'daily_return', 'return']:
                if col in df.columns:
                    data['returns'] = df[col]
                    break
            
            # Calculate returns if not found
            if data['returns'].empty and not data['equity_curve'].empty:
                data['returns'] = data['equity_curve'].pct_change().dropna()
            
            # Use backtest_results.csv as trades if it contains relevant columns
            if 'ticker' in df.columns or 'symbol' in df.columns or 'side' in df.columns:
                 data['trades'] = df.reset_index()
            # Else if there's a separate trades.csv
            else:
                trades_file_separate = results_path / "trades.csv"
                if trades_file_separate.exists():
                    trades_df_sep = pd.read_csv(trades_file_separate)
                    if 'date' in trades_df_sep.columns:
                        trades_df_sep['date'] = pd.to_datetime(trades_df_sep['date'])
                    data['trades'] = trades_df_sep.copy() # Use .copy() to avoid SettingWithCopyWarning later

        except Exception as e:
            st.error(f"Error loading backtest_results.csv: {e}")
            return None

    # Load feature importance
    feature_file = results_path / "feature_importance.csv"
    if feature_file.exists():
        try:
            fi_raw = pd.read_csv(feature_file)
            
            # Check for required columns and rename
            if 'feature' in fi_raw.columns and 'importance_mean' in fi_raw.columns:
                fi = fi_raw[['feature', 'importance_mean']].copy()
                fi = fi.rename(columns={'importance_mean': 'importance'})
                
                # Ensure 'importance' is numeric and handle NaNs
                fi['importance'] = pd.to_numeric(fi['importance'], errors='coerce')
                fi = fi.fillna(0) # Fill NaN importance with 0 instead of dropping rows
                
                # Sort for display
                fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
                data['feature_importance'] = fi
            else:
                st.warning(f"Feature importance CSV '{feature_file.name}' does not contain 'feature' and 'importance_mean' columns. Feature importance data will be empty.")
        except Exception as e:
            st.error(f"Error loading feature_importance.csv: {e}")
            
    # Load validation results
    validation_file = results_path / "validation_results.csv"
    if validation_file.exists():
        try:
            data['validation'] = pd.read_csv(validation_file)
        except Exception as e:
            st.error(f"Error loading validation_results.csv: {e}")
            data['validation'] = pd.DataFrame()
    
    # Create config based on loaded data
    data['config'] = {
        'start_date': str(data['equity_curve'].index.min())[:10] if not data['equity_curve'].empty else 'N/A',
        'end_date': str(data['equity_curve'].index.max())[:10] if not data['equity_curve'].empty else 'N/A',
        'universe': 'S&P 500',
        'initial_capital': data['metrics'].get('initial_capital', 1000000) # Use initial_capital from metrics if available
    }
    
    # Calculate missing metrics
    data = calculate_metrics(data)
    
    return data


# =============================================================================
# STREAMLIT APP RENDERING FUNCTIONS (ORDERED FOR PYTHON EXECUTION)
# =============================================================================

# This function defines general CSS styling for the Streamlit app
def apply_css():
    """Apply custom styling."""
    st.markdown(f"""
    <style>
    .main {{ background-color: {Config.COLORS['background']}; }}
    section[data-testid="stSidebar"] {{
        background-color: {Config.COLORS['card_bg']};
    }}
    h1, h2, h3 {{ color: {Config.COLORS['primary']} !important; }}
    div[data-testid="metric-container"] {{
        background-color: {Config.COLORS['card_bg']};
        border: 1px solid {Config.COLORS['border']};
        border-radius: 10px;
        padding: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)


# This function renders the content of the welcome page
def render_welcome():
    """Welcome page when no data loaded."""
    st.title("🚀 ML Alpha Dashboard")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Getting Started
        
        1. **Run Backtest First:**
        ```bash
        python scripts/run_research.py
        python scripts/run_backtest.py
        ```
        
        2. **Load Results:** Click "🔄 Load Results" in sidebar
        
        ---
        
        ### Dashboard Pages
        
        | Page | Description |
        |------|-------------|
        | 📈 **Overview** | Key metrics, equity curve |
        | 📊 **Performance** | Rolling metrics, distributions |
        | 🔬 **Factors** | Feature importance |
        | 💹 **Trades** | Trade analysis |
        | ⚠️ **Risk** | VaR, volatility |
        | 📋 **Data** | Raw data, exports |
        """)
    
    with col2:
        st.markdown("### Status")
        
        results_path = Path("results")
        if results_path.exists():
            files = list(results_path.glob("*"))
            st.success(f"📂 Results: {len(files)} files")
            
            # Check specific files
            for f in ['backtest_metrics.json', 'backtest_results.csv', 'feature_importance.csv']:
                if (results_path / f).exists():
                    st.info(f"✅ {f}")
                else:
                    st.warning(f"❌ {f} - Missing. Backtest did not generate this file or path is incorrect.")
        else:
            st.error("📂 Results folder 'results/' not found. Please create it or ensure scripts output here.")


# This function renders the content of the overview page
def render_overview():
    """Overview page."""
    st.title("📈 Performance Overview")
    st.divider()
    
    # Use app_data from session_state
    data = st.session_state.app_data 
    metrics = data.get('metrics', {})
    equity = data.get('equity_curve', pd.Series())
    returns = data.get('returns', pd.Series())
    config = data.get('config', {})
    
    # Config bar
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"📅 {config.get('start_date', 'N/A')} → {config.get('end_date', 'N/A')}")
    col2.info(f"🏛️ {config.get('universe', 'N/A')}")
    col3.info(f"💰 ${config.get('initial_capital', 0):,.0f}")
    
    # Display trading days or number of trades
    if not equity.empty:
        col4.info(f"📊 {len(equity)} trading days") 
    elif not data.get('trades', pd.DataFrame()).empty:
         col4.info(f"📊 {len(data['trades'])} trades")
    else:
        col4.info(f"📊 0 periods")
    
    st.divider()
    
    # Key Metrics Row 1
    st.subheader("🎯 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cagr = metrics.get('cagr', 0)
        cagr_pct = cagr * 100 
        st.metric("CAGR", f"{cagr_pct:.2f}%")
    
    with col2:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    with col3:
        max_dd = metrics.get('max_drawdown', 0)
        dd_pct = max_dd * 100 
        st.metric("Max Drawdown", f"{dd_pct:.2f}%")
    
    with col4:
        win_rate = metrics.get('win_rate', 0)
        wr_pct = win_rate * 100 
        st.metric("Win Rate", f"{wr_pct:.1f}%")
    
    st.divider() # Separator for better layout
    
    # Key Metrics Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sortino = metrics.get('sortino_ratio', 0)
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    
    with col2:
        calmar = metrics.get('calmar_ratio', 0)
        st.metric("Calmar Ratio", f"{calmar:.2f}")
    
    with col3:
        pf = metrics.get('profit_factor', 0)
        # Handle inf for profit factor if no losing trades
        if pf == np.inf:
            st.metric("Profit Factor", "∞")
        else:
            st.metric("Profit Factor", f"{pf:.2f}")
    
    with col4:
        total_ret = metrics.get('total_return', 0)
        tr_pct = total_ret * 100 
        st.metric("Total Return", f"{tr_pct:.2f}%")
    
    st.divider()
    
    # Equity Curve
    st.subheader("📊 Equity Curve")
    if not equity.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode='lines', name='Portfolio',
            line=dict(color=Config.COLORS['primary'], width=2),
            fill='tozeroy', fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        fig.update_layout(
            template=Config.CHART_TEMPLATE,
            height=Config.CHART_HEIGHT,
            xaxis_title="Date", yaxis_title="Value ($)",
            yaxis=dict(tickformat=',.0f')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No equity curve data available.")
    
    # Drawdown & Monthly
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Drawdown")
        if not equity.empty:
            peak = equity.expanding().max()
            dd = (equity - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                mode='lines', fill='tozeroy',
                line=dict(color=Config.COLORS['negative'], width=1),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            fig.update_layout(
                template=Config.CHART_TEMPLATE,
                height=Config.CHART_HEIGHT_SMALL,
                xaxis_title="Date", yaxis_title="Drawdown (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No equity curve data for drawdown analysis.")
    
    with col2:
        st.subheader("📅 Monthly Returns")
        if not returns.empty:
            monthly = returns.resample('M').apply(lambda x: (1+x).prod() - 1)
            df = pd.DataFrame({
                'Year': monthly.index.year,
                'Month': monthly.index.month,
                'Return': monthly.values * 100
            })
            pivot = df.pivot(index='Year', columns='Month', values='Return')
            
            fig = px.imshow(
                pivot, color_continuous_scale='RdYlGn',
                labels=dict(x="Month", y="Year", color="Return (%)"),
                text_auto='.1f', aspect='auto'
            )
            fig.update_layout(
                template=Config.CHART_TEMPLATE,
                height=Config.CHART_HEIGHT_SMALL
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No returns data for monthly heatmap.")

# This function renders the content of the performance page
def render_performance():
    """Performance analysis page."""
    st.title("📊 Performance Analysis")
    st.divider()
    
    returns = st.session_state.app_data.get('returns', pd.Series()) 
    
    if returns.empty:
        st.warning("No returns data available.")
        return
    
    # Controls
    window = st.slider("Rolling Window (days)", 21, 252, 63)
    
    # Rolling Metrics
    st.subheader(f"📈 Rolling {window}-Day Metrics")
    
    rolling_ret = returns.rolling(window).mean() * 252 * 100
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_sharpe = rolling_ret / rolling_vol
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=['Return (%)', 'Volatility (%)', 'Sharpe'],
                        vertical_spacing=0.08)
    
    fig.add_trace(go.Scatter(x=rolling_ret.index, y=rolling_ret.values,
                             line=dict(color=Config.COLORS['primary'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                             line=dict(color=Config.COLORS['warning'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                             line=dict(color=Config.COLORS['info'])), row=3, col=1)
    
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color='green', row=3, col=1)
    
    fig.update_layout(template=Config.CHART_TEMPLATE, height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution
    st.subheader("📊 Returns Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns*100, nbinsx=50,
                                   marker_color=Config.COLORS['primary'], opacity=0.7))
        fig.add_vline(x=0, line_dash='dash', line_color='white')
        fig.update_layout(template=Config.CHART_TEMPLATE, height=350,
                          xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        df = pd.DataFrame({'Return': returns*100, 'Month': returns.index.month_name().str[:3]})
        fig = px.box(df, x='Month', y='Return', color_discrete_sequence=[Config.COLORS['primary']])
        fig.update_layout(template=Config.CHART_TEMPLATE, height=350)
        st.plotly_chart(fig, use_container_width=True)


# This function renders the content of the factors page
def render_factors():
    """Factor analysis page."""
    st.title("🔬 Factor Analysis")
    st.divider()
    
    fi = st.session_state.app_data.get('feature_importance', pd.DataFrame()) 
    
    if fi.empty:
        st.warning("No feature importance data available. Ensure 'feature_importance.csv' is generated correctly with 'feature' and 'importance_mean' columns.")
        return
    
    # Validate columns after loading and renaming
    if 'importance' not in fi.columns or 'feature' not in fi.columns:
        st.error(f"Feature importance data format incorrect. Expected 'feature' and 'importance' columns. Found: {fi.columns.tolist()}")
        st.dataframe(fi.head())
        return
    
    # Ensure 'importance' is numeric and drop NaNs (already handled in load_results now)
    # The fi coming here should already be clean.
    
    # Slider min value to 1, max to length of features, default 15
    top_n = st.slider("Top N Features", 1, min(30, len(fi)), 15) 
    
    top = fi.nlargest(top_n, 'importance')
    
    # Bar chart
    st.subheader(f"📊 Top {top_n} Features")
    fig = px.bar(top, x='importance', y='feature', orientation='h',
                 color='importance', color_continuous_scale='viridis')
    fig.update_layout(template=Config.CHART_TEMPLATE, height=max(400, top_n*25),
                      yaxis=dict(categoryorder='total ascending'), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cumulative Importance")
        sorted_fi = fi.sort_values('importance', ascending=False)
        cumulative = sorted_fi['importance'].cumsum() / sorted_fi['importance'].sum() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(cumulative)+1)), y=cumulative.values,
                                 fill='tozeroy', line=dict(color=Config.COLORS['primary'])))
        fig.add_hline(y=80, line_dash='dash', line_color=Config.COLORS['warning'])
        fig.add_hline(y=95, line_dash='dash', line_color=Config.COLORS['negative'])
        fig.update_layout(template=Config.CHART_TEMPLATE, height=350,
                          xaxis_title="Number of Features", yaxis_title="Cumulative Importance (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Feature Table")
        display = top.copy()
        display['rank'] = range(1, len(display)+1)
        display['importance'] = display['importance'].apply(lambda x: f"{x:.6f}")
        st.dataframe(display[['rank', 'feature', 'importance']], use_container_width=True, hide_index=True)


# This function renders the content of the trades page
def render_trades():
    """Trade analysis page."""
    st.title("💹 Trade Analysis")
    st.divider()
    
    trades = st.session_state.app_data.get('trades', pd.DataFrame()) 
    returns = st.session_state.app_data.get('returns', pd.Series()) 
    
    if trades.empty and returns.empty:
        st.warning("No trade data available. Ensure backtest generates trades or returns.")
        return
    
    # Calculate P&L from returns if trades don't have it (or if trades are empty)
    if trades.empty or 'pnl' not in trades.columns:
        st.info("No individual trade data found ('trades.csv' not generated or 'pnl' column missing). Showing returns-based analysis.")
        
        if returns.empty:
            st.warning("No returns data to generate trade-like analysis.")
            return

        # Monthly P&L
        monthly = returns.resample('M').apply(lambda x: (1+x).prod() - 1) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Monthly Returns Distribution")
            fig = go.Figure()
            colors = [Config.COLORS['primary'] if r > 0 else Config.COLORS['negative'] for r in monthly.values]
            fig.add_trace(go.Bar(x=monthly.index.strftime('%Y-%m'), y=monthly.values, marker_color=colors))
            fig.add_hline(y=0, line_dash='dash', line_color='white')
            fig.update_layout(template=Config.CHART_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Cumulative Returns")
            cum_ret = (1 + returns).cumprod() - 1
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret.values*100,
                                     fill='tozeroy', line=dict(color=Config.COLORS['primary'])))
            fig.update_layout(template=Config.CHART_TEMPLATE, height=400, yaxis_title="Return (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Trade-based analysis (if pnl is available in trades)
    if 'pnl' in trades.columns:
        total_pnl = trades['pnl'].sum()
        wins = (trades['pnl'] > 0).sum()
        losses = (trades['pnl'] <= 0).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total P&L", f"${total_pnl:,.0f}")
        col2.metric("Avg P&L", f"${trades['pnl'].mean():,.0f}")
        col3.metric("Wins", wins)
        col4.metric("Losses", losses)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 P&L Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=trades['pnl']/1000, nbinsx=50,
                                       marker_color=Config.COLORS['primary']))
            fig.add_vline(x=0, line_dash='dash', line_color='white')
            fig.update_layout(template=Config.CHART_TEMPLATE, height=350, xaxis_title="P&L ($K)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Cumulative P&L")
            cum_pnl = trades['pnl'].cumsum() / 1000
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(cum_pnl))), y=cum_pnl.values,
                                     fill='tozeroy', line=dict(color=Config.COLORS['primary'])))
            fig.update_layout(template=Config.CHART_TEMPLATE, height=350, yaxis_title="Cumulative P&L ($K)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Trade log
    st.subheader("📋 Data Log")
    st.dataframe(trades.head(50), use_container_width=True)


# This function renders the content of the risk page
def render_risk():
    """Risk analysis page."""
    st.title("⚠️ Risk Analysis")
    st.divider()
    
    returns = st.session_state.app_data.get('returns', pd.Series()) 
    metrics = st.session_state.app_data.get('metrics', {}) # Get metrics from state
    
    if returns.empty:
        st.warning("No returns data available.")
        return
    
    # Risk Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Use metrics from the calculated dict
    var_95_val = metrics.get('var_95', 0) * 100
    var_99_val = metrics.get('var_99', 0) * 100
    cvar_95_val = metrics.get('cvar_95', 0) * 100
    cvar_99_val = metrics.get('cvar_99', 0) * 100 # Ensure this is also from metrics
    volatility = returns.std() * np.sqrt(252) * 100 # Re-calculate here for chart for simplicity
    
    col1.metric("VaR (95%)", f"{var_95_val:.2f}%")
    col2.metric("VaR (99%)", f"{var_99_val:.2f}%")
    col3.metric("CVaR (95%)", f"{cvar_95_val:.2f}%")
    col4.metric("Ann. Volatility", f"{volatility:.2f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 VaR Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns*100, nbinsx=50,
                                   marker_color=Config.COLORS['primary'], opacity=0.7))
        fig.add_vline(x=var_95_val, line_dash='dash', line_color=Config.COLORS['warning'],
                      annotation_text=f'VaR 95%: {var_95_val:.2f}%')
        fig.add_vline(x=var_99_val, line_dash='dash', line_color=Config.COLORS['negative'],
                      annotation_text=f'VaR 99%: {var_99_val:.2f}%')
        fig.update_layout(template=Config.CHART_TEMPLATE, height=400, xaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Rolling Volatility")
        rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                                 line=dict(color=Config.COLORS['warning'])))
        fig.add_hline(y=rolling_vol.mean(), line_dash='dash', line_color=Config.COLORS['primary'])
        fig.update_layout(template=Config.CHART_TEMPLATE, height=400, yaxis_title="Volatility (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Table
    st.subheader("📋 Risk Metrics Table")
    risk_data = {
        'Metric': ['Volatility (Ann.)', 'VaR (95%)', 'VaR (99%)', 'CVaR (95%)', 'CVaR (99%)',
                   'Skewness', 'Kurtosis', 'Best Day', 'Worst Day', 'Positive Days %'],
        'Value': [
            f"{volatility:.2f}%", f"{var_95_val:.2f}%", f"{var_99_val:.2f}%", f"{cvar_95_val:.2f}%", f"{cvar_99_val:.2f}%",
            f"{metrics.get('skewness', 0):.4f}", f"{metrics.get('kurtosis', 0):.4f}",
            f"{metrics.get('best_day', 0)*100:.2f}%", f"{metrics.get('worst_day', 0)*100:.2f}%",
            f"{(returns > 0).mean()*100:.1f}%"
        ]
    }
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)


# This function renders the content of the data explorer page
def render_data():
    """Data explorer page."""
    st.title("📋 Data Explorer")
    st.divider()
    
    data = st.session_state.app_data
    
    # Data options
    options = []
    if not data.get('equity_curve', pd.Series()).empty:
        options.append("Equity Curve")
    if not data.get('returns', pd.Series()).empty:
        options.append("Returns")
    if not data.get('trades', pd.DataFrame()).empty:
        options.append("Trades")
    if not data.get('feature_importance', pd.DataFrame()).empty:
        options.append("Features")
    if data.get('metrics'):
        options.append("Metrics")
    if not data.get('validation', pd.DataFrame()).empty:
        options.append("Validation")
    
    if not options:
        st.warning("No data to display. Load backtest results first.")
        return

    selected = st.selectbox("Select Data", options)
    st.divider()
    
    if selected == "Equity Curve":
        df = data['equity_curve'].to_frame('value')
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(), "equity_curve.csv", "text/csv")
    
    elif selected == "Returns":
        df = data['returns'].to_frame('return')
        df['cumulative'] = (1 + data['returns']).cumprod() - 1
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(), "returns.csv", "text/csv")
    
    elif selected == "Trades":
        df = data['trades']
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "trades.csv", "text/csv")
    
    elif selected == "Features":
        df = data['feature_importance']
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "features.csv", "text/csv")
    
    elif selected == "Metrics":
        df = pd.DataFrame([{'Metric': k, 'Value': v} for k, v in data['metrics'].items()])
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "metrics.csv", "text/csv")
    
    elif selected == "Validation":
        df = data['validation']
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "validation.csv", "text/csv")


# This function renders the sidebar content and handles data loading/navigation
def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>📊 ML Alpha</h2>", unsafe_allow_html=True)
        st.divider()
        
        # Data Loading Section
        st.subheader("📁 Load Results")
        
        # Text input for results folder path
        results_dir = st.text_input("Results folder:", value="results", key="results_folder_input")
        
        # Display files found in the results directory (for user info)
        results_path = Path(results_dir)
        if results_path.exists():
            files = [f.name for f in results_path.glob("*") if f.is_file()]
            if files:
                with st.expander(f"📂 {len(files)} files found"):
                    for f in files[:10]: # Show up to 10 files
                        st.caption(f"• {f}")
        else:
             st.warning(f"Folder '{results_dir}' not found.")

        # Button to trigger data loading
        if st.button("🔄 Load Results", use_container_width=True, type="primary", key="load_results_button"):
            with st.spinner("Loading..."):
                data = load_results(results_dir)
            
            # Check if data was loaded successfully (has metrics and equity curve)
            if data and not data.get('equity_curve', pd.Series()).empty:
                st.session_state.app_data = data  # Assign data to unique session state key
                st.success("✅ Loaded!")
                st.rerun() # Rerun to update the dashboard with new data
            else:
                st.error("❌ No valid results found. Make sure 'backtest_metrics.json' and 'backtest_results.csv' exist and contain data.")
        
        # Auto-load on the very first run of the app
        # This prevents the welcome page from showing if results are already there
        if st.session_state.app_data is None:
            auto_data = load_results(results_dir)
            if auto_data and not auto_data.get('equity_curve', pd.Series()).empty:
                st.session_state.app_data = auto_data
                st.rerun() # Rerun to display loaded data

        st.divider()
        
        # Navigation section
        if st.session_state.app_data: # Only show navigation if data is loaded
            st.subheader("🧭 Navigation")
            
            for icon, name, page_id in Config.PAGES:
                is_active = st.session_state.page == page_id
                btn_type = "primary" if is_active else "secondary"
                
                # IMPORTANT: Use a unique key for each button to avoid StreamlitValueAssignmentNotAllowedError
                if st.button(f"{icon} {name}", key=f"nav_{page_id}", use_container_width=True, type=btn_type):
                    st.session_state.page = page_id # Update current page in session state
                    st.rerun() # Rerun to display the selected page
            
            st.divider()
            
            # Quick Stats section (if data is loaded)
            st.subheader("📊 Quick Stats")
            metrics = st.session_state.app_data.get('metrics', {}) # Access data from unique session state key
            
            col1, col2 = st.columns(2)
            with col1:
                cagr = metrics.get('cagr', 0)
                cagr_pct = cagr * 100
                st.metric("CAGR", f"{cagr_pct:.1f}%")
                
                sharpe = metrics.get('sharpe_ratio', 0)
                st.metric("Sharpe", f"{sharpe:.2f}")
            with col2:
                max_dd = metrics.get('max_drawdown', 0)
                dd_pct = max_dd * 100
                st.metric("Max DD", f"{dd_pct:.1f}%")
                
                win_rate = metrics.get('win_rate', 0)
                wr_pct = win_rate * 100
                st.metric("Win Rate", f"{wr_pct:.0f}%")
        
        st.divider()
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# This function routes to the appropriate rendering function based on the current page
def render_page():
    """Render current page based on session state."""
    page = st.session_state.page
    
    if page == 'overview':
        render_overview()
    elif page == 'performance':
        render_performance()
    elif page == 'factors':
        render_factors()
    elif page == 'trades':
        render_trades()
    elif page == 'risk':
        render_risk()
    elif page == 'data':
        render_data()


# =============================================================================
# MAIN STREAMLIT APP ENTRY POINT
# =============================================================================

def main():
    """Main dashboard function to setup the Streamlit app."""
    if not STREAMLIT_AVAILABLE or not PLOTLY_AVAILABLE:
        print("❌ Required packages not installed! Please install streamlit and plotly.")
        sys.exit(1) # Exit if essential libraries are missing
    
    st.set_page_config(**Config.PAGE_CONFIG)
    apply_css()
    
    # Initialize session state variables with unique keys
    if 'page' not in st.session_state:
        st.session_state.page = 'overview'
    if 'app_data' not in st.session_state: # Renamed from 'data' to 'app_data' to avoid conflicts
        st.session_state.app_data = None
    
    render_sidebar() # Render the sidebar first
    
    # Render main content area
    if st.session_state.app_data is None: # If no data is loaded yet
        render_welcome()
    else: # If data is loaded, render the selected page
        render_page()


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()