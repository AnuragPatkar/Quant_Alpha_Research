"""
Interactive Dashboard - ML Alpha Model
=======================================
Streamlit-based dashboard for backtest analysis.

Usage:
    streamlit run quant_alpha/visualization/dashboards.py

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import json
import sys
import logging

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

STREAMLIT_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    logger.warning("Streamlit not installed. Install with: pip install streamlit")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not installed. Install with: pip install plotly")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration with sensible defaults."""
    
    # Color scheme
    colors: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#00D4AA',
        'secondary': '#4ECDC4',
        'positive': '#00D4AA',
        'negative': '#FF6B6B',
        'warning': '#FFD93D',
        'info': '#4ECDC4',
        'neutral': '#888888',
        'background': '#0E1117',
        'card_bg': '#1E2130',
        'border': '#2D3748',
        'text': '#FFFFFF',
        'text_secondary': '#A0AEC0',
        'grid': '#2D3748',
    })
    
    # Page configuration
    page_title: str = '📊 ML Alpha Dashboard'
    page_icon: str = '📊'
    layout: str = 'wide'
    initial_sidebar_state: str = 'expanded'
    
    # Chart settings
    chart_height: int = 450
    chart_height_small: int = 350
    chart_height_large: int = 600
    chart_template: str = 'plotly_dark'
    
    # Default paths
    default_results_dir: str = 'results'
    
    @property
    def page_config(self) -> Dict:
        """Get Streamlit page config dict."""
        return {
            'page_title': self.page_title,
            'page_icon': self.page_icon,
            'layout': self.layout,
            'initial_sidebar_state': self.initial_sidebar_state
        }
    
    @property
    def pages(self) -> List[tuple]:
        """Get page definitions."""
        return [
            ("📈", "Overview", "overview"),
            ("📊", "Performance", "performance"),
            ("🔬", "Factors", "factors"),
            ("💹", "Trades", "trades"),
            ("⚠️", "Risk", "risk"),
            ("📋", "Data", "data")
        ]


# Global config instance
Config = DashboardConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division handling zero and NaN."""
    if b == 0 or np.isnan(b) or np.isnan(a):
        return default
    return a / b


def format_pct(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with handling for special values."""
    if np.isnan(value):
        return "N/A"
    if np.isinf(value):
        return "∞" if value > 0 else "-∞"
    return f"{value:.{decimals}f}"


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as currency."""
    if np.isnan(value) or np.isinf(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


# =============================================================================
# DATA LOADING & METRIC CALCULATION
# =============================================================================

def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all missing metrics from available data.
    
    Args:
        data: Dictionary with 'metrics', 'returns', 'equity_curve', 'trades'
        
    Returns:
        Updated data dictionary with complete metrics
    """
    metrics = data.get('metrics', {})
    returns = data.get('returns', pd.Series(dtype=float))
    equity = data.get('equity_curve', pd.Series(dtype=float))
    trades = data.get('trades', pd.DataFrame())
    
    # Handle empty returns
    if returns.empty or len(returns) < 2:
        default_metrics = {
            'total_return': 0.0,
            'cagr': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'volatility': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'best_day': 0.0,
            'worst_day': 0.0,
        }
        for key, value in default_metrics.items():
            metrics.setdefault(key, value)
        data['metrics'] = metrics
        return data
    
    # Clean returns
    returns = returns.dropna()
    
    # ===== RETURN METRICS =====
    
    # Total Return
    if 'total_return' not in metrics or metrics['total_return'] == 0:
        if not equity.empty and len(equity) > 1:
            metrics['total_return'] = equity.iloc[-1] / equity.iloc[0] - 1
        else:
            metrics['total_return'] = (1 + returns).prod() - 1
    
    # CAGR
    if 'cagr' not in metrics or metrics['cagr'] == 0:
        if not equity.empty and len(equity) > 1:
            n_years = (equity.index.max() - equity.index.min()).days / 365.25
            if n_years > 0:
                total_ret = metrics.get('total_return', 0)
                metrics['cagr'] = (1 + total_ret) ** (1 / n_years) - 1
            else:
                metrics['cagr'] = 0.0
        else:
            # Estimate from returns
            n_years = len(returns) / 252
            if n_years > 0:
                metrics['cagr'] = (1 + metrics.get('total_return', 0)) ** (1 / n_years) - 1
            else:
                metrics['cagr'] = 0.0
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    metrics['volatility'] = volatility
    
    # ===== RISK-ADJUSTED METRICS =====
    
    # Sharpe Ratio
    if 'sharpe_ratio' not in metrics or metrics['sharpe_ratio'] == 0:
        annual_return = returns.mean() * 252
        metrics['sharpe_ratio'] = safe_divide(annual_return, volatility, 0.0)
    
    # Sortino Ratio
    if 'sortino_ratio' not in metrics or metrics['sortino_ratio'] == 0:
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0:
            downside_std = neg_returns.std() * np.sqrt(252)
            annual_return = returns.mean() * 252
            metrics['sortino_ratio'] = safe_divide(annual_return, downside_std, 0.0)
        else:
            metrics['sortino_ratio'] = np.inf if returns.mean() > 0 else 0.0
    
    # Max Drawdown
    if 'max_drawdown' not in metrics or metrics['max_drawdown'] == 0:
        if not equity.empty and len(equity) > 1:
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            metrics['max_drawdown'] = drawdown.min()  # Negative value
        else:
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            metrics['max_drawdown'] = drawdown.min()
    
    # Calmar Ratio
    if 'calmar_ratio' not in metrics or metrics['calmar_ratio'] == 0:
        cagr = metrics.get('cagr', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))
        metrics['calmar_ratio'] = safe_divide(cagr, max_dd, 0.0)
    
    # ===== TRADE METRICS =====
    
    # Win Rate
    if 'win_rate' not in metrics or metrics['win_rate'] == 0:
        if not trades.empty and 'pnl' in trades.columns:
            total_trades = len(trades)
            if total_trades > 0:
                wins = (trades['pnl'] > 0).sum()
                metrics['win_rate'] = wins / total_trades
            else:
                metrics['win_rate'] = 0.0
        else:
            # Use daily returns
            metrics['win_rate'] = (returns > 0).mean()
    
    # Profit Factor
    if 'profit_factor' not in metrics or metrics['profit_factor'] == 0:
        if not trades.empty and 'pnl' in trades.columns:
            gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
            metrics['profit_factor'] = safe_divide(gross_profit, gross_loss, np.inf if gross_profit > 0 else 0.0)
        else:
            pos_ret = returns[returns > 0].sum()
            neg_ret = abs(returns[returns < 0].sum())
            metrics['profit_factor'] = safe_divide(pos_ret, neg_ret, np.inf if pos_ret > 0 else 0.0)
    
    # ===== RISK METRICS =====
    
    # VaR
    if 'var_95' not in metrics:
        metrics['var_95'] = np.percentile(returns, 5)
    if 'var_99' not in metrics:
        metrics['var_99'] = np.percentile(returns, 1)
    
    # CVaR (Expected Shortfall)
    if 'cvar_95' not in metrics:
        var_95 = metrics['var_95']
        tail_returns = returns[returns <= var_95]
        metrics['cvar_95'] = tail_returns.mean() if len(tail_returns) > 0 else var_95
    
    if 'cvar_99' not in metrics:
        var_99 = metrics['var_99']
        tail_returns = returns[returns <= var_99]
        metrics['cvar_99'] = tail_returns.mean() if len(tail_returns) > 0 else var_99
    
    # ===== DISTRIBUTION METRICS =====
    
    metrics['skewness'] = returns.skew() if len(returns) > 2 else 0.0
    metrics['kurtosis'] = returns.kurtosis() if len(returns) > 3 else 0.0
    metrics['best_day'] = returns.max()
    metrics['worst_day'] = returns.min()
    
    data['metrics'] = metrics
    return data


def load_results(results_dir: str = "results") -> Optional[Dict[str, Any]]:
    """
    Load backtest results from folder.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with loaded data, or None if loading fails
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory '{results_dir}' not found.")
        return None
    
    data = {
        'metrics': {},
        'equity_curve': pd.Series(dtype=float),
        'returns': pd.Series(dtype=float),
        'trades': pd.DataFrame(),
        'feature_importance': pd.DataFrame(columns=['feature', 'importance']),
        'validation': pd.DataFrame(),
        'config': {}
    }
    
    # Load backtest metrics
    metrics_file = results_path / "backtest_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                data['metrics'] = json.load(f)
            logger.info("Loaded backtest_metrics.json")
        except json.JSONDecodeError as e:
            logger.error(f"Error loading backtest_metrics.json: {e}")
    
    # Load backtest results (equity curve, returns)
    backtest_file = results_path / "backtest_results.csv"
    if backtest_file.exists():
        try:
            df = pd.read_csv(backtest_file)
            
            # Parse dates
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Find equity curve column
            equity_cols = ['portfolio_value', 'equity', 'value', 'cumulative_value', 'nav']
            for col in equity_cols:
                if col in df.columns:
                    data['equity_curve'] = df[col].dropna()
                    break
            
            # Find returns column
            return_cols = ['returns', 'daily_return', 'return', 'pct_change']
            for col in return_cols:
                if col in df.columns:
                    data['returns'] = df[col].dropna()
                    break
            
            # Calculate returns if not found
            if data['returns'].empty and not data['equity_curve'].empty:
                data['returns'] = data['equity_curve'].pct_change().dropna()
            
            logger.info(f"Loaded backtest_results.csv: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error loading backtest_results.csv: {e}")
    
    # Load trades
    trades_file = results_path / "trades.csv"
    if trades_file.exists():
        try:
            trades_df = pd.read_csv(trades_file)
            if 'date' in trades_df.columns:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
            data['trades'] = trades_df
            logger.info(f"Loaded trades.csv: {len(trades_df)} trades")
        except Exception as e:
            logger.error(f"Error loading trades.csv: {e}")
    
    # Load feature importance
    feature_file = results_path / "feature_importance.csv"
    if feature_file.exists():
        try:
            fi = pd.read_csv(feature_file)
            
            # Standardize column names
            if 'importance_mean' in fi.columns:
                fi = fi.rename(columns={'importance_mean': 'importance'})
            
            if 'feature' in fi.columns and 'importance' in fi.columns:
                fi['importance'] = pd.to_numeric(fi['importance'], errors='coerce').fillna(0)
                fi = fi.sort_values('importance', ascending=False).reset_index(drop=True)
                data['feature_importance'] = fi[['feature', 'importance']]
                logger.info(f"Loaded feature_importance.csv: {len(fi)} features")
            else:
                logger.warning("Feature importance file missing required columns")
                
        except Exception as e:
            logger.error(f"Error loading feature_importance.csv: {e}")
    
    # Load validation results
    validation_file = results_path / "validation_results.csv"
    if validation_file.exists():
        try:
            data['validation'] = pd.read_csv(validation_file)
            logger.info(f"Loaded validation_results.csv")
        except Exception as e:
            logger.error(f"Error loading validation_results.csv: {e}")
    
    # Build config from loaded data
    equity = data['equity_curve']
    data['config'] = {
        'start_date': str(equity.index.min().date()) if not equity.empty else 'N/A',
        'end_date': str(equity.index.max().date()) if not equity.empty else 'N/A',
        'trading_days': len(equity) if not equity.empty else 0,
        'universe': 'S&P 500',
        'initial_capital': data['metrics'].get('initial_capital', 1_000_000)
    }
    
    # Calculate missing metrics
    data = calculate_metrics(data)
    
    return data


# =============================================================================
# CSS STYLING
# =============================================================================

def apply_css() -> None:
    """Apply custom CSS styling to the dashboard."""
    css = f"""
    <style>
        /* Main background */
        .main {{
            background-color: {Config.colors['background']};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {Config.colors['card_bg']};
        }}
        
        /* Headers */
        h1, h2, h3 {{
            color: {Config.colors['primary']} !important;
        }}
        
        /* Metric containers */
        div[data-testid="metric-container"] {{
            background-color: {Config.colors['card_bg']};
            border: 1px solid {Config.colors['border']};
            border-radius: 10px;
            padding: 15px;
        }}
        
        /* Dataframes */
        .stDataFrame {{
            border: 1px solid {Config.colors['border']};
            border-radius: 5px;
        }}
        
        /* Buttons */
        .stButton > button {{
            border-radius: 5px;
        }}
        
        /* Dividers */
        hr {{
            border-color: {Config.colors['border']};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# PAGE RENDERERS
# =============================================================================

def render_welcome() -> None:
    """Render welcome page when no data is loaded."""
    st.title("🚀 ML Alpha Dashboard")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 👋 Welcome!
        
        This dashboard visualizes backtest results from the ML Alpha model.
        
        #### Getting Started
        
        1. **Run the backtest first:**
        ```bash
        python scripts/run_research.py
        python scripts/run_backtest.py
        ```
        
        2. **Load results:** Click "🔄 Load Results" in the sidebar
        
        ---
        
        #### Dashboard Pages
        
        | Page | Description |
        |------|-------------|
        | 📈 **Overview** | Key metrics and equity curve |
        | 📊 **Performance** | Rolling metrics and distributions |
        | 🔬 **Factors** | Feature importance analysis |
        | 💹 **Trades** | Trade-level analysis |
        | ⚠️ **Risk** | VaR, volatility, risk metrics |
        | 📋 **Data** | Raw data explorer and exports |
        """)
    
    with col2:
        st.markdown("### 📁 Status")
        
        results_path = Path(Config.default_results_dir)
        
        if results_path.exists():
            files = list(results_path.glob("*"))
            st.success(f"📂 Found {len(files)} files")
            
            expected_files = [
                'backtest_metrics.json',
                'backtest_results.csv',
                'feature_importance.csv',
                'trades.csv',
                'validation_results.csv'
            ]
            
            for f in expected_files:
                if (results_path / f).exists():
                    st.info(f"✅ {f}")
                else:
                    st.warning(f"⚠️ {f}")
        else:
            st.error(f"📂 Results folder not found")
            st.caption(f"Expected: {results_path.absolute()}")


def render_overview() -> None:
    """Render overview page."""
    st.title("📈 Performance Overview")
    st.divider()
    
    data = st.session_state.app_data
    metrics = data.get('metrics', {})
    equity = data.get('equity_curve', pd.Series())
    returns = data.get('returns', pd.Series())
    config = data.get('config', {})
    
    # Config summary
    col1, col2, col3, col4 = st.columns(4)
    col1.info(f"📅 {config.get('start_date', 'N/A')} → {config.get('end_date', 'N/A')}")
    col2.info(f"🏛️ {config.get('universe', 'N/A')}")
    col3.info(f"💰 {format_currency(config.get('initial_capital', 0))}")
    col4.info(f"📊 {config.get('trading_days', 0)} days")
    
    st.divider()
    
    # Key Metrics - Row 1
    st.subheader("🎯 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("CAGR", format_pct(metrics.get('cagr', 0)))
    col2.metric("Sharpe Ratio", format_number(metrics.get('sharpe_ratio', 0)))
    col3.metric("Max Drawdown", format_pct(metrics.get('max_drawdown', 0)))
    col4.metric("Win Rate", format_pct(metrics.get('win_rate', 0)))
    
    # Key Metrics - Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Sortino Ratio", format_number(metrics.get('sortino_ratio', 0)))
    col2.metric("Calmar Ratio", format_number(metrics.get('calmar_ratio', 0)))
    col3.metric("Profit Factor", format_number(metrics.get('profit_factor', 0)))
    col4.metric("Total Return", format_pct(metrics.get('total_return', 0)))
    
    st.divider()
    
    # Equity Curve
    st.subheader("📊 Equity Curve")
    
    if not equity.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Portfolio',
            line=dict(color=Config.colors['primary'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(0, 212, 170, 0.1)"
        ))
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height,
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            yaxis=dict(tickformat=',.0f'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No equity curve data available")
    
    # Drawdown and Monthly Returns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Drawdown")
        
        if not equity.empty:
            peak = equity.expanding().max()
            dd = (equity - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                mode='lines',
                fill='tozeroy',
                line=dict(color=Config.colors['negative'], width=1),
                fillcolor=f"rgba(255, 107, 107, 0.3)"
            ))
            fig.update_layout(
                template=Config.chart_template,
                height=Config.chart_height_small,
                xaxis_title="Date",
                yaxis_title="Drawdown (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data for drawdown")
    
    with col2:
        st.subheader("📅 Monthly Returns")
        
        if not returns.empty:
            try:
                monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                df_monthly = pd.DataFrame({
                    'Year': monthly.index.year,
                    'Month': monthly.index.month,
                    'Return': monthly.values * 100
                })
                
                pivot = df_monthly.pivot(index='Year', columns='Month', values='Return')
                
                fig = px.imshow(
                    pivot,
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="Month", y="Year", color="Return (%)"),
                    text_auto='.1f',
                    aspect='auto'
                )
                fig.update_layout(
                    template=Config.chart_template,
                    height=Config.chart_height_small
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate monthly heatmap: {e}")
        else:
            st.warning("No returns data for heatmap")


def render_performance() -> None:
    """Render performance analysis page."""
    st.title("📊 Performance Analysis")
    st.divider()
    
    returns = st.session_state.app_data.get('returns', pd.Series())
    
    if returns.empty:
        st.warning("No returns data available")
        return
    
    # Controls
    window = st.slider("Rolling Window (days)", min_value=21, max_value=252, value=63, step=21)
    
    st.divider()
    
    # Rolling Metrics
    st.subheader(f"📈 Rolling {window}-Day Metrics")
    
    # Calculate rolling metrics with safe division
    rolling_ret = returns.rolling(window).mean() * 252 * 100
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    rolling_sharpe = rolling_ret / rolling_vol.replace(0, np.nan)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['Annualized Return (%)', 'Volatility (%)', 'Sharpe Ratio'],
        vertical_spacing=0.08
    )
    
    fig.add_trace(
        go.Scatter(x=rolling_ret.index, y=rolling_ret.values,
                   line=dict(color=Config.colors['primary']), name='Return'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                   line=dict(color=Config.colors['warning']), name='Volatility'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                   line=dict(color=Config.colors['info']), name='Sharpe'),
        row=3, col=1
    )
    
    # Reference lines
    fig.add_hline(y=0, line_dash='dash', line_color=Config.colors['neutral'], row=1, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color=Config.colors['positive'], row=3, col=1)
    
    fig.update_layout(
        template=Config.chart_template,
        height=Config.chart_height_large,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Distribution Analysis
    st.subheader("📊 Returns Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            marker_color=Config.colors['primary'],
            opacity=0.7
        ))
        fig.add_vline(x=0, line_dash='dash', line_color=Config.colors['text'])
        fig.add_vline(x=returns.mean() * 100, line_dash='dot',
                      line_color=Config.colors['positive'],
                      annotation_text=f"Mean: {returns.mean()*100:.2f}%")
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height_small,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly box plot
        df_box = pd.DataFrame({
            'Return': returns * 100,
            'Month': returns.index.month_name().str[:3]
        })
        
        fig = px.box(
            df_box,
            x='Month',
            y='Return',
            color_discrete_sequence=[Config.colors['primary']]
        )
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height_small,
            yaxis_title="Daily Return (%)"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_factors() -> None:
    """Render factor analysis page."""
    st.title("🔬 Factor Analysis")
    st.divider()
    
    fi = st.session_state.app_data.get('feature_importance', pd.DataFrame())
    
    if fi.empty or 'feature' not in fi.columns or 'importance' not in fi.columns:
        st.warning("No feature importance data available")
        st.info("Make sure 'feature_importance.csv' exists with 'feature' and 'importance' columns")
        return
    
    # Controls
    max_features = min(30, len(fi))
    top_n = st.slider("Number of Features", min_value=5, max_value=max_features, value=min(15, max_features))
    
    top = fi.nlargest(top_n, 'importance')
    
    st.divider()
    
    # Bar chart
    st.subheader(f"📊 Top {top_n} Features")
    
    fig = px.bar(
        top,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        template=Config.chart_template,
        height=max(400, top_n * 25),
        yaxis=dict(categoryorder='total ascending'),
        showlegend=False,
        xaxis_title="Importance",
        yaxis_title="Feature"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Cumulative importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Cumulative Importance")
        
        sorted_fi = fi.sort_values('importance', ascending=False)
        total_importance = sorted_fi['importance'].sum()
        
        if total_importance > 0:
            cumulative = sorted_fi['importance'].cumsum() / total_importance * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative) + 1)),
                y=cumulative.values,
                fill='tozeroy',
                line=dict(color=Config.colors['primary'])
            ))
            fig.add_hline(y=80, line_dash='dash', line_color=Config.colors['warning'],
                          annotation_text="80%")
            fig.add_hline(y=95, line_dash='dash', line_color=Config.colors['negative'],
                          annotation_text="95%")
            fig.update_layout(
                template=Config.chart_template,
                height=Config.chart_height_small,
                xaxis_title="Number of Features",
                yaxis_title="Cumulative Importance (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Total importance is zero")
    
    with col2:
        st.subheader("📋 Feature Table")
        
        display = top.copy()
        display.insert(0, 'Rank', range(1, len(display) + 1))
        display['importance'] = display['importance'].apply(lambda x: f"{x:.6f}")
        
        st.dataframe(
            display[['Rank', 'feature', 'importance']],
            use_container_width=True,
            hide_index=True
        )


def render_trades() -> None:
    """Render trade analysis page."""
    st.title("💹 Trade Analysis")
    st.divider()
    
    trades = st.session_state.app_data.get('trades', pd.DataFrame())
    returns = st.session_state.app_data.get('returns', pd.Series())
    
    if trades.empty and returns.empty:
        st.warning("No trade or returns data available")
        return
    
    # If no trade-level P&L, use returns
    if trades.empty or 'pnl' not in trades.columns:
        st.info("Showing returns-based analysis (no trade-level P&L data)")
        
        if returns.empty:
            return
        
        # Monthly returns
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Monthly Returns")
            
            colors = [Config.colors['positive'] if r > 0 else Config.colors['negative'] 
                      for r in monthly.values]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly.index.strftime('%Y-%m'),
                y=monthly.values,
                marker_color=colors
            ))
            fig.add_hline(y=0, line_dash='dash', line_color=Config.colors['text'])
            fig.update_layout(
                template=Config.chart_template,
                height=Config.chart_height,
                xaxis_title="Month",
                yaxis_title="Return (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Cumulative Returns")
            
            cum_ret = (1 + returns).cumprod() - 1
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cum_ret.index,
                y=cum_ret.values * 100,
                fill='tozeroy',
                line=dict(color=Config.colors['primary'])
            ))
            fig.update_layout(
                template=Config.chart_template,
                height=Config.chart_height,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Trade-level analysis
    st.subheader("📊 Trade Statistics")
    
    total_pnl = trades['pnl'].sum()
    avg_pnl = trades['pnl'].mean()
    wins = (trades['pnl'] > 0).sum()
    losses = (trades['pnl'] <= 0).sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total P&L", format_currency(total_pnl))
    col2.metric("Avg P&L", format_currency(avg_pnl))
    col3.metric("Winning Trades", wins)
    col4.metric("Losing Trades", losses)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 P&L Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades['pnl'] / 1000,
            nbinsx=50,
            marker_color=Config.colors['primary']
        ))
        fig.add_vline(x=0, line_dash='dash', line_color=Config.colors['text'])
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height_small,
            xaxis_title="P&L ($K)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Cumulative P&L")
        
        cum_pnl = trades['pnl'].cumsum() / 1000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(cum_pnl))),
            y=cum_pnl.values,
            fill='tozeroy',
            line=dict(color=Config.colors['primary'])
        ))
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height_small,
            xaxis_title="Trade #",
            yaxis_title="Cumulative P&L ($K)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Trade log
    st.subheader("📋 Recent Trades")
    st.dataframe(trades.head(50), use_container_width=True)


def render_risk() -> None:
    """Render risk analysis page."""
    st.title("⚠️ Risk Analysis")
    st.divider()
    
    returns = st.session_state.app_data.get('returns', pd.Series())
    metrics = st.session_state.app_data.get('metrics', {})
    
    if returns.empty:
        st.warning("No returns data available")
        return
    
    # Risk Metrics
    st.subheader("📊 Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("VaR (95%)", format_pct(metrics.get('var_95', 0)))
    col2.metric("VaR (99%)", format_pct(metrics.get('var_99', 0)))
    col3.metric("CVaR (95%)", format_pct(metrics.get('cvar_95', 0)))
    col4.metric("Volatility (Ann.)", format_pct(metrics.get('volatility', 0)))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 VaR Distribution")
        
        var_95 = metrics.get('var_95', 0) * 100
        var_99 = metrics.get('var_99', 0) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            marker_color=Config.colors['primary'],
            opacity=0.7
        ))
        fig.add_vline(x=var_95, line_dash='dash', line_color=Config.colors['warning'],
                      annotation_text=f'VaR 95%: {var_95:.2f}%')
        fig.add_vline(x=var_99, line_dash='dash', line_color=Config.colors['negative'],
                      annotation_text=f'VaR 99%: {var_99:.2f}%')
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height,
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Rolling Volatility")
        
        rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol.values,
            line=dict(color=Config.colors['warning'])
        ))
        fig.add_hline(y=rolling_vol.mean(), line_dash='dash',
                      line_color=Config.colors['primary'],
                      annotation_text=f"Avg: {rolling_vol.mean():.1f}%")
        fig.update_layout(
            template=Config.chart_template,
            height=Config.chart_height,
            xaxis_title="Date",
            yaxis_title="21-Day Rolling Vol (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Risk metrics table
    st.subheader("📋 Risk Summary")
    
    risk_data = pd.DataFrame({
        'Metric': [
            'Annualized Volatility',
            'VaR (95%)',
            'VaR (99%)',
            'CVaR (95%)',
            'CVaR (99%)',
            'Skewness',
            'Kurtosis',
            'Best Day',
            'Worst Day',
            'Positive Days %'
        ],
        'Value': [
            format_pct(metrics.get('volatility', 0)),
            format_pct(metrics.get('var_95', 0)),
            format_pct(metrics.get('var_99', 0)),
            format_pct(metrics.get('cvar_95', 0)),
            format_pct(metrics.get('cvar_99', 0)),
            format_number(metrics.get('skewness', 0), 4),
            format_number(metrics.get('kurtosis', 0), 4),
            format_pct(metrics.get('best_day', 0)),
            format_pct(metrics.get('worst_day', 0)),
            format_pct((returns > 0).mean())
        ]
    })
    
    st.dataframe(risk_data, use_container_width=True, hide_index=True)


def render_data() -> None:
    """Render data explorer page."""
    st.title("📋 Data Explorer")
    st.divider()
    
    data = st.session_state.app_data
    
    # Build options based on available data
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
        st.warning("No data available to display")
        return
    
    selected = st.selectbox("Select Dataset", options)
    st.divider()
    
    if selected == "Equity Curve":
        df = data['equity_curve'].to_frame('portfolio_value')
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(),
            "equity_curve.csv",
            "text/csv"
        )
    
    elif selected == "Returns":
        df = data['returns'].to_frame('daily_return')
        df['cumulative_return'] = (1 + data['returns']).cumprod() - 1
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(),
            "returns.csv",
            "text/csv"
        )
    
    elif selected == "Trades":
        df = data['trades']
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "trades.csv",
            "text/csv"
        )
    
    elif selected == "Features":
        df = data['feature_importance']
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "features.csv",
            "text/csv"
        )
    
    elif selected == "Metrics":
        df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in data['metrics'].items()
        ])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "metrics.csv",
            "text/csv"
        )
    
    elif selected == "Validation":
        df = data['validation']
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            "validation.csv",
            "text/csv"
        )


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render sidebar with navigation and controls."""
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>📊 ML Alpha</h2>", unsafe_allow_html=True)
        st.divider()
        
        # Data loading section
        st.subheader("📁 Data")
        
        results_dir = st.text_input(
            "Results folder:",
            value=Config.default_results_dir,
            key="results_folder"
        )
        
        # Show found files
        results_path = Path(results_dir)
        if results_path.exists():
            files = [f.name for f in results_path.glob("*") if f.is_file()]
            if files:
                with st.expander(f"📂 {len(files)} files"):
                    for f in files[:10]:
                        st.caption(f"• {f}")
        else:
            st.warning("Folder not found")
        
        # Load button
        if st.button("🔄 Load Results", use_container_width=True, type="primary"):
            with st.spinner("Loading..."):
                loaded_data = load_results(results_dir)
            
            if loaded_data and not loaded_data.get('equity_curve', pd.Series()).empty:
                st.session_state.app_data = loaded_data
                st.success("✅ Loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Could not load results")
        
        # Auto-load on first run
        if st.session_state.app_data is None:
            auto_data = load_results(results_dir)
            if auto_data and not auto_data.get('equity_curve', pd.Series()).empty:
                st.session_state.app_data = auto_data
                st.rerun()
        
        st.divider()
        
        # Navigation (only if data loaded)
        if st.session_state.app_data is not None:
            st.subheader("🧭 Navigation")
            
            for icon, name, page_id in Config.pages:
                is_active = st.session_state.page == page_id
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(f"{icon} {name}", key=f"nav_{page_id}",
                             use_container_width=True, type=btn_type):
                    st.session_state.page = page_id
                    st.rerun()
            
            st.divider()
            
            # Quick stats
            st.subheader("📊 Quick Stats")
            metrics = st.session_state.app_data.get('metrics', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CAGR", format_pct(metrics.get('cagr', 0), 1))
                st.metric("Sharpe", format_number(metrics.get('sharpe_ratio', 0)))
            with col2:
                st.metric("Max DD", format_pct(metrics.get('max_drawdown', 0), 1))
                st.metric("Win Rate", format_pct(metrics.get('win_rate', 0), 0))
        
        st.divider()
        st.caption(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    """Main dashboard entry point."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not installed!")
        print("   Install with: pip install streamlit")
        sys.exit(1)
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not installed!")
        print("   Install with: pip install plotly")
        sys.exit(1)
    
    # Page config
    st.set_page_config(**Config.page_config)
    apply_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'overview'
    if 'app_data' not in st.session_state:
        st.session_state.app_data = None
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    if st.session_state.app_data is None:
        render_welcome()
    else:
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
        else:
            render_overview()


# =============================================================================
# PUBLIC API
# =============================================================================

class QuantDashboard:
    """
    Dashboard wrapper class for programmatic access.
    
    Example:
        >>> dashboard = QuantDashboard()
        >>> dashboard.run()
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None, data: Optional[Dict] = None):
        """
        Initialize dashboard.
        
        Args:
            config: Custom configuration
            data: Pre-loaded data
        """
        self.config = config or DashboardConfig()
        self.data = data
    
    def run(self) -> None:
        """Run the Streamlit dashboard."""
        main()


def run_dashboard() -> None:
    """Convenience function to run the dashboard."""
    main()


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()