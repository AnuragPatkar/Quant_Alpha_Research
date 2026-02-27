"""
Quant Alpha Production Dashboard
Visualizes real-time monitoring metrics: Performance, Drift, and Data Quality.

Run with: streamlit run quant_alpha/monitoring/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
import collections

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from quant_alpha.monitoring.performance_tracker import PerformanceTracker
from quant_alpha.monitoring.model_drift import ModelDriftDetector
from quant_alpha.monitoring.data_quality import DataQualityMonitor
from config.settings import config

# --- CONFIG ---
st.set_page_config(page_title="Quant Alpha Monitor", layout="wide")
CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

@st.cache_data
def load_data():
    if not os.path.exists(CACHE_PRED_PATH) or not os.path.exists(CACHE_DATA_PATH):
        return None, None
    
    preds = pd.read_parquet(CACHE_PRED_PATH)
    data = pd.read_parquet(CACHE_DATA_PATH)
    
    # Ensure dates
    preds['date'] = pd.to_datetime(preds['date'])
    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    
    # Calculate PnL Return if missing
    if 'pnl_return' not in data.columns:
        data = data.sort_values(['ticker', 'date'])
        data['pnl_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
        
    return preds, data

def run_performance_tracker(preds, data, rebalance_freq, cost_bps, target_vol=0.15, top_n=20):
    # Use large history to capture full backtest period
    tracker = PerformanceTracker(window_days=60, max_history=5000)
    
    if 'pnl_return' in preds.columns:
        preds = preds.drop(columns=['pnl_return'])
    merged = pd.merge(preds, data[['date', 'ticker', 'pnl_return']], on=['date', 'ticker'])
    merged = merged.dropna(subset=['ensemble_alpha', 'pnl_return'])
    dates = sorted(merged['date'].unique())
    
    # Calculate daily market return (Equal Weighted Universe)
    market_returns = merged.groupby('date')['pnl_return'].mean()
    
    # Volatility Targeting (Match Backtest Engine)
    vol_window = collections.deque(maxlen=20)
    
    # Strategy State
    current_positions = []
    days_since_rebalance = 0
    
    for i, d in enumerate(dates):
        day_slice = merged[merged['date'] == d]
        if day_slice.empty: continue
        
        # Determine Rebalance Logic
        should_rebalance = False
        if rebalance_freq == 'Daily':
            should_rebalance = True
        elif rebalance_freq == 'Weekly' and days_since_rebalance >= 5:
            should_rebalance = True
            days_since_rebalance = 0
        
        # Execute Rebalance
        daily_cost = 0.0
        if should_rebalance or i == 0:
            top_picks = day_slice.sort_values('ensemble_alpha', ascending=False).head(top_n)
            if not top_picks.empty:
                new_positions = top_picks['ticker'].tolist()
                
                # Calculate Turnover Cost (Simple approximation: 2 * cost for turnover)
                # If completely new portfolio, turnover is 100%
                if current_positions:
                    # Intersection
                    kept = set(current_positions).intersection(set(new_positions))
                    turnover_pct = 1.0 - (len(kept) / len(current_positions))
                    daily_cost = turnover_pct * 2 * (cost_bps / 10000)
                
                current_positions = new_positions
        
        days_since_rebalance += 1
        
        # Calculate Portfolio Return (Equal Weight on Held Positions)
        if current_positions:
            # Filter day_slice for held tickers
            held_slice = day_slice[day_slice['ticker'].isin(current_positions)]
            if not held_slice.empty:
                raw_ret = held_slice['pnl_return'].mean()
            else:
                raw_ret = 0.0
        else:
            raw_ret = 0.0
            
        bench_ret = market_returns.loc[d]
        
        # Apply Volatility Control (Risk Management)
        if target_vol is not None and len(vol_window) >= 20:
            current_vol = np.std(vol_window) * np.sqrt(252)
            if current_vol > 0:
                # Reduce leverage if volatility exceeds target
                leverage = min(1.0, target_vol / current_vol)
            else:
                leverage = 1.0
        else:
            leverage = 1.0
            
        # Net Return = (Raw Return * Leverage) - Costs
        managed_ret = (raw_ret * leverage) - daily_cost
        vol_window.append(raw_ret) # Track RAW volatility
        
        # For tracking, we pass the predictions for the *current* universe
        # This ensures IC is calculated on the active set or full set depending on preference
        # Here we pass full set to monitor signal quality globally
        
        tracker.update(
            date=d.strftime('%Y-%m-%d'),
            predictions=day_slice.set_index('ticker')['ensemble_alpha'].to_dict(),
            actual_returns=day_slice.set_index('ticker')['pnl_return'].to_dict(),
            portfolio_return=managed_ret,
            benchmark_return=bench_ret
        )
    return tracker

def run_drift_detector(preds, data):
    detector = ModelDriftDetector(rolling_window=20)
    if 'pnl_return' in preds.columns:
        preds = preds.drop(columns=['pnl_return'])
    merged = pd.merge(preds, data[['date', 'ticker', 'pnl_return']], on=['date', 'ticker'])
    merged = merged.dropna(subset=['ensemble_alpha', 'pnl_return'])
    dates = sorted(merged['date'].unique())
    
    drift_history = []
    
    for d in dates:
        day_slice = merged[merged['date'] == d]
        if day_slice.empty: continue
        
        preds_series = day_slice.set_index('ticker')['ensemble_alpha']
        actuals_series = day_slice.set_index('ticker')['pnl_return']
        
        detector.update(d.strftime('%Y-%m-%d'), preds_series, actuals_series)
        res = detector.detect_drift()
        res['date'] = d
        drift_history.append(res)
        
    return detector, pd.DataFrame(drift_history)

# --- UI ---
st.title("🚀 Quant Alpha Production Monitor")

preds, data = load_data()

if preds is None:
    st.error("Cache files not found. Run the pipeline first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 Performance", "⚠️ Model Drift", "🛡️ Data Quality"])

with tab1:
    # Sidebar Controls for "What-If" Analysis
    st.sidebar.header("Strategy Settings")
    rebalance_freq = st.sidebar.selectbox("Rebalance Frequency", ["Weekly", "Daily"], index=0)
    cost_bps = st.sidebar.slider("Transaction Costs (bps)", 0, 50, 10)
    top_n = st.sidebar.slider("Top N Stocks", 5, 50, 25)
    enable_vol_target = st.sidebar.checkbox("Enable Volatility Targeting", value=True)
    target_vol = st.sidebar.slider("Target Volatility", 0.05, 0.50, 0.15) if enable_vol_target else None
    
    st.header("Live Performance Tracking")
    st.caption(f"Simulating: {rebalance_freq} Rebalance | {cost_bps} bps Cost | 15% Vol Target")
    st.caption(f"Simulating: {rebalance_freq} | Top {top_n} | Equal Weight | {cost_bps} bps Cost")
    
    with st.spinner("Calculating Performance Metrics..."):
        tracker = run_performance_tracker(preds, data, rebalance_freq, cost_bps, target_vol, top_n)
        status = tracker.get_status()
        hist_df = tracker.get_history_df()
    
    # KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Rolling IC (60d)", f"{status.get('ic_rolling', 0):.4f}")
    col2.metric("Sharpe Ratio", f"{status.get('sharpe_rolling', 0):.2f}")
    col3.metric("Beta (Total)", f"{status.get('beta_total', 0):.2f}")
    col4.metric("Alpha (Total)", f"{status.get('alpha_total', 0):.2%}")
    col5.metric("Active Ret (Total)", f"{status.get('active_return_total_annual', 0):.2%}")
    col6.metric("Max Drawdown", f"{status.get('max_drawdown', 0):.2%}")
    
    # Charts
    st.subheader("Cumulative Returns")
    if not hist_df.empty:
        hist_df['cum_ret'] = (1 + hist_df['portfolio_return']).cumprod() - 1
        hist_df['cum_bench'] = (1 + hist_df['benchmark_return']).cumprod() - 1
        
        fig_ret = px.line(hist_df, x='date', y=['cum_ret', 'cum_bench'], 
                          labels={'value': 'Return', 'variable': 'Series'},
                          title="Portfolio vs Benchmark")
        st.plotly_chart(fig_ret, width="stretch")
        
        st.subheader("Rolling IC")
        hist_df['rolling_ic'] = hist_df['ic'].rolling(20).mean()
        fig_ic = px.line(hist_df, x='date', y='rolling_ic', title="Rolling 20-Day Information Coefficient (IC)")
        fig_ic.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="Target IC (0.05)")
        fig_ic.add_hline(y=0.02, line_dash="dot", line_color="orange", annotation_text="Minimum Threshold (0.02)")
        st.plotly_chart(fig_ic, width="stretch")

with tab2:
    st.header("Concept Drift Detection")
    with st.spinner("Analyzing Drift..."):
        detector, drift_df = run_drift_detector(preds, data)
        
    if not drift_df.empty:
        st.subheader("Prediction Distribution Shift")
        
        # Plot Prediction Mean over time
        hist_recs = pd.DataFrame(list(detector.history))
        fig_drift = px.line(hist_recs, x='date', y=['pred_mean', 'actual_mean'], 
                           title="Prediction Mean vs Actual Return Mean")
        st.plotly_chart(fig_drift, width="stretch")
        
        st.subheader("MSE Degradation")
        if 'mse' in hist_recs.columns:
            fig_mse = px.line(hist_recs, x='date', y='mse', title="Model Mean Squared Error (MSE)")
            st.plotly_chart(fig_mse, width="stretch")
            
        # Recent Alerts
        st.subheader("Recent Drift Alerts")
        recent_drifts = drift_df[drift_df['drift_detected'] == True].tail(10)
        if not recent_drifts.empty:
            st.dataframe(recent_drifts[['date', 'alerts']])
        else:
            st.success("No significant drift detected in recent history.")

with tab3:
    st.header("Data Quality Checks")
    
    # Simulate a check on the latest data
    latest_date = data['date'].max()
    latest_batch = data[data['date'] == latest_date]
    
    # Reference data (older)
    ref_data = data[data['date'] < latest_date - timedelta(days=30)]
    
    monitor = DataQualityMonitor(reference_data=ref_data)
    report = monitor.check_incoming_data(latest_batch, data_type='features')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Data Date", latest_date.strftime('%Y-%m-%d'))
    with col2:
        status_color = "normal"
        if report['status'] == 'FAIL': status_color = "inverse"
        st.metric("Quality Status", report['status'])
        
    if report['issues']:
        st.error(f"Issues Found: {len(report['issues'])}")
        st.write(report['issues'])
    else:
        st.success("No Data Integrity Issues Found.")
        
    if 'psi_scores' in report:
        st.subheader("Feature Drift (PSI Scores)")
        psi_df = pd.DataFrame(list(report['psi_scores'].items()), columns=['Feature', 'PSI'])
        psi_df = psi_df.sort_values('PSI', ascending=False).head(20)
        
        fig_psi = px.bar(psi_df, x='PSI', y='Feature', orientation='h', title="Top 20 Drifted Features")
        fig_psi.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Critical Drift")
        fig_psi.add_vline(x=0.10, line_dash="dash", line_color="orange", annotation_text="Minor Drift")
        st.plotly_chart(fig_psi, width="stretch")