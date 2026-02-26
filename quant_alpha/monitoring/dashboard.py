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

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from quant_alpha.monitoring.performance_tracker import PerformanceTracker
from quant_alpha.monitoring.model_drift import ModelDriftDetector
from quant_alpha.monitoring.data_quality import DataQualityMonitor

# --- CONFIG ---
st.set_page_config(page_title="Quant Alpha Monitor", layout="wide")
CACHE_PRED_PATH = r"E:\coding\quant_alpha_research\data\cache\ensemble_predictions.parquet"
CACHE_DATA_PATH = r"E:\coding\quant_alpha_research\data\cache\master_data_with_factors.parquet"

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

def run_performance_tracker(preds, data):
    tracker = PerformanceTracker(window_days=60)
    
    merged = pd.merge(preds, data[['date', 'ticker', 'pnl_return']], on=['date', 'ticker'])
    merged = merged.dropna(subset=['ensemble_alpha', 'pnl_return'])
    dates = sorted(merged['date'].unique())
    
    for d in dates:
        day_slice = merged[merged['date'] == d]
        if day_slice.empty: continue
        
        # Top 20 Equal Weight Strategy
        top_picks = day_slice.sort_values('ensemble_alpha', ascending=False).head(20)
        if top_picks.empty: continue
        
        port_ret = top_picks['pnl_return'].mean()
        
        preds_dict = day_slice.set_index('ticker')['ensemble_alpha'].to_dict()
        actuals_dict = day_slice.set_index('ticker')['pnl_return'].to_dict()
        
        tracker.update(
            date=d.strftime('%Y-%m-%d'),
            predictions=preds_dict,
            actual_returns=actuals_dict,
            portfolio_return=port_ret,
            benchmark_return=0.0005 # Dummy benchmark
        )
    return tracker

def run_drift_detector(preds, data):
    detector = ModelDriftDetector(rolling_window=20)
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
    st.header("Live Performance Tracking")
    with st.spinner("Calculating Performance Metrics..."):
        tracker = run_performance_tracker(preds, data)
        status = tracker.get_status()
        hist_df = tracker.get_history_df()
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rolling IC (60d)", f"{status.get('ic_rolling', 0):.4f}")
    col2.metric("Sharpe Ratio", f"{status.get('sharpe_rolling', 0):.2f}")
    col3.metric("Current Drawdown", f"{status.get('current_drawdown', 0):.2%}")
    col4.metric("Alpha (Annual)", f"{status.get('alpha_annual', 0):.2%}")
    
    # Charts
    st.subheader("Cumulative Returns")
    if not hist_df.empty:
        hist_df['cum_ret'] = (1 + hist_df['portfolio_return']).cumprod() - 1
        hist_df['cum_bench'] = (1 + hist_df['benchmark_return']).cumprod() - 1
        
        fig_ret = px.line(hist_df, x='date', y=['cum_ret', 'cum_bench'], 
                          labels={'value': 'Return', 'variable': 'Series'},
                          title="Portfolio vs Benchmark")
        st.plotly_chart(fig_ret, use_container_width=True)
        
        st.subheader("Rolling IC")
        hist_df['rolling_ic'] = hist_df['ic'].rolling(20).mean()
        fig_ic = px.line(hist_df, x='date', y=['ic', 'rolling_ic'], title="Daily & Rolling IC")
        fig_ic.add_hline(y=0.02, line_dash="dash", line_color="green", annotation_text="Target")
        st.plotly_chart(fig_ic, use_container_width=True)

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
        st.plotly_chart(fig_drift, use_container_width=True)
        
        st.subheader("MSE Degradation")
        if 'mse' in hist_recs.columns:
            fig_mse = px.line(hist_recs, x='date', y='mse', title="Model Mean Squared Error (MSE)")
            st.plotly_chart(fig_mse, use_container_width=True)
            
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
        st.plotly_chart(fig_psi, use_container_width=True)