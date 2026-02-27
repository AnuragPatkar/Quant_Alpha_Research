"""
TEST SCRIPT FOR MONITORING SUITE
Verifies Data Quality, Model Drift, Performance Tracking, and Alerts.
"""

import pandas as pd
import numpy as np
import os
import logging
import shutil
import subprocess
from datetime import datetime, timedelta
import collections

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Monitoring Components
from quant_alpha.monitoring.data_quality import DataQualityMonitor
from quant_alpha.monitoring.model_drift import ModelDriftDetector
from quant_alpha.monitoring.performance_tracker import PerformanceTracker
from quant_alpha.monitoring.alerts import AlertSystem

# --- CONFIGURATION ---
CACHE_PRED_PATH = r"E:\coding\quant_alpha_research\data\cache\ensemble_predictions.parquet"
CACHE_DATA_PATH = r"E:\coding\quant_alpha_research\data\cache\master_data_with_factors.parquet"

def load_real_data():
    if not os.path.exists(CACHE_PRED_PATH) or not os.path.exists(CACHE_DATA_PATH):
        logger.error("❌ Cache files not found. Please run 'run_trainer_and_ensemble.py' first.")
        return None, None
    
    logger.info("📂 Loading Real Data from Cache...")
    preds = pd.read_parquet(CACHE_PRED_PATH)
    data = pd.read_parquet(CACHE_DATA_PATH)
    
    # Ensure dates
    preds['date'] = pd.to_datetime(preds['date'])
    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    
    return preds, data

def test_data_quality():
    logger.info("\n🧪 Testing DataQualityMonitor with REAL DATA...")
    _, data = load_real_data()
    if data is None: return
    
    # Sort and Split: 70% Reference, 30% Incoming
    data = data.sort_values('date')
    dates = sorted(data['date'].unique())
    split_idx = int(len(dates) * 0.7)
    split_date = dates[split_idx]
    
    ref_data = data[data['date'] < split_date]
    incoming_data = data[data['date'] >= split_date]
    
    logger.info(f"Reference Data: {len(ref_data)} rows (until {split_date.date()})")
    monitor = DataQualityMonitor(reference_data=ref_data)
    
    # Test a specific day from incoming
    test_day = dates[-1]
    daily_batch = incoming_data[incoming_data['date'] == test_day]
    
    logger.info(f"Checking data for {test_day.date()}...")
    report = monitor.check_incoming_data(daily_batch, data_type='features')
    
    logger.info(f"Status: {report['status']}")
    if report['issues']:
        logger.info(f"Issues Found: {report['issues']}")
    if 'psi_scores' in report:
        # Show top 3 drifted features
        sorted_psi = sorted(report['psi_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"Top Drifted Features: {sorted_psi}")

def test_model_drift():
    logger.info("\n🧪 Testing ModelDriftDetector with REAL DATA...")
    preds, data = load_real_data()
    if preds is None: return

    detector = ModelDriftDetector(rolling_window=20)
    
    # Calculate Actual Returns (Next Day Return)
    if 'pnl_return' not in data.columns:
        data = data.sort_values(['ticker', 'date'])
        data['pnl_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
    
    # Merge
    if 'pnl_return' in preds.columns:
        preds = preds.drop(columns=['pnl_return'])
    merged = pd.merge(preds, data[['date', 'ticker', 'pnl_return']], on=['date', 'ticker'])
    merged = merged.dropna(subset=['ensemble_alpha', 'pnl_return'])
    
    dates = sorted(merged['date'].unique())
    logger.info(f"Simulating drift detection over {len(dates)} days...")
    
    for d in dates:
        day_slice = merged[merged['date'] == d]
        if day_slice.empty: continue
        
        # Pass as Series with ticker index
        preds_series = day_slice.set_index('ticker')['ensemble_alpha']
        actuals_series = day_slice.set_index('ticker')['pnl_return']
        
        detector.update(d.strftime('%Y-%m-%d'), preds_series, actuals_series)
        
    status = detector.detect_drift()
    logger.info(f"Final Drift Status: {status.get('drift_detected')}")
    if status.get('alerts'):
        logger.info(f"Alerts: {status.get('alerts')}")

def test_performance_tracker():
    logger.info("\n🧪 Testing PerformanceTracker with REAL DATA...")
    preds, data = load_real_data()
    if preds is None: return

    # Use large history to capture full backtest period
    tracker = PerformanceTracker(window_days=60, max_history=5000)
    
    # Calculate Actual Returns
    if 'pnl_return' not in data.columns:
        data = data.sort_values(['ticker', 'date'])
        data['pnl_return'] = data.groupby('ticker')['close'].shift(-1) / data['close'] - 1
    
    if 'pnl_return' in preds.columns:
        preds = preds.drop(columns=['pnl_return'])
    merged = pd.merge(preds, data[['date', 'ticker', 'pnl_return']], on=['date', 'ticker'])
    merged = merged.dropna(subset=['ensemble_alpha', 'pnl_return'])
    
    dates = sorted(merged['date'].unique())
    
    # Calculate daily market return (Equal Weighted Universe)
    market_returns = merged.groupby('date')['pnl_return'].mean()
    logger.info(f"Simulating performance tracking over {len(dates)} days...")
    
    # Volatility Targeting (Match Backtest Engine)
    TARGET_VOL = 0.15
    vol_window = collections.deque(maxlen=20)
    
    for d in dates:
        day_slice = merged[merged['date'] == d]
        if day_slice.empty: continue
        
        # Strategy: Top 20 Equal Weight
        top_picks = day_slice.sort_values('ensemble_alpha', ascending=False).head(20)
        if top_picks.empty: continue
        
        # Portfolio Return (Simple Average of Top 20)
        port_ret = top_picks['pnl_return'].mean()
        bench_ret = market_returns.loc[d]
        
        # Apply Volatility Control
        if len(vol_window) >= 20:
            current_vol = np.std(vol_window) * np.sqrt(252)
            if current_vol > 0:
                leverage = min(1.0, TARGET_VOL / current_vol)
            else:
                leverage = 1.0
        else:
            leverage = 1.0
            
        managed_ret = port_ret * leverage
        vol_window.append(port_ret) # Track RAW volatility
        
        # Pass as Dicts
        preds_dict = day_slice.set_index('ticker')['ensemble_alpha'].to_dict()
        actuals_dict = day_slice.set_index('ticker')['pnl_return'].to_dict()
        
        tracker.update(
            date=d.strftime('%Y-%m-%d'),
            predictions=preds_dict,
            actual_returns=actuals_dict,
            portfolio_return=managed_ret,
            benchmark_return=bench_ret,
            turnover=0.10, # Dummy turnover
            transaction_costs=0.001 # Dummy costs
        )
        
    status = tracker.get_status()
    logger.info("Final Performance Status:")
    for k, v in status.items():
        logger.info(f"  {k}: {v}")
        
    # Test Plotting
    os.makedirs("results", exist_ok=True)
    plot_path = "results/real_data_performance.png"
    tracker.plot_performance(save_path=plot_path)
    if os.path.exists(plot_path):
        logger.info(f"✅ Plot saved to {plot_path}")
    else:
        logger.error("❌ Plot generation failed")

def test_alerts():
    logger.info("\n🧪 Testing AlertSystem...")
    alerts = AlertSystem(env='development') # Should only log
    alerts.send('WARNING', 'Test Alert', 'This is a test warning from the test script.')
    logger.info("✅ Alert sent (check logs above)")

def launch_dashboard():
    logger.info("\n📊 Launching Streamlit Dashboard...")
    # Construct absolute path to dashboard.py
    dashboard_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "quant_alpha", "monitoring", "dashboard.py"))
    
    if os.path.exists(dashboard_path):
        logger.info(f"Running: streamlit run {dashboard_path}")
        try:
            # Use subprocess to run the command
            subprocess.run(["streamlit", "run", dashboard_path], check=True)
        except KeyboardInterrupt:
            logger.info("\nDashboard stopped by user.")
        except Exception as e:
            logger.error(f"Error launching dashboard: {e}")
    else:
        logger.error(f"❌ Dashboard file not found at: {dashboard_path}")

if __name__ == "__main__":
    test_data_quality()
    test_model_drift()
    test_performance_tracker()
    test_alerts()
    print("\n✅ All Monitoring Tests Completed.")
    
    # Launch dashboard directly
    launch_dashboard()
