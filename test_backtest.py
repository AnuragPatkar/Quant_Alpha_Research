"""
FAST BACKTEST RUNNER
Uses cached predictions to simulate portfolio performance instantly.
Use this when tweaking Risk Manager or Execution logic.
"""

import pandas as pd
import os
import logging
import warnings
import time
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
from quant_alpha.utils import setup_logging, load_parquet
setup_logging()

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.metrics import print_metrics_report
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from config.settings import config

# --- CONFIGURATION ---
CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

TOP_N_STOCKS = 25
TRANSACTION_COST = 0.001

def run_fast_backtest():
    if not os.path.exists(CACHE_PRED_PATH) or not os.path.exists(CACHE_DATA_PATH):
        logger.error("❌ Cache files not found. Please run 'run_trainer_and_ensemble.py' at least once to generate cache.")
        return

    # Check cache age
    cache_time = time.ctime(os.path.getmtime(CACHE_PRED_PATH))
    logger.info(f"🕒 Using Predictions Cached at: {cache_time}")
    logger.info("💡 If you changed model parameters, run 'run_trainer_and_ensemble.py' to update this cache.")

    logger.info("🚀 Loading Cached Data for Fast Backtest...")
    predictions = load_parquet(CACHE_PRED_PATH)
    data = load_parquet(CACHE_DATA_PATH)

    # Ensure dates are datetime
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # Handle Data Index/Columns
    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])

    # Prepare Data for Engine
    logger.info("⚙️ Preparing Data...")
    
    # Identify prediction column
    pred_col = 'ensemble_alpha'
    if pred_col not in predictions.columns:
        if 'prediction' in predictions.columns:
            pred_col = 'prediction'
        else:
            logger.error(f"❌ Prediction column '{pred_col}' not found in cache. Available: {predictions.columns.tolist()}")
            return

    backtest_preds = predictions[['date', 'ticker', pred_col]].rename(columns={pred_col: 'prediction'})
    backtest_preds = backtest_preds.drop_duplicates(subset=['date', 'ticker'])

    # Prepare Prices
    if 'volatility' not in data.columns:
        data['volatility'] = 0.02
    
    price_cols = ['date', 'ticker', 'close', 'volume', 'volatility']
    if 'sector' in data.columns:
        price_cols.append('sector')
    
    backtest_prices = data[price_cols].copy()
    backtest_prices = backtest_prices.drop_duplicates(subset=['date', 'ticker'])

    # Initialize Engine
    logger.info("🕹️ Initializing Backtest Engine...")
    engine = BacktestEngine(
        initial_capital=1_000_000,
        commission=TRANSACTION_COST,
        spread=0.0005,
        slippage=0.0002,
        position_limit=0.10, # FIX: Relaxed to 10% to allow Inverse-Vol weighting to work
        rebalance_freq='weekly',
        use_market_impact=True,
        target_volatility=0.15, # FIX: Lowered from 0.30 to 0.15 to reduce Drawdown
        max_adv_participation=0.02,
        trailing_stop_pct=getattr(config, 'TRAILING_STOP_PCT', 0.10) # NEW: Enable Trailing Stop in Fast Backtest
    )

    # Run Simulation
    logger.info("🏃 Running Simulation...")
    results = engine.run(
        predictions=backtest_preds,
        prices=backtest_prices,
        top_n=TOP_N_STOCKS
    )

    # Log Backtest Period
    m = results['metrics']
    logger.info(f"📅 Backtest Period: {m.get('start_date')} to {m.get('end_date')} ({m.get('trading_days')} days)")

    # Report
    print_metrics_report(results['metrics'])
    
    # --- NEW: Save Detailed Trade Report ---
    trade_report_path = config.RESULTS_DIR / "detailed_trade_report_fast.csv"
    if not results['trades'].empty:
        results['trades'].to_csv(trade_report_path, index=False)
        logger.info(f"📄 Detailed Trade Report Saved: {trade_report_path}")

    # Simple Attribution
    simple_attr = SimpleAttribution()
    simple_results = simple_attr.analyze_pnl_drivers(results['trades'])
    
    print(f"\n[ PnL Attribution ]")
    print(f"  Hit Ratio:     {simple_results.get('hit_ratio',0):.2%}")
    print(f"  Win/Loss Ratio:{simple_results.get('win_loss_ratio',0):.2f}")
    print(f"  Long PnL:      ${simple_results.get('long_pnl_contribution',0):,.0f}")
    print(f"  Short PnL:     ${simple_results.get('short_pnl_contribution',0):,.0f}")
    
    # Factor Attribution (IC Analysis)
    logger.info("🔍 Calculating Information Coefficient (IC)...")
    
    # Calculate Forward Returns for IC
    # FIX: Calculate 5-day forward return for IC to match training target (shift(-5))
    # Previously comparing 5-day prediction vs 1-day return caused low IC.
    data_sorted = data.sort_values(['ticker', 'date']).copy()
    data_sorted['fwd_ret_5d'] = data_sorted.groupby('ticker')['close'].shift(-5) / data_sorted['close'] - 1
    
    # Merge pnl_return into predictions
    ic_df = pd.merge(backtest_preds, data_sorted[['date', 'ticker', 'fwd_ret_5d']], on=['date', 'ticker'], how='inner')
    
    if not ic_df.empty:
        factor_attr = FactorAttribution()
        factor_vals = ic_df.set_index(['date', 'ticker'])[['prediction']]
        fwd_rets = ic_df.set_index(['date', 'ticker'])[['fwd_ret_5d']]
        
        rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets, window=30)
        print(f"\n[ Factor Analysis (Predictive Power) ]")
        print(f"  Mean IC:       {rolling_ic.mean():.4f}  (Target: >0.05)")
        print(f"  IC Std Dev:    {rolling_ic.std():.4f}")
        print(f"  IR (IC/Std):   {rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() != 0 else 0:.2f}  (Target: >0.5)")

if __name__ == "__main__":
    run_fast_backtest()