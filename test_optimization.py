"""
TEST SCRIPT FOR OPTIMIZATION MODULE
Verifies Mean-Variance, Risk Parity, Kelly, and Black-Litterman logic using REAL DATA.
Loads cached predictions and master data from 'run_trainer_and_ensemble.py'.
"""

import pandas as pd
import numpy as np
import os
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from quant_alpha.optimization.allocator import PortfolioAllocator
from config.settings import config

# Paths (Dynamic from Config)
CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

def load_real_data():
    if not os.path.exists(CACHE_PRED_PATH) or not os.path.exists(CACHE_DATA_PATH):
        logger.error("❌ Cache files not found. Please run 'run_trainer_and_ensemble.py' first to generate data.")
        return None, None

    logger.info(f"📂 Loading Real Data from Cache...")
    preds = pd.read_parquet(CACHE_PRED_PATH)
    data = pd.read_parquet(CACHE_DATA_PATH)
    
    # Ensure dates
    preds['date'] = pd.to_datetime(preds['date'])
    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    
    # Ensure unique index for pivoting
    data = data.drop_duplicates(subset=['date', 'ticker'])
    
    return preds, data

def prepare_optimization_inputs(preds, data, target_date, top_n=25):
    logger.info(f"📅 Preparing Optimization Inputs for {target_date.date()}...")
    
    # 1. Select Universe (Top N by Alpha)
    day_preds = preds[preds['date'] == target_date].copy()
    if day_preds.empty:
        logger.error(f"No predictions found for {target_date}")
        return None, None, None
        
    # Sort by alpha and take top N
    top_picks = day_preds.sort_values('ensemble_alpha', ascending=False).head(top_n)
    tickers = top_picks['ticker'].tolist()
    
    # 2. Expected Returns (Alpha)
    # Using ensemble_alpha (0-1 rank) as proxy for expected return magnitude
    # In a real live scenario, you might map this to an annualized return target (e.g. 10% to 30%)
    expected_returns = top_picks.set_index('ticker')['ensemble_alpha'].to_dict()
    
    # 3. Covariance Matrix (Historical)
    # Lookback 1 year (252 trading days)
    start_date = target_date - pd.Timedelta(days=365)
    mask = (data['date'] >= start_date) & (data['date'] < target_date) & (data['ticker'].isin(tickers))
    hist_data = data[mask]
    
    # Pivot to wide format (Date x Ticker)
    price_matrix = hist_data.pivot(index='date', columns='ticker', values='close')
    
    # Calculate Returns
    returns_matrix = price_matrix.pct_change().dropna()
    
    # Annualized Covariance
    covariance_matrix = returns_matrix.cov() * 252
    
    # Handle missing data in covariance (if some stocks are new or have gaps)
    # Intersect tickers again to ensure alignment
    valid_tickers = list(set(tickers) & set(covariance_matrix.columns))
    covariance_matrix = covariance_matrix.loc[valid_tickers, valid_tickers]
    
    # Filter expected returns to match valid tickers
    expected_returns = {t: expected_returns[t] for t in valid_tickers}
    
    # 4. Market Caps (for Black-Litterman)
    market_caps = {}
    # Check if market_cap column exists, otherwise use dummy or skip
    if 'market_cap' in data.columns:
        mc_data = data[(data['date'] == target_date) & (data['ticker'].isin(valid_tickers))]
        market_caps = mc_data.set_index('ticker')['market_cap'].to_dict()
    
    # Fill missing with a default value to prevent BL crash
    for t in valid_tickers:
        if t not in market_caps or pd.isna(market_caps[t]):
            market_caps[t] = 1e9 # Default 1B
            
    return expected_returns, covariance_matrix, market_caps

def print_weights(weights):
    if not weights:
        print("   ❌ Optimization Failed or Returned Empty.")
        return
    
    # Sort by weight
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"   Allocated Assets: {len(sorted_w)}")
    print(f"   Top 5 Allocations:")
    for t, w in sorted_w[:5]:
        print(f"     {t:<6}: {w:.2%}")
    print(f"   Total Weight: {sum(weights.values()):.2%}")

def run_real_optimization():
    print("\n" + "="*60)
    print("🧪 TESTING PORTFOLIO OPTIMIZATION WITH REAL DATA")
    print("="*60)

    # 1. Load Data
    preds, data = load_real_data()
    if preds is None: return

    # 2. Pick the latest date available in predictions
    latest_date = preds['date'].max()
    print(f"📅 Latest Prediction Date: {latest_date.date()}")
    
    # 3. Prepare Inputs
    expected_returns, covariance_matrix, market_caps = prepare_optimization_inputs(preds, data, latest_date)
    
    if not expected_returns:
        logger.error("Failed to prepare inputs.")
        return

    print(f"\n📊 Optimization Universe: {len(expected_returns)} Assets (Top Alpha Picks)")
    
    # 4. Run Optimizers
    
    print("\n" + "-"*40)
    print("1️⃣  Testing Mean-Variance (Markowitz)")
    print("-"*40)
    allocator_mvo = PortfolioAllocator(method='mean_variance', risk_aversion=2.5)
    weights_mvo = allocator_mvo.allocate(expected_returns, covariance_matrix)
    print_weights(weights_mvo)

    print("\n" + "-"*40)
    print("2️⃣  Testing Risk Parity (Equal Risk)")
    print("-"*40)
    allocator_rp = PortfolioAllocator(method='risk_parity')
    weights_rp = allocator_rp.allocate(expected_returns, covariance_matrix)
    print_weights(weights_rp)

    print("\n" + "-"*40)
    print("3️⃣  Testing Kelly Criterion (Fractional)")
    print("-"*40)
    allocator_kelly = PortfolioAllocator(method='kelly', fraction=0.5)
    weights_kelly = allocator_kelly.allocate(expected_returns, covariance_matrix, risk_free_rate=0.04)
    print_weights(weights_kelly)

    print("\n" + "-"*40)
    print("4️⃣  Testing Black-Litterman")
    print("-"*40)
    allocator_bl = PortfolioAllocator(method='black_litterman', tau=0.05)
    weights_bl = allocator_bl.allocate(expected_returns, covariance_matrix, market_caps=market_caps, confidence_level=0.6)
    print_weights(weights_bl)

if __name__ == "__main__":
    run_real_optimization()