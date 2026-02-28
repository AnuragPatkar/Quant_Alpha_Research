"""
TEST SCRIPT FOR VISUALIZATION MODULE
Verifies that all plotting functions run correctly and generate outputs.
"""

import pandas as pd
import numpy as np
import os
import logging
import shutil
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

# Configure logging
from quant_alpha.utils import setup_logging, load_parquet, calculate_returns
setup_logging()
logger = logging.getLogger(__name__)

# Import Visualization Components
from quant_alpha.visualization import (
    plot_equity_curve, 
    plot_drawdown, 
    plot_monthly_heatmap,
    plot_interactive_equity,
    plot_ic_time_series,
    plot_quantile_returns,
    generate_tearsheet
)

# Import Core Components for Real Data Test
from config.settings import config
from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.attribution import FactorAttribution
from quant_alpha.optimization.allocator import PortfolioAllocator

# --- CONFIGURATION ---
OUTPUT_DIR = "results/test_viz"
REAL_OUTPUT_DIR = "results/test_viz_real"

def setup_environment():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"📂 Created output directory: {OUTPUT_DIR}")

def generate_dummy_data():
    logger.info("🎲 Generating dummy data for testing...")
    
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    n = len(dates)
    
    # Equity Curve
    returns = np.random.normal(0.0005, 0.01, n)
    # Start exactly at 1M
    initial_capital = 1_000_000
    equity = initial_capital * (1 + returns).cumprod()
    # Prepend initial state
    equity = np.insert(equity, 0, initial_capital)
    dates_extended = [dates[0] - timedelta(days=1)] + list(dates)
    
    equity_df = pd.DataFrame({
        'date': dates_extended,
        'total_value': equity,
        'return': np.insert(returns, 0, 0.0)
    })
    # Truncate to n to match other series for simplicity in this dummy generator
    equity_df = equity_df.iloc[1:].reset_index(drop=True)
    equity_df['date'] = dates # Align with other dataframes
    
    # Benchmark
    bench_returns = np.random.normal(0.0004, 0.008, n)
    bench_equity = 1_000_000 * (1 + bench_returns).cumprod()
    benchmark_df = pd.DataFrame({
        'date': dates,
        'close': bench_equity
    })
    
    # Trades
    trades_df = pd.DataFrame({
        'date': [dates[10], dates[50], dates[100], dates[200]],
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
        'side': ['buy', 'sell', 'buy', 'sell'],
        'shares': [100, 100, 50, 50],
        'price': [150, 160, 200, 210]
    })
    
    # Factor IC
    ic_series = pd.Series(np.random.normal(0.05, 0.1, n), index=dates)
    
    # Quantile Returns
    quantiles = pd.Series(
        [0.001, 0.002, 0.003, 0.004, 0.008], 
        index=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )
    
    return {
        'equity': equity_df,
        'benchmark': benchmark_df,
        'trades': trades_df,
        'ic': ic_series,
        'quantiles': quantiles
    }

def test_static_plots(data):
    logger.info("\n🧪 Testing Static Plots...")
    
    # Equity Curve
    save_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
    try:
        plot_equity_curve(data['equity'], data['benchmark'], save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ Equity curve saved: {save_path}")
        else:
            logger.error(f"❌ Equity curve file not created")
    except Exception as e:
        logger.error(f"❌ Equity curve failed: {e}")

    # Drawdown
    save_path = os.path.join(OUTPUT_DIR, "drawdown.png")
    try:
        plot_drawdown(data['equity'], save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ Drawdown plot saved: {save_path}")
        else:
            logger.error(f"❌ Drawdown plot file not created")
    except Exception as e:
        logger.error(f"❌ Drawdown plot failed: {e}")

    # Monthly Heatmap
    save_path = os.path.join(OUTPUT_DIR, "heatmap.png")
    try:
        returns_series = data['equity'].set_index('date')['return']
        plot_monthly_heatmap(returns_series, save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ Heatmap saved: {save_path}")
        else:
            logger.error(f"❌ Heatmap file not created")
    except Exception as e:
        logger.error(f"❌ Heatmap failed: {e}")

def test_interactive_plots(data):
    logger.info("\n🧪 Testing Interactive Plots...")
    
    try:
        fig = plot_interactive_equity(data['equity'], data['trades'])
        # We can't easily test display, but we can check if object is created
        if fig:
            logger.info("✅ Interactive equity plot object created")
            # Optionally save to HTML
            save_path = os.path.join(OUTPUT_DIR, "interactive_equity.html")
            fig.write_html(save_path)
            logger.info(f"   Saved to {save_path}")
        else:
            logger.error("❌ Interactive equity plot returned None")
    except Exception as e:
        logger.error(f"❌ Interactive plot failed: {e}")

def test_factor_viz(data):
    logger.info("\n🧪 Testing Factor Visualization...")
    
    # IC Time Series
    save_path = os.path.join(OUTPUT_DIR, "ic_timeseries.png")
    try:
        plot_ic_time_series(data['ic'], save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ IC plot saved: {save_path}")
        else:
            logger.error(f"❌ IC plot file not created")
    except Exception as e:
        logger.error(f"❌ IC plot failed: {e}")

    # Quantile Returns
    save_path = os.path.join(OUTPUT_DIR, "quantile_returns.png")
    try:
        plot_quantile_returns(data['quantiles'], save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ Quantile plot saved: {save_path}")
        else:
            logger.error(f"❌ Quantile plot file not created")
    except Exception as e:
        logger.error(f"❌ Quantile plot failed: {e}")

def test_reports(data):
    logger.info("\n🧪 Testing Reports...")
    
    save_path = os.path.join(OUTPUT_DIR, "tearsheet.pdf")
    
    # Mock results dictionary expected by generate_tearsheet
    results = {
        'equity_curve': data['equity'],
        'metrics': {
            'Sharpe': 1.5,
            'CAGR': 0.20,
            'Max DD': -0.10
        }
    }
    
    try:
        generate_tearsheet(results, save_path=save_path)
        if os.path.exists(save_path):
            logger.info(f"✅ Tearsheet saved: {save_path}")
        else:
            logger.error(f"❌ Tearsheet file not created")
    except Exception as e:
        logger.error(f"❌ Tearsheet generation failed: {e}")

def generate_optimized_weights(predictions, prices_df, method='mean_variance'):
    """
    Runs Portfolio Optimization on a rolling basis (Mean-Variance).
    Replicates logic from run_trainer_and_ensemble.py
    """
    logger.info(f"⚖️  Running Portfolio Optimization ({method})...")
    
    # Initialize Allocator
    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=getattr(config, 'OPT_RISK_AVERSION', 2.5),
        fraction=getattr(config, 'OPT_KELLY_FRACTION', 0.5),
        tau=0.05
    )
    
    # Prepare Data
    # Pivot prices for covariance: Index=Date, Columns=Ticker
    price_matrix = prices_df.pivot(index='date', columns='ticker', values='close')
    
    # Get unique rebalance dates (Weekly)
    unique_dates = sorted(predictions['date'].unique())
    
    optimized_allocations = []
    
    # Rolling Optimization Loop
    for current_date in tqdm(unique_dates, desc="Optimizing Portfolio"):
        # 1. Get Alpha Signals for this date
        day_preds = predictions[predictions['date'] == current_date]
        if day_preds.empty: continue
        
        # Select Top N candidates based on Alpha Score
        top_candidates = day_preds.sort_values('ensemble_alpha', ascending=False).head(25)
        tickers = top_candidates['ticker'].tolist()
        
        # Expected Returns
        expected_returns = top_candidates.set_index('ticker')['ensemble_alpha'].to_dict()
        
        # 2. Calculate Historical Covariance (Lookback Window)
        lookback = getattr(config, 'OPT_LOOKBACK_DAYS', 252)
        start_date = current_date - pd.Timedelta(days=lookback)
        
        # Slice price matrix for speed
        hist_prices = price_matrix.loc[start_date:current_date, tickers]
        
        if len(hist_prices) < 60: 
            weights = {t: 1.0/len(tickers) for t in tickers}
        else:
            returns = calculate_returns(hist_prices).dropna()
            if returns.empty:
                weights = {t: 1.0/len(tickers) for t in tickers}
            else:
                cov_matrix = returns.cov() * 252 # Annualized
                
                weights = allocator.allocate(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    market_caps={}, # Simplified for MV
                    risk_free_rate=getattr(config, 'RISK_FREE_RATE', 0.04)
                )
        
        # 4. Store Results
        for ticker, w in weights.items():
            optimized_allocations.append({'date': current_date, 'ticker': ticker, 'optimized_weight': w})
            
    return pd.DataFrame(optimized_allocations)

def test_with_real_data():
    """
    Integration Test: Loads cached production data, runs a fast backtest,
    and verifies visualization on REAL data structures.
    """
    logger.info("\n🚀 Testing Visualization with REAL CACHED DATA...")
    
    # 1. Check for Cache
    pred_path = config.CACHE_DIR / "ensemble_predictions.parquet"
    data_path = config.CACHE_DIR / "master_data_with_factors.parquet"
    
    if not pred_path.exists() or not data_path.exists():
        logger.warning(f"⚠️ Real data cache not found at {config.CACHE_DIR}. Skipping real data test.")
        logger.warning("   Run 'run_trainer_and_ensemble.py' first to generate cache.")
        return

    try:
        # 2. Load Data
        logger.info("   Loading cached data...")
        preds = load_parquet(pred_path)
        data = load_parquet(data_path)
        
        # Ensure types
        preds['date'] = pd.to_datetime(preds['date'])
        if 'date' not in data.columns: data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        
        # 3. Prepare for Backtest (Simplified logic from test_backtest.py)
        # Prediction Column
        if 'ensemble_alpha' not in preds.columns and 'prediction' in preds.columns:
            preds = preds.rename(columns={'prediction': 'ensemble_alpha'})
        
        # Ensure we have ensemble_alpha
        if 'ensemble_alpha' not in preds.columns:
            logger.error("❌ 'ensemble_alpha' column missing in predictions.")
            return
        
        # Price Data
        if 'volatility' not in data.columns: data['volatility'] = 0.02
        price_cols = ['date', 'ticker', 'close', 'open', 'volume', 'volatility']
        if 'sector' in data.columns: price_cols.append('sector')
        
        backtest_prices = data[price_cols].drop_duplicates(subset=['date', 'ticker'])
        
        # 4. Run Optimization (Mean-Variance) to match Production
        logger.info("   Running Mean-Variance Optimization (Replicating Production)...")
        opt_weights = generate_optimized_weights(preds, backtest_prices, method='mean_variance')
        
        if opt_weights.empty:
            logger.warning("⚠️ Optimization returned empty. Using raw equal weights.")
            backtest_preds = preds[['date', 'ticker', 'ensemble_alpha']].rename(columns={'ensemble_alpha': 'prediction'})
        else:
            backtest_preds = opt_weights.rename(columns={'optimized_weight': 'prediction'})
            
        backtest_preds = backtest_preds.drop_duplicates(subset=['date', 'ticker'])
        
        # 5. Run Backtest with Production Settings
        logger.info("   Running Backtest Engine...")
        engine = BacktestEngine(
            initial_capital=1_000_000,
            commission=getattr(config, 'TRANSACTION_COST_BPS', 10.0) / 10000,
            spread=0.0005,
            slippage=0.0002,
            position_limit=0.10,
            rebalance_freq='weekly',
            use_market_impact=True,
            target_volatility=0.15,
            max_adv_participation=0.02,
            trailing_stop_pct=getattr(config, 'TRAILING_STOP_PCT', 0.10),
            execution_price='open', # Trade at Next Open (Realistic)
            max_turnover=0.20       # Limit turnover to 20% per rebalance
        )
        results = engine.run(backtest_preds, backtest_prices, top_n=25)
        
        # 6. Generate Plots (Mirroring run_trainer_and_ensemble.py logic)
        os.makedirs(REAL_OUTPUT_DIR, exist_ok=True)
        logger.info(f"   Generating plots in {REAL_OUTPUT_DIR}...")
        
        # Equity Curve
        plot_equity_curve(results['equity_curve'], save_path=os.path.join(REAL_OUTPUT_DIR, "equity_curve.png"))
        
        # Drawdown
        plot_drawdown(results['equity_curve'], save_path=os.path.join(REAL_OUTPUT_DIR, "drawdown.png"))
        
        # Monthly Heatmap (Calculated from Equity Curve)
        eq_df = results['equity_curve'].copy()
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        # Calculate returns from total_value
        returns_series = eq_df.set_index('date')['total_value'].pct_change()
        plot_monthly_heatmap(returns_series, save_path=os.path.join(REAL_OUTPUT_DIR, "monthly_heatmap.png"))
        
        # Tearsheet
        generate_tearsheet(results, save_path=os.path.join(REAL_OUTPUT_DIR, "tearsheet.pdf"))
        
        # 7. Factor Analysis (IC)
        logger.info("   Calculating Real Factor IC...")
        # Calculate 5-day forward return for IC (matching production target)
        data_sorted = data.sort_values(['ticker', 'date']).copy()
        data_sorted['fwd_ret_5d'] = data_sorted.groupby('ticker')['close'].shift(-5) / data_sorted['close'] - 1
        
        # Merge
        # Use original predictions for IC, not optimized weights
        ic_data = pd.merge(preds, data_sorted[['date', 'ticker', 'fwd_ret_5d']], on=['date', 'ticker'], how='inner')
        ic_data = ic_data.dropna()
        
        if not ic_data.empty:
            factor_attr = FactorAttribution()
            factor_vals = ic_data.set_index(['date', 'ticker'])[['ensemble_alpha']]
            fwd_rets = ic_data.set_index(['date', 'ticker'])[['fwd_ret_5d']]
            
            rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets, window=30)
            
            plot_ic_time_series(rolling_ic, save_path=os.path.join(REAL_OUTPUT_DIR, "ic_time_series.png"))
            
            # Quantile Returns (Simple proxy)
            ic_data['quantile'] = pd.qcut(ic_data['ensemble_alpha'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            q_rets = ic_data.groupby('quantile', observed=False)['fwd_ret_5d'].mean()
            plot_quantile_returns(q_rets, save_path=os.path.join(REAL_OUTPUT_DIR, "quantile_returns.png"))
            
        logger.info("✅ Real Data Visualization Test Completed.")
        
    except Exception as e:
        logger.error(f"❌ Real Data Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_environment()
    data = generate_dummy_data()
    
    test_static_plots(data)
    test_interactive_plots(data)
    test_factor_viz(data)
    test_reports(data)
    
    test_with_real_data()
    
    print(f"\n✅ Visualization tests completed.\n   - Dummy Data Plots: {OUTPUT_DIR}\n   - Real Data Plots:  {REAL_OUTPUT_DIR}")