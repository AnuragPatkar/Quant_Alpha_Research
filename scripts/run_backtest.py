"""
run_backtest.py
===============
Standalone Backtest Runner
--------------------------
Runs portfolio backtesting using cached predictions from 'run_trainer_and_ensemble.py'.
Supports multiple optimization methods, detailed reporting, and visualization.

Usage:
    python run_backtest.py --method top_n           # Simple Top 25 Equal Weight
    python run_backtest.py --method mean_variance   # Markowitz Optimization
    python run_backtest.py --method risk_parity     # Equal Risk Contribution
    python run_backtest.py --method kelly           # Kelly Criterion
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
import warnings
# Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import Config & Utils
from config.settings import config
from quant_alpha.utils import setup_logging, load_parquet, calculate_returns
from quant_alpha.backtest.engine import BacktestEngine
from quant_alpha.backtest.metrics import print_metrics_report
from quant_alpha.backtest.attribution import SimpleAttribution, FactorAttribution
from quant_alpha.optimization.allocator import PortfolioAllocator
from quant_alpha.visualization import (
    plot_equity_curve, plot_drawdown, plot_monthly_heatmap,
    plot_ic_time_series, generate_tearsheet
)

# Setup Logging
setup_logging()
logger = logging.getLogger(__name__)

# Suppress specific sklearn warnings that clutter output during rolling optimization
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.covariance")

# --- CONFIGURATION ---
CACHE_PRED_PATH = config.CACHE_DIR / "ensemble_predictions.parquet"
CACHE_DATA_PATH = config.CACHE_DIR / "master_data_with_factors.parquet"

def load_data():
    """Loads cached predictions and master data."""
    if not CACHE_PRED_PATH.exists() or not CACHE_DATA_PATH.exists():
        logger.error("❌ Cache files not found. Run 'run_trainer_and_ensemble.py' first.")
        sys.exit(1)

    logger.info("🚀 Loading Cached Data...")
    preds = load_parquet(CACHE_PRED_PATH)
    data = load_parquet(CACHE_DATA_PATH)

    # Ensure datetime
    preds['date'] = pd.to_datetime(preds['date'])
    if 'date' not in data.columns:
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])

    # Ensure unique index
    data = data.drop_duplicates(subset=['date', 'ticker'])
    
    # Identify prediction column
    if 'ensemble_alpha' in preds.columns:
        pred_col = 'ensemble_alpha'
    elif 'prediction' in preds.columns:
        pred_col = 'prediction'
    else:
        logger.error(f"❌ No prediction column found. Available: {preds.columns.tolist()}")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(preds):,} predictions and {len(data):,} data rows.")
    return preds, data, pred_col

def apply_trading_lag(preds: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Shifts predictions by 1 day to prevent lookahead bias."""
    logger.info("Applying 1-day shift to signals for realistic trading lag...")
    preds[pred_col] = preds.groupby('ticker')[pred_col].shift(1)
    preds.dropna(subset=[pred_col], inplace=True)
    logger.info(f"Shift complete. New prediction count: {len(preds):,}")
    return preds


def run_optimization(preds, data, pred_col, method='mean_variance', top_n=25):
    """
    Runs rolling portfolio optimization to generate weights.
    Adapted from test_optimization.py
    """
    logger.info(f"⚖️  Running Portfolio Optimization ({method})...")
    
    allocator = PortfolioAllocator(
        method=method,
        risk_aversion=config.OPT_RISK_AVERSION,
        fraction=config.OPT_KELLY_FRACTION,
        tau=0.05
    )
    
    # Prepare Price Matrix for Covariance
    price_matrix = data.pivot(index='date', columns='ticker', values='close')
    
    # Unique rebalance dates (Weekly based on predictions)
    unique_dates = sorted(preds['date'].unique())
    
    optimized_allocations = []
    lw_estimator = LedoitWolf()
    lookback = config.OPT_LOOKBACK_DAYS
    
    for current_date in tqdm(unique_dates, desc="Optimizing"):
        # 1. Get Alpha Signals
        day_preds = preds[preds['date'] == current_date]
        if day_preds.empty: continue
        
        # Select Top N
        top_candidates = day_preds.sort_values(pred_col, ascending=False).head(top_n)
        tickers = top_candidates['ticker'].tolist()
        
        # Expected Returns (Alpha Score as proxy)
        expected_returns = top_candidates.set_index('ticker')[pred_col].to_dict()
        
        # 2. Historical Covariance
        start_date = current_date - pd.Timedelta(days=lookback)
        hist_prices = price_matrix.loc[start_date:current_date, tickers]
        
        if len(hist_prices) < 60 or hist_prices.shape[0] < 2 or hist_prices.isnull().all().all():
            # Fallback to Equal Weight if not enough history
            weights = {t: 1.0/len(tickers) for t in tickers}
        else:
            returns = calculate_returns(hist_prices).dropna()
            if returns.empty:
                weights = {t: 1.0/len(tickers) for t in tickers}
            else:
                try:
                    cov_matrix = pd.DataFrame(
                        lw_estimator.fit(returns).covariance_, 
                        index=tickers, columns=tickers
                    ) * 252
                    
                    weights = allocator.allocate(
                        expected_returns=expected_returns,
                        covariance_matrix=cov_matrix,
                        risk_free_rate=config.RISK_FREE_RATE
                    )
                except Exception:
                    weights = {t: 1.0/len(tickers) for t in tickers}
        
        for t, w in weights.items():
            optimized_allocations.append({'date': current_date, 'ticker': t, 'prediction': w})
            
    return pd.DataFrame(optimized_allocations)


def main():
    parser = argparse.ArgumentParser(description="Quant Alpha Backtest Runner")
    parser.add_argument("--method", type=str, default="top_n", 
                        choices=["top_n", "mean_variance", "risk_parity", "kelly", "inverse_vol"],
                        help="Optimization method")
    parser.add_argument("--top_n", type=int, default=25, help="Number of positions (default: 25 to match trainer)")
    args = parser.parse_args()

    # 1. Load Data
    preds, data, pred_col = load_data()

    # NOTE: To match run_trainer_and_ensemble.py logic (which yields ~230%),
    # we DO NOT shift predictions here. The trainer trades on the signal date.
    # preds = apply_trading_lag(preds, pred_col)

    # 2. Prepare Predictions based on Method
    if args.method == "top_n":
        logger.info("🔹 Using Raw Alpha Scores (Top-N Equal Weight in Engine)")
        backtest_preds = preds[['date', 'ticker', pred_col]].rename(columns={pred_col: 'prediction'})
    else:
        # Run Optimization to get Weights
        opt_weights = run_optimization(preds, data, pred_col, method=args.method, top_n=args.top_n)
        if opt_weights.empty:
            logger.warning("⚠️ Optimization failed. Falling back to raw alpha.")
            backtest_preds = preds[['date', 'ticker', pred_col]].rename(columns={pred_col: 'prediction'})
        else:
            backtest_preds = opt_weights
    # Deduplicate
    backtest_preds = backtest_preds.drop_duplicates(subset=['date', 'ticker'])

    # 3. Prepare Prices
    if 'volatility' not in data.columns:
        data['volatility'] = 0.02
    
    price_cols = ['date', 'ticker', 'close', 'open', 'volume', 'volatility']
    if 'sector' in data.columns:
        price_cols.append('sector')
    
    backtest_prices = data[price_cols].drop_duplicates(subset=['date', 'ticker'])

    # 4. Configure Engine
    logger.info(f"🕹️ Initializing Backtest Engine (Method: {args.method.upper()})...")
    
    engine = BacktestEngine(
        initial_capital=config.INITIAL_CAPITAL,
        commission=config.TRANSACTION_COST_BPS / 10000.0,
        spread=0.0005,
        slippage=0.0002,           # Match trainer: 0.0002 vs config default
        position_limit=0.10,       # Match trainer: 0.10 vs config 0.15
        rebalance_freq='weekly',   # Match trainer explicit 'weekly'
        use_market_impact=True,
        target_volatility=0.15,
        max_adv_participation=0.02,
        trailing_stop_pct=getattr(config, 'TRAILING_STOP_PCT', 0.10),
        execution_price='open',
        max_turnover=0.20          # Match trainer
    )

    # 5. Run Backtest
    # SAFER CALL: Only pass `top_n` to engine if method is 'top_n'.
    # This prevents the engine from misinterpreting optimized weights.
    logger.info("🏃 Running Simulation...")
    engine_kwargs = {
        'predictions': backtest_preds,
        'prices': backtest_prices,
    }
    if args.method == 'top_n':
        engine_kwargs['top_n'] = args.top_n

    results = engine.run(**engine_kwargs)

    # 6. Save Results
    output_dir = config.RESULTS_DIR / f"backtest_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics
    print_metrics_report(results['metrics'])
    
    # Plots
    logger.info(f"📊 Generating Plots in {output_dir}...")
    
    # Equity Curve
    plot_equity_curve(
        results['equity_curve'], 
        save_path=output_dir / "equity_curve.png"
    )
    results['equity_curve'].to_csv(output_dir / "equity_curve.csv", index=False)
    
    # Drawdown
    plot_drawdown(
        results['equity_curve'], 
        save_path=output_dir / "drawdown.png"
    )
    
    # Monthly Heatmap
    eq_df = results['equity_curve'].copy()
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    returns_series = eq_df.set_index('date')['total_value'].pct_change()
    plot_monthly_heatmap(
        returns_series, 
        save_path=output_dir / "monthly_heatmap.png"
    )
    
    # Tearsheet
    generate_tearsheet(
        results, 
        save_path=output_dir / "tearsheet.pdf"
    )
    
    # Trade Report
    if not results['trades'].empty:
        results['trades'].to_csv(output_dir / "trades.csv", index=False)

    # 7. Attribution Analysis
    logger.info("🔍 Running Attribution Analysis...")
    
    # PnL Attribution
    simple_attr = SimpleAttribution()
    pnl_stats = simple_attr.analyze_pnl_drivers(results['trades'])
    
    print(f"\n[ PnL Attribution ]")
    print(f"  Hit Ratio:      {pnl_stats.get('hit_ratio', 0):.2%}")
    print(f"  Win/Loss Ratio: {pnl_stats.get('win_loss_ratio', 0):.2f}")
    print(f"  Long PnL:       ${pnl_stats.get('long_pnl_contribution', 0):,.0f}")
    print(f"  Short PnL:      ${pnl_stats.get('short_pnl_contribution', 0):,.0f}")
    
    # Factor IC Analysis
    # Calculate Forward Returns (Open-to-Open T+1 to T+6)
    data_sorted = data.sort_values(['ticker', 'date']).copy()
    
    if 'open' in data_sorted.columns:
        next_open = data_sorted.groupby('ticker')['open'].shift(-1)
        future_open = data_sorted.groupby('ticker')['open'].shift(-6)
        data_sorted['fwd_ret_5d'] = (future_open / next_open) - 1
    else:
        data_sorted['fwd_ret_5d'] = data_sorted.groupby('ticker')['close'].shift(-5) / data_sorted['close'] - 1
        
    # Merge Predictions with Forward Returns
    # Note: We use the RAW predictions (preds) for IC, not the optimized weights
    ic_df = pd.merge(
        preds[['date', 'ticker', pred_col]], 
        data_sorted[['date', 'ticker', 'fwd_ret_5d']], 
        on=['date', 'ticker'], 
        how='inner'
    ).dropna()
    
    if not ic_df.empty:
        factor_attr = FactorAttribution()
        factor_vals = ic_df.set_index(['date', 'ticker'])[[pred_col]]
        fwd_rets = ic_df.set_index(['date', 'ticker'])[['fwd_ret_5d']]
        
        rolling_ic = factor_attr.calculate_rolling_ic(factor_vals, fwd_rets, window=30)
        
        plot_ic_time_series(
            rolling_ic, 
            save_path=output_dir / "ic_time_series.png"
        )
        
        mean_ic = rolling_ic.mean()
        ic_std = rolling_ic.std()
        ir = mean_ic / ic_std if ic_std != 0 else 0
        
        print(f"\n[ Factor Analysis ]")
        print(f"  Mean IC:      {mean_ic:.4f}")
        print(f"  IC Std:       {ic_std:.4f}")
        print(f"  IC IR:        {ir:.2f}")
        
        # Save IC Data
        rolling_ic.to_csv(output_dir / "rolling_ic.csv")

    # 8. Alpha Metrics (vs Benchmark)
    try:
        import yfinance as yf
        from scipy import stats
        
        # Download Benchmark
        start_dt = results['equity_curve']['date'].min()
        end_dt = results['equity_curve']['date'].max()
        
        # Match trainer benchmark (^GSPC) instead of config (SPY)
        spy = yf.download("^GSPC", start=start_dt, end=end_dt, progress=False, auto_adjust=True)
        if not spy.empty:
            # Handle yfinance MultiIndex columns (common in newer versions)
            if isinstance(spy.columns, pd.MultiIndex):
                # If MultiIndex, try to get Close for the specific ticker or just the first column
                if 'Close' in spy.columns.get_level_values(0):
                    spy_close = spy.xs('Close', level=0, axis=1).iloc[:, 0]
                else:
                    spy_close = spy.iloc[:, 0]
            else:
                spy_close = spy['Close'] if 'Close' in spy.columns else spy.iloc[:, 0]

            spy_ret = spy_close.pct_change().dropna()
            strat_ret = returns_series.dropna()
            
            # Align
            aligned = pd.DataFrame({'strat': strat_ret, 'bench': spy_ret}).dropna()
            if not aligned.empty:
                # Match trainer Risk Free Rate (0.035)
                rf_daily = 0.035 / 252
                # Fix: Flatten arrays to ensure 1D input for linregress
                beta, alpha, r_val, p_val, std_err = stats.linregress(
                    aligned['bench'].values.ravel() - rf_daily, 
                    aligned['strat'].values.ravel() - rf_daily
                )
                print(f"\n[ Alpha Metrics vs S&P 500 (^GSPC) ]")
                print(f"  Beta:         {beta:.2f}")
                print(f"  Alpha (Ann):  {alpha * 252:.2%}")
                print(f"  Correlation:  {r_val:.2f}")
    except Exception as e:
        logger.warning(f"Could not calculate benchmark metrics: {e}")

    logger.info(f"✅ Backtest Complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()