import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import logging
import warnings

# --- CONFIGURATION & SUPPRESSION ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.trainer import WalkForwardTrainer
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel

# --- RISK & PORTFOLIO PARAMETERS ---
TOP_N_STOCKS = 25
STOCK_STOP_LOSS = -0.20
SL_SLIPPAGE_PENALTY = -0.02  # 2% extra loss for gap-downs
PORTFOLIO_DD_EXIT = -0.25    # Exit market at 25% drawdown
PORTFOLIO_DD_REENTRY = -0.15 # Re-enter market when drawdown recovers to 15%
TRANSACTION_COST = 0.001     # 0.1% per trade

def calculate_ranks_robust(df):
    """Vectorized ranking across dates for maximum speed and robustness."""
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    
    # Calculate ranks for all model columns at once (vectorized)
    for col in pred_cols:
        df[f'rank_{col}'] = df.groupby('date')[col].rank(pct=True)
    
    # Robust average: Handle cases where some models might have failed (NaNs)
    rank_cols = [f'rank_{col}' for col in pred_cols]
    df['ensemble_alpha'] = df[rank_cols].mean(axis=1, skipna=True)
    df['ensemble_alpha'] = df['ensemble_alpha'].fillna(0.5)
    
    return df

def plot_performance_tear_sheet(df):
    logger.info("📊 Generating Performance Tear Sheet...")
    
    # 1. Top N Selection (Portfolio Construction)
    # Fix: Set date as index to preserve it during groupby apply
    portfolio = df.set_index('date').groupby(level='date', group_keys=False).apply(lambda x: x.nlargest(TOP_N_STOCKS, 'ensemble_alpha'))
    
    # 2. Realistic Target Adjustment (Stop Loss + Slippage)
    portfolio['target_adj'] = np.where(
        portfolio['target'] <= STOCK_STOP_LOSS, 
        STOCK_STOP_LOSS + SL_SLIPPAGE_PENALTY, 
        portfolio['target']
    )
    
    # 3. Strategy Returns Calculation (Scaled for 5-day overlap)
    daily_returns = portfolio.groupby(level='date')['target_adj'].mean() / 5 
    daily_returns = daily_returns - (TRANSACTION_COST / 5) # Pro-rated cost

    # 4. Drawdown-Based Risk Manager (Market Filter with Bug Fix)
    equity_curve = [1.0]
    peak = 1.0
    active = True
    active_flags = []
    
    virtual_equity = 1.0 
    virtual_peak = 1.0

    for ret in daily_returns:
        # Always track virtual performance to know when to re-enter
        virtual_equity *= (1 + ret)
        virtual_peak = max(virtual_peak, virtual_equity)
        virtual_dd = (virtual_equity - virtual_peak) / virtual_peak

        if active:
            new_val = equity_curve[-1] * (1 + ret)
            equity_curve.append(new_val)
            peak = max(peak, new_val)
            
            # Check for Exit Condition
            if virtual_dd <= PORTFOLIO_DD_EXIT:
                logger.warning(f"🚨 Drawdown Limit Reached ({virtual_dd:.2%}). Switching to Cash.")
                active = False
        else:
            # Sitting in cash: Wealth stays flat
            equity_curve.append(equity_curve[-1])
            
            # Check for Re-entry Condition
            if virtual_dd >= PORTFOLIO_DD_REENTRY:
                logger.info(f"🟢 Market Recovered ({virtual_dd:.2%}). Re-entering trades.")
                active = True
                peak = equity_curve[-1] # Reset real peak on re-entry
        
        active_flags.append(1 if active else 0)

    # Slice [1:] to match index length
    equity_series = pd.Series(equity_curve[1:], index=daily_returns.index)
    drawdown = (equity_series - equity_series.cummax()) / equity_series.cummax()

    # --- Metrics Logic ---
    total_return = (equity_series.iloc[-1] - 1) * 100
    ann_return = ((equity_series.iloc[-1]) ** (252 / len(equity_series)) - 1) * 100
    vol = equity_series.pct_change().std() * np.sqrt(252)
    sharpe = (ann_return / 100) / vol if vol != 0 else 0

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(equity_series, label='Strategy Equity', color='#27ae60', lw=2)
    ax1.set_title(f"Final Wealth: {equity_series.iloc[-1]:.2f}x | Sharpe: {sharpe:.2f}", fontsize=14)
    ax1.fill_between(equity_series.index, equity_series.min(), equity_series.max(), 
                     where=pd.Series(active_flags, index=equity_series.index)==0, color='red', alpha=0.15, label='In Cash (Filter Active)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.fill_between(drawdown.index, 0, drawdown, color='#c0392b', alpha=0.3)
    ax2.plot(drawdown, color='#c0392b', lw=1)
    ax2.set_title(f"Max Drawdown: {drawdown.min()*100:.2f}%", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print(f"\n✅ BACKTEST COMPLETE")
    print("="*40)
    print(f"Total Return:      {total_return:.2f}%")
    print(f"Annualized Return: {ann_return:.2f}%")
    print(f"Annualized Vol:    {vol*100:.2f}%")
    print(f"Max Drawdown:      {drawdown.min()*100:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print("="*40)

def run_production_pipeline():
    logger.info("🚀 Booting Quant Engine...")
    dm = DataManager()
    data = dm.get_master_data()
    
    if 'date' not in data.columns: 
        data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    
    data['target'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    data = data.dropna(subset=['target'])
    
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0']
    features = [c for c in data.columns if c not in exclude]
    for col in ['sector', 'industry']:
        if col in data.columns: data[col] = data[col].astype('category')

    # Model Params (Optimized for stability)
    models_config = {
        "LightGBM": (LightGBMModel, {'n_jobs': 1, 'n_estimators': 500, 'reg_lambda': 10.0}),
        "XGBoost": (XGBoostModel, {'n_jobs': 1, 'n_estimators': 500, 'reg_lambda': 10.0}),
        "CatBoost": (CatBoostModel, {'thread_count': 1, 'iterations': 500, 'l2_leaf_reg': 10.0})
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oos_preds_master = {}
        for name, (model_class, params) in models_config.items():
            trainer = WalkForwardTrainer(model_class=model_class, min_train_months=36, test_months=6, 
                                         step_months=6, window_type='sliding', embargo_days=21, model_params=params)
            oos_preds_master[name] = trainer.train(data, features, 'target')

    # Merge OOS predictions safely on date and ticker
    ensemble_df = None
    for name, df in oos_preds_master.items():
        temp_df = df[['date', 'ticker', 'target', 'prediction']].rename(columns={'prediction': f'pred_{name}'})
        if ensemble_df is None:
            ensemble_df = temp_df
        else:
            # Fix: Do not merge on 'target' (float) and drop target from right df to avoid duplication
            ensemble_df = pd.merge(ensemble_df, temp_df.drop(columns=['target']), on=['date', 'ticker'], how='inner')

    logger.info("📊 Processing Vectorized Cross-Sectional Ranking...")
    final_results = calculate_ranks_robust(ensemble_df)

    # Actionable Alpha for next period
    latest_date = final_results['date'].max()
    top_picks = final_results[final_results['date'] == latest_date].nlargest(TOP_N_STOCKS, 'ensemble_alpha')
    
    print(f"\n🚀 TOP {TOP_N_STOCKS} ALPHA PICKS FOR NEXT PERIOD ({latest_date.date()}):")
    print("-" * 55)
    # Print only top 10 in console to save space, but all 25 are calculated
    print(top_picks[['ticker', 'ensemble_alpha']].head(10).to_string(index=False)) 
    print("... (and 15 more)")
    print("-" * 55)

    plot_performance_tear_sheet(final_results)

if __name__ == "__main__":
    run_production_pipeline()