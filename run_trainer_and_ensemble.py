import pandas as pd
import os
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

def run_final_alpha_pipeline():
    logger.info("🚀 Starting Final Leak-Free Alpha Pipeline...")
    dm = DataManager()
    
    # 1. Data Loading & Preparation
    data = dm.get_master_data()
    if data.empty:
        logger.error("❌ Master Data is empty!")
        return
    
    # FIX: Ensure 'date' is a column for indexing
    if 'date' not in data.columns:
        data = data.reset_index()
    
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    
    # 🎯 Target: 5-day forward returns
    data['target'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    data = data.dropna(subset=['target'])

    # Identify Features
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0']
    features = [c for c in data.columns if c not in exclude]
    
    # Category cast for GBDT stability
    for col in ['sector', 'industry']:
        if col in data.columns:
            data[col] = data[col].astype('category')

    logger.info(f"🧠 Features: {len(features)} | Total Rows: {len(data)}")

    # 2. Walk-Forward Training (Out-of-Sample Generation)
    models_config = {
        "LightGBM": (LightGBMModel, {'n_jobs': 1, 'n_estimators': 500}),
        "XGBoost": (XGBoostModel, {'n_jobs': 1, 'n_estimators': 500}),
        "CatBoost": (CatBoostModel, {'thread_count': 1, 'iterations': 500})
    }

    oos_preds_master = {}
    
    for name, (model_class, params) in models_config.items():
        logger.info(f"🏃 Training {name} in Walk-Forward mode...")
        trainer = WalkForwardTrainer(
            model_class=model_class,
            min_train_months=36,
            test_months=6,
            step_months=6,
            window_type='sliding',
            embargo_days=21,
            model_params=params
        )
        # Each trainer returns a DF with [date, ticker, target, prediction]
        oos_preds_master[name] = trainer.train(data, features, 'target')

    # 3. Point-in-Time Ensemble Blending
    logger.info("🧬 Merging Model Predictions...")
    ensemble_df = None
    for name, df in oos_preds_master.items():
        temp_df = df[['date', 'ticker', 'target', 'prediction']].rename(columns={'prediction': f'pred_{name}'})
        if ensemble_df is None:
            ensemble_df = temp_df
        else:
            ensemble_df = pd.merge(ensemble_df, temp_df, on=['date', 'ticker', 'target'], how='inner')

    # 4. Cross-Sectional Rank Blending (The Fix for 'nan' and Scale)
    pred_cols = [f'pred_{name}' for name in models_config.keys()]
    
    def calculate_ranks(group):
        # Calculate percentile ranks within each date to normalize scales
        for col in pred_cols:
            # Handle cases where model predicts constant values (nan IC fix)
            if group[col].std() < 1e-9:
                group[f'rank_{col}'] = 0.5
            else:
                group[f'rank_{col}'] = group[col].rank(pct=True)
        
        # Blended Alpha Score
        group['ensemble_alpha'] = group[[f'rank_{col}' for col in pred_cols]].mean(axis=1)
        return group

    logger.info("📊 Generating Final Ensemble Ranks...")
    # Fix: Set date as index so it's preserved through the groupby operation
    final_results = ensemble_df.set_index('date').groupby(level='date', group_keys=False).apply(calculate_ranks).reset_index()

    # 5. Metrics & Reporting
    print("\n" + "="*55)
    print("🏁 FINAL OUT-OF-SAMPLE PERFORMANCE REPORT")
    print("="*55)

    for name in models_config.keys():
        col_name = f'pred_{name}'
        # Drop NaNs just for IC calculation if any exist
        valid_data = final_results.dropna(subset=[col_name, 'target'])
        ic, _ = spearmanr(valid_data[col_name], valid_data['target'])
        print(f"{name.ljust(15)} | OOS Rank IC: {ic:.4f}")

    final_ic, _ = spearmanr(final_results['ensemble_alpha'], final_results['target'])
    print("-" * 55)
    print(f"ENSEMBLE ALPHA  | OOS Rank IC: {final_ic:.4f}")
    print("="*55)

    # 6. Actionable Alpha: Tomorrow's Picks
    latest_date = final_results['date'].max()
    top_picks = final_results[final_results['date'] == latest_date].nlargest(5, 'ensemble_alpha')
    
    print(f"\n🚀 TOP 5 PICKS FOR NEXT PERIOD ({latest_date.date()}):")
    print("-" * 45)
    print(top_picks[['ticker', 'ensemble_alpha']].to_string(index=False))
    print("-" * 45)

    # 7. Visualization
    plot_equity(final_results)

def plot_equity(df):
    """Calculates and plots the strategy equity curve."""
    # Portfolio: Pick Top 10 stocks each rebalance date
    portfolio = df.set_index('date').groupby(level='date', group_keys=False).apply(lambda x: x.nlargest(10, 'ensemble_alpha'))
    
    # Equal weight returns
    strat_returns = portfolio.groupby(level='date')['target'].mean()
    
    # 0.1% Transaction cost per trade (conservative)
    strat_returns = strat_returns - 0.001 
    
    equity_curve = (1 + strat_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, color='#2ecc71', linewidth=2, label='Alpha Ensemble Strategy')
    plt.fill_between(equity_curve.index, 1, equity_curve, color='#2ecc71', alpha=0.1)
    plt.title("Institutional Alpha Ensemble: Out-of-Sample Equity Curve", fontsize=14)
    plt.ylabel("Growth of ₹1")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_final_alpha_pipeline()