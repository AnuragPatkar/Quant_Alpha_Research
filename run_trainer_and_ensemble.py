import pandas as pd
import os
import logging
import warnings
import numpy as np
from scipy.stats import spearmanr

# Set environment variable to suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.trainer import WalkForwardTrainer
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.models.ensemble import EnsembleModel

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_production_grade_backtest():
    logger.info("🚀 Starting Leak-Free Alpha Pipeline...")
    dm = DataManager()
    
    # 1. Data Prep
    data = dm.get_master_data()
    if data.empty: return

    if 'date' not in data.columns:
        logger.info("🔄 Resetting index to bring 'date' into columns...")
        data = data.reset_index()

    
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data['target'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    data = data.dropna(subset=['target'])

    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0']
    features = [c for c in data.columns if c not in exclude]
    
    for col in ['sector', 'industry']:
        if col in data.columns: data[col] = data[col].astype('category')

    # 2. Step 1: Generate Out-of-Sample (OOS) Predictions for each model
    models_config = {
        "LightGBM": (LightGBMModel, {'n_jobs': 1, 'n_estimators': 500}),
        "XGBoost": (XGBoostModel, {'n_jobs': 1, 'n_estimators': 500}),
        "CatBoost": (CatBoostModel, {'thread_count': 1, 'iterations': 500})
    }

    oos_preds_master = {}
    
    for name, (model_class, params) in models_config.items():
        logger.info(f"🏃 Running Walk-Forward for {name}...")
        trainer = WalkForwardTrainer(
            model_class=model_class,
            min_train_months=36,
            test_months=6,
            step_months=6, # Standard OOS testing step
            window_type='sliding',
            embargo_days=21,
            model_params=params
        )
        
        # trainer.train returns the OOS predictions DataFrame
        oos_preds_master[name] = trainer.train(data, features, 'target')

    # 3. Step 2: Merge OOS Predictions (The Point-in-Time Ensemble)
    logger.info("🧬 Blending Models into Point-in-Time Ensemble...")
    
    ensemble_df = None
    for name, df in oos_preds_master.items():
        temp_df = df[['date', 'ticker', 'target', 'prediction']].rename(columns={'prediction': f'pred_{name}'})
        if ensemble_df is None:
            ensemble_df = temp_df
        else:
            ensemble_df = pd.merge(ensemble_df, temp_df, on=['date', 'ticker', 'target'], how='inner')

    # 4. Step 3: Cross-Sectional Rank Blending (No Future Leakage)
    pred_cols = [f'pred_{name}' for name in models_config.keys()]
    
    def calculate_ensemble_rank(group):
        # Calculate rank pct for each model within this specific date
        for col in pred_cols:
            group[f'rank_{col}'] = group[col].rank(pct=True)
        
        # Average the ranks
        group['ensemble_alpha'] = group[[f'rank_{col}' for col in pred_cols]].mean(axis=1)
        return group

    logger.info("📊 Applying Cross-Sectional Ranking...")
    final_oos_results = ensemble_df.groupby('date').apply(calculate_ensemble_rank)

    # 5. Final Metrics Calculation
    print("\n" + "="*50)
    print("🏁 FINAL BACKTEST RESULTS (LEAK-FREE)")
    print("="*50)

    for name in models_config.keys():
        ic, _ = spearmanr(final_oos_results[f'pred_{name}'], final_oos_results['target'])
        print(f"{name.ljust(15)} OOS Rank IC: {ic:.4f}")

    ensemble_ic, _ = spearmanr(final_oos_results['ensemble_alpha'], final_oos_results['target'])
    print("-" * 50)
    print(f"ENSEMBLE ALPHA OOS Rank IC: {ensemble_ic:.4f}")
    print("="*50)

    # Latest Signal Check (For Tomorrow's Trade)
    latest_date = final_oos_results['date'].max()
    top_picks = final_oos_results[final_oos_results['date'] == latest_date].nlargest(5, 'ensemble_alpha')
    
    print(f"\n🚀 Top Picks for {latest_date.date()}:")
    print(top_picks[['ticker', 'ensemble_alpha']].to_string(index=False))

    return final_oos_results

if __name__ == "__main__":
    run_production_grade_backtest()