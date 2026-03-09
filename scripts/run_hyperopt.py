"""
run_hyperopt.py
===============
Hyperparameter Optimization Script
----------------------------------
Runs Bayesian optimization (Optuna) for LightGBM, XGBoost, and CatBoost models
using a walk-forward validation scheme.
"""

import sys
import optuna
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Core Quant Alpha Imports
from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.utils import setup_logging
from quant_alpha.utils.preprocessing import WinsorisationScaler, SectorNeutralScaler
from quant_alpha.models.hyperopt import HyperparameterOptimizer # ✅ Import instead of define

# Logging setup
setup_logging()
logger = logging.getLogger("Hyperopt")

def run_hyperopt():
    logger.info("🚀 Starting Full Ensemble Optimization...")
    dm = DataManager()
    data = dm.get_master_data()

    # Ensure 'date' is a column and in datetime format
    if 'date' not in data.columns:
        data = data.reset_index()
    
    if 'date' not in data.columns:
        # Fallback for potential capitalization issues
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'date'}, inplace=True)
        
    data['date'] = pd.to_datetime(data['date'])
    
    # Target & Feature Prep
    data = data.sort_values(['ticker', 'date'])
    
    if 'open' in data.columns:
        next_open = data.groupby('ticker')['open'].shift(-1)
        future_open = data.groupby('ticker')['open'].shift(-6)
        data['raw_ret_5d'] = (future_open / next_open) - 1
    else:
        data['raw_ret_5d'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    
    # Align with Production (Sector Neutral Target)
    sector_mean = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('mean')
    data['target'] = data['raw_ret_5d'] - sector_mean
    
    data = data.dropna(subset=['target'])
    
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0', 'raw_ret_5d']
    
    # Identify Numeric vs Categorical features for preprocessing
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude]
    
    # All features (including categorical) for model training
    features = [c for c in data.columns if c not in exclude]
    
    # 🧹 Apply Winsorization (Shared Logic)
    logger.info("🧹 Applying Winsorization (Shared Logic)...")
    data = WinsorisationScaler(clip_pct=0.01).fit(data, numeric_features).transform(data, numeric_features)

    # ⚖️ Apply Sector-Neutral Normalization (Shared Logic)
    logger.info("⚖️ Applying Sector-Neutral Normalization (Shared Logic)...")
    data = SectorNeutralScaler(sector_col="sector").fit(data, numeric_features).transform(data, numeric_features)

    # Models to optimize
    models_to_run = {
        "LightGBM": LightGBMModel,
        "XGBoost": XGBoostModel,
        "CatBoost": CatBoostModel
    }

    global_best_params = {}

    for name, m_class in models_to_run.items():
        opt = HyperparameterOptimizer(model_class=m_class, model_name=name)
        # Testing with 20 trials per model
        best = opt.optimize(data, features, 'target', n_trials=20)
        global_best_params[name] = best

    print("\n" + "═"*45)
    print("🏆 FINAL MULTI-MODEL SUMMARY")
    print("═"*45)
    for model, params in global_best_params.items():
        print(f"✅ {model}: {params}")
    print("═"*45)

if __name__ == "__main__":
    run_hyperopt()


"""
═════════════════════════════════════════════
🏆 FINAL MULTI-MODEL SUMMARY
═════════════════════════════════════════════
✅ LightGBM: {'num_leaves': 46, 'max_depth': 4, 'learning_rate': 0.023087828391001746, 'reg_lambda': 68.09528642110561, 'subsample': 0.7604141133605463}
✅ XGBoost: {'max_depth': 6, 'learning_rate': 0.020859390326749598, 'reg_lambda': 83.12165551680617, 'min_child_weight': 9, 'subsample': 0.9293913220462858}
✅ CatBoost: {'subsample': 0.7599253260040655, 'depth': 4, 'learning_rate': 0.010188046124634994, 'l2_leaf_reg': 13.333479718677754}
═════════════════════════════════════════════
"""