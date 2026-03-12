"""
Bayesian Hyperparameter Optimization Pipeline
=============================================
Orchestrates the automated tuning of Gradient Boosted Decision Tree (GBDT) models
using Tree-structured Parzen Estimator (TPE) algorithms.

Purpose
-------
This module serves as the **Model Selection & Tuning Layer**, systematically exploring
the hyperparameter space to maximize Out-of-Sample (OOS) performance. It employs
a rigorous Walk-Forward Validation scheme to prevent look-ahead bias and overfitting,
optimizing for the Information Coefficient (IC) of the alpha signal.

Key capabilities:
1.  **Search Strategy**: Uses Optuna for efficient Bayesian optimization, prioritizing
    regions of the hyperparameter space likely to yield improvements.
2.  **Regime Robustness**: Validates parameters over expanding time windows to ensure
    stability across changing market conditions.
3.  **Ensemble Tuning**: Independently optimizes LightGBM, XGBoost, and CatBoost to
    maintain diversity in the final production ensemble.

Usage:
------
Executed via CLI to update the optimal parameter configurations.

.. code-block:: bash

    python scripts/run_hyperopt.py

Importance
----------
-   **Alpha Maximization**: Directly targets the optimization of the signal-to-noise ratio.
    -   **Overfitting Mitigation**: Penalizes excessive model complexity via regularization
        parameter tuning (L1/L2, depth, subsample).

Tools & Frameworks
------------------
-   **Optuna**: Bayesian optimization framework (TPE sampler).
-   **Gradient Boosting**: LightGBM, XGBoost, CatBoost.
-   **Pandas/NumPy**: Vectorized data manipulation.
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

    # Schema Validation: Enforce strict temporal indexing.
    if 'date' not in data.columns:
        data = data.reset_index()
    
    if 'date' not in data.columns:
        # Fallback for potential capitalization issues
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'date'}, inplace=True)
        
    data['date'] = pd.to_datetime(data['date'])
    
    # Target & Feature Preparation
    data = data.sort_values(['ticker', 'date'])
    
    # Forward Return Construction:
    # Calculates $R_{t+1 \to t+6}$ (5-day forward return).
    # Logic handles both Open-to-Open (tradeable) and Close-to-Close (indicative) prices.
    if 'open' in data.columns:
        next_open = data.groupby('ticker')['open'].shift(-1)
        future_open = data.groupby('ticker')['open'].shift(-6)
        data['raw_ret_5d'] = (future_open / next_open) - 1
    else:
        # Fallback: Close-to-Close implies execution at Close(T), susceptible to bid-ask bounce noise.
        data['raw_ret_5d'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    
    # Target Engineering: Residualize returns relative to sector peers.
    # $y_{target} = r_{ticker} - \mu_{sector}$
    sector_mean = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('mean')
    data['target'] = data['raw_ret_5d'] - sector_mean
    
    data = data.dropna(subset=['target'])
    
    # Feature Selection Mask
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0', 'raw_ret_5d']
    
    # Identify Numeric vs Categorical features for preprocessing
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude]
    
    # Feature Universe Definition
    features = [c for c in data.columns if c not in exclude]
    
    # Robustness: Apply Winsorization ($1^{st}/99^{th}$ percentile clipping) to stabilize gradient descent.
    logger.info("🧹 Applying Winsorization (Shared Logic)...")
    data = WinsorisationScaler(clip_pct=0.01).fit(data, numeric_features).transform(data, numeric_features)

    # Normalization: Cross-sectional Z-Scoring ($z = \frac{x - \mu}{\sigma}$) within sectors.
    logger.info("⚖️ Applying Sector-Neutral Normalization (Shared Logic)...")
    data = SectorNeutralScaler(sector_col="sector").fit(data, numeric_features).transform(data, numeric_features)

    # Strategy Pattern: Map identifiers to concrete model implementations
    models_to_run = {
        "LightGBM": LightGBMModel,
        "XGBoost": XGBoostModel,
        "CatBoost": CatBoostModel
    }

    global_best_params = {}

    for name, m_class in models_to_run.items():
        opt = HyperparameterOptimizer(model_class=m_class, model_name=name)
        # Execution: Runs $N=20$ trials. In production, $N \ge 50$ is recommended for convergence.
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