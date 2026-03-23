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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.utils import setup_logging

from quant_alpha.models.hyperopt import HyperparameterOptimizer

setup_logging()
logger = logging.getLogger("Hyperopt")

def run_hyperopt():
    """
    Executes the Bayesian hyperparameter optimization sequence across the modeling ensemble.

    Acquires the full historical dataset, constructs volatility-dampened forward 
    return targets, and delegates Tree-structured Parzen Estimator (TPE) tuning 
    to the underlying model wrappers. Output parameters are strictly evaluated 
    via purged walk-forward cross-validation to guarantee out-of-sample robustness.

    Args:
        None

    Returns:
        None
    """
    logger.info("🚀 Starting Full Ensemble Optimization...")
    dm = DataManager()
    data = dm.get_master_data()

    # Enforce strict temporal indexing to guarantee chronological walk-forward splitting
    if 'date' not in data.columns:
        data = data.reset_index()
    
    if 'date' not in data.columns:
        # Resolves potential upstream data provider capitalization inconsistencies
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'date'}, inplace=True)
        
    data['date'] = pd.to_datetime(data['date'])
    
    data = data.sort_values(['ticker', 'date'])
    
    # Target Construction: Extracts $R_{t+1 \to t+6}$ (5-day forward return)
    # Binds execution structurally to Open-to-Open prices to reflect realistic trade entry
    if 'open' in data.columns:
        next_open = data.groupby('ticker')['open'].shift(-1)
        future_open = data.groupby('ticker')['open'].shift(-6)
        data['raw_ret_5d'] = (future_open / next_open) - 1
    else:
        # Fallback formulation: Close-to-Close proxy, introducing bid-ask bounce noise
        data['raw_ret_5d'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    
    # Volatility Dampening: Adjusts the objective landscape to prioritize persistent,
    # structural alpha over transient variance spikes. Exact match to the production
    # target generation logic to guarantee hyperparameter convergence.
    sector_mean = data.groupby(['date', 'sector'])['raw_ret_5d'].transform('mean')
    resid = data['raw_ret_5d'] - sector_mean
    roll_std = (
        data.groupby('ticker')['raw_ret_5d']
        .transform(lambda x: x.rolling(63, min_periods=21).std())
    )
    vol_damp = 1.0 / (1.0 + roll_std.fillna(roll_std.median()))
    data['target'] = resid * vol_damp

    data = data.dropna(subset=['target'])

    # Establish isolation masks for strict exclusion of non-predictive metadata 
    # and look-ahead temporal markers
    exclude = [
        'open', 'high', 'low', 'close', 'volume', 'target', 'date',
        'ticker', 'index', 'level_0', 'raw_ret_5d',
        'macro_mom_5d', 'macro_mom_21d', 'macro_vix_proxy', 'macro_trend_200d',
        'us_10y_close', 'vix_close', 'oil_close', 'usd_close', 'sp500_close'
    ]

    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in exclude]

    # Bind the final actionable feature matrix for optimization
    features = [c for c in data.columns if c not in exclude]

    # Data Leakage Prevention: Defers scalar operations strictly to per-fold initialization.
    # Executing global standardization prior to the TPE search would inject future 
    # statistical moments, invalidating the optimization integrity.

    # Strategy Pattern: Map identifiers to concrete model implementations
    models_to_run = {
        "LightGBM": LightGBMModel,
        "XGBoost": XGBoostModel,
        "CatBoost": CatBoostModel
    }

    global_best_params = {}

    for name, m_class in models_to_run.items():
        opt = HyperparameterOptimizer(model_class=m_class, model_name=name)
        # Hyperparameter Search: Executes $N=20$ Bayesian trials via TPE.
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
