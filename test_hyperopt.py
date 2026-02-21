import optuna
import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr
from typing import Dict, Any, List
import inspect

# Core Quant Alpha Imports
from quant_alpha.data.DataManager import DataManager
from quant_alpha.models.lightgbm_model import LightGBMModel
from quant_alpha.models.trainer import WalkForwardTrainer
# Tip: Import your other model wrappers here
from quant_alpha.models.xgboost_model import XGBoostModel
from quant_alpha.models.catboost_model import CatBoostModel
from quant_alpha.features.utils import winsorize

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Institutional Grade Multi-Model Bayesian Optimizer.
    Optimizes Rank IC across recent Walk-Forward folds.
    """
    
    def __init__(self, model_class: Any, model_name: str, direction: str = 'maximize'):
        self.model_class = model_class
        self.model_name = model_name.lower()
        self.direction = direction
        self.study = None
        self.best_params = None
    
    def _instantiate_model(self, params: Dict):
        """
        Smart Instantiation:
        1. Identifies arguments needed for the Constructor (__init__).
        2. Passes the rest into the model's internal parameter dictionary.
        """
        # Get the constructor signature
        signature = inspect.signature(self.model_class.__init__)
        constructor_keys = signature.parameters.keys()
        
        # Split params into 'constructor args' and 'model hyperparams'
        constructor_args = {k: v for k, v in params.items() if k in constructor_keys}
        hyperparams = {k: v for k, v in params.items() if k not in constructor_keys}

        # Case 1: Model wrapper expects a 'params' dictionary directly
        if 'params' in constructor_keys:
            return self.model_class(params=params)
            
        # Case 2: Standard instantiation + Attribute Injection
        # We initialize with whatever the constructor accepts
        model_instance = self.model_class(**constructor_args)
        
        # Now, we "Inject" the leftover hyperparams into the instance's params dict
        # Most of our wrappers (and standard LGBM/XGB) store config in .params
        if hasattr(model_instance, 'params'):
            model_instance.params.update(hyperparams)
        else:
            # Fallback: Set them as direct attributes if no .params dict exists
            for k, v in hyperparams.items():
                set_attr_name = k
                setattr(model_instance, set_attr_name, v)
        
        return model_instance

    def optimize(self, data: pd.DataFrame, features: List[str], target: str, n_trials: int = 20) -> Dict:
        logger.info(f"🧬 Starting Walk-Forward Hyperopt for {self.model_name.upper()}...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = self._get_search_space(trial)
            
            # Use WalkForwardTrainer for consistent split logic
            trainer = WalkForwardTrainer(
                model_class=self.model_class,
                min_train_months=36,
                test_months=6,
                step_months=6,
                window_type='sliding'
            )
            
            splits = trainer._generate_splits(data)
            fold_ics = []
            optimization_splits = splits[-5:] # Recent market regimes focus

            for step, (train_start, train_end, test_start, test_end) in enumerate(optimization_splits):
                train_mask = (data['date'] >= train_start) & (data['date'] <= train_end)
                test_mask = (data['date'] >= test_start) & (data['date'] <= test_end)
                
                train_fold = data[train_mask]
                test_fold = data[test_mask]
                
                if train_fold.empty or test_fold.empty: continue

                # ✅ Interface matches your LightGBMModel(params=params)
                model = self._instantiate_model(params)
                
                model.fit(train_fold[features], train_fold[target])
                preds = model.predict(test_fold[features])
                
                if np.std(preds) < 1e-9:
                    fold_ic = -1.0
                else:
                    fold_ic, _ = spearmanr(preds, test_fold[target])
                
                fold_ics.append(fold_ic)
                
                # Pruning logic
                current_mean = np.nanmean(fold_ics)
                trial.report(np.nan_to_num(current_mean, nan=-1.0), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            final_ic = np.nanmean(fold_ics) if fold_ics else -1.0
            return np.nan_to_num(final_ic, nan=-1.0)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        self.study = optuna.create_study(direction=self.direction, pruner=pruner)
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        return self.best_params

    def _get_search_space(self, trial) -> Dict:
        """Standardized search spaces for the Big Three GBDT models."""
        
        # 1. LIGHTGBM
        if 'lgbm' in self.model_name or 'lightgbm' in self.model_name:
            return {
                'num_leaves': trial.suggest_int('num_leaves', 20, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0, log=True),
                'n_estimators': 300,
                'n_jobs': 1
            }
        
        # 2. XGBOOST
        elif 'xgb' in self.model_name or 'xgboost' in self.model_name:
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 10.0, 100.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'n_estimators': 300,
                'n_jobs': 1
            }
            
        # 3. CATBOOST
        elif 'cat' in self.model_name or 'catboost' in self.model_name:
            return {
                'depth': trial.suggest_int('depth', 4, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10.0, 100.0, log=True),
                'bootstrap_type': 'Bernoulli',
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'iterations': 300,
                'thread_count': 1,
                'verbose': 0
            }
            
        return {}

# =============================================
# RUNNER: Optimize Everything
# =============================================
def verify_all_models():
    logger.info("🚀 Starting Full Ensemble Optimization...")
    dm = DataManager()
    data = dm.get_master_data()

    # Ensure 'date' is a column and in datetime format
    if 'date' not in data.columns:
        data = data.reset_index()
    
    if 'date' not in data.columns:
        # Agar abhi bhi nahi mila, matlab column ka naam 'Date' (capital) ho sakta hai
        data.rename(columns={'Date': 'date'}, inplace=True)
        
    data['date'] = pd.to_datetime(data['date'])
    
    # Target & Feature Prep
    data = data.sort_values(['ticker', 'date'])
    data['target'] = data.groupby('ticker')['close'].shift(-5) / data['close'] - 1
    data = data.dropna(subset=['target'])
    
    exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'date', 'ticker', 'index', 'level_0']
    features = [c for c in data.columns if c not in exclude]
    
    # Apply Winsorization (Consistency with Production)
    logger.info("🧹 Applying Winsorization before Hyperopt...")
    data = winsorize(data, features)

    # Models to optimize
    # Note: Replace placeholders with your actual imported classes
    models_to_run = {
        "LightGBM": LightGBMModel,
        "XGBoost": XGBoostModel,   # Uncomment when ready
        "CatBoost": CatBoostModel  # Uncomment when ready
    }

    global_best_params = {}

    for name, m_class in models_to_run.items():
        opt = HyperparameterOptimizer(model_class=m_class, model_name=name)
        # Testing with 10 trials per model
        best = opt.optimize(data, features, 'target', n_trials=10)
        global_best_params[name] = best

    print("\n" + "═"*45)
    print("🏆 FINAL MULTI-MODEL SUMMARY")
    print("═"*45)
    for model, params in global_best_params.items():
        print(f"✅ {model}: {params}")
    print("═"*45)

if __name__ == "__main__":
    verify_all_models()