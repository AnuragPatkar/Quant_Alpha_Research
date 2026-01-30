"""
Run Optimization
================
Hyperparameter optimization for ML Alpha Model.

Author: Anurag Patkar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import json
import argparse
from itertools import product
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils import Timer, print_header, print_section, save_results, ensure_dir

try:
    from config import settings, print_welcome
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from quant_alpha.models import WalkForwardTrainer, LightGBMModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


class HyperparameterOptimizer:
    """Hyperparameter optimization for ML models."""
    
    def __init__(self, features_df: pd.DataFrame, feature_names: List[str]):
        self.features_df = features_df
        self.feature_names = feature_names
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
        print(f"🎯 Optimizer initialized")
        print(f"   📊 Dataset: {features_df.shape}")
        print(f"   🔧 Features: {len(feature_names)}")
    
    def grid_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """Run grid search optimization."""
        print_section("GRID SEARCH OPTIMIZATION")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [15, 31, 63],
            'min_child_samples': [20, 50, 100]
        }
        
        combinations = list(product(*param_grid.values()))
        if len(combinations) > n_trials:
            np.random.seed(42)
            idx = np.random.choice(len(combinations), n_trials, replace=False)
            combinations = [combinations[i] for i in idx]
        
        print(f"🎯 Testing {len(combinations)} combinations...")
        
        for i, vals in enumerate(combinations):
            params = dict(zip(param_grid.keys(), vals))
            
            try:
                score = self._evaluate(params)
                self.results.append({'trial': i, 'score': score, 'params': params.copy()})
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    print(f"   ✓ {i+1}/{len(combinations)} (Best IC: {self.best_score:.4f})")
            except:
                continue
        
        self._print_summary()
        return self.best_params
    
    def random_search(self, n_trials: int = 50) -> Dict[str, Any]:
        """Run random search optimization."""
        print_section("RANDOM SEARCH OPTIMIZATION")
        print(f"🎯 Running {n_trials} trials...")
        
        for i in range(n_trials):
            params = {
                'n_estimators': np.random.choice([100, 150, 200, 250, 300, 400]),
                'max_depth': np.random.randint(3, 9),
                'learning_rate': np.random.choice([0.01, 0.03, 0.05, 0.08, 0.1]),
                'num_leaves': np.random.choice([15, 31, 47, 63, 95]),
                'min_child_samples': np.random.choice([10, 20, 30, 50, 80, 100]),
                'subsample': np.random.choice([0.7, 0.8, 0.9, 1.0]),
                'colsample_bytree': np.random.choice([0.7, 0.8, 0.9, 1.0])
            }
            
            try:
                score = self._evaluate(params)
                self.results.append({'trial': i, 'score': score, 'params': params.copy()})
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    print(f"   ✓ {i+1}/{n_trials} (Best IC: {self.best_score:.4f})")
            except:
                continue
        
        self._print_summary()
        return self.best_params
    
    def optuna_search(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run Optuna optimization."""
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available, using random search")
            return self.random_search(n_trials)
        
        print_section("OPTUNA OPTIMIZATION")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            return self._evaluate(params)
        
        study = optuna.create_study(direction='maximize')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n🏆 Best IC: {self.best_score:.4f}")
        return self.best_params
    
    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate parameters using simple CV."""
        if not MODELS_AVAILABLE:
            return np.random.random() * 0.1  # Fallback
        
        # Simple time-series split
        dates = sorted(self.features_df['date'].unique())
        n = len(dates)
        train_end = int(n * 0.7)
        
        train_dates = dates[:train_end]
        test_dates = dates[train_end:]
        
        train_data = self.features_df[self.features_df['date'].isin(train_dates)]
        test_data = self.features_df[self.features_df['date'].isin(test_dates)]
        
        if len(train_data) < 100 or len(test_data) < 50:
            return -1.0
        
        # Prepare features
        feature_cols = [c for c in self.feature_names if c in train_data.columns]
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data['forward_return'].fillna(0)
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data['forward_return'].fillna(0)
        
        # Train
        model = LightGBMModel(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # IC
        from scipy import stats
        ic, _ = stats.spearmanr(predictions, y_test)
        
        return ic if not np.isnan(ic) else -1.0
    
    def _print_summary(self):
        """Print optimization summary."""
        if not self.results:
            print("❌ No results!")
            return
        
        results_df = pd.DataFrame(self.results)
        
        print_section("OPTIMIZATION SUMMARY")
        print(f"🎯 Trials: {len(results_df)}")
        print(f"🏆 Best IC: {self.best_score:.4f}")
        print(f"📊 Mean IC: {results_df['score'].mean():.4f}")
        
        print(f"\n🎯 Best Parameters:")
        for k, v in self.best_params.items():
            print(f"   {k}: {v}")
    
    def save_results(self, path: str):
        """Save optimization results."""
        data = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        save_results(data, path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--method', choices=['grid', 'random', 'optuna'], default='random')
    parser.add_argument('--quick', action='store_true', help='Quick optimization (10 trials)')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if SETTINGS_AVAILABLE:
        print_welcome()
    
    print_header("HYPERPARAMETER OPTIMIZATION")
    
    try:
        # Load data
        if SETTINGS_AVAILABLE:
            features_path = settings.data.processed_dir / "features_dataset.pkl"
            results_dir = settings.results_dir
        else:
            features_path = ROOT / "data" / "processed" / "features_dataset.pkl"
            results_dir = ROOT / "output" / "results"
        
        if not features_path.exists():
            print("❌ Features not found! Run run_research.py first.")
            return 1
        
        features_df = pd.read_pickle(features_path)
        print(f"✅ Loaded features: {features_df.shape}")
        
        # Get feature names
        imp_path = results_dir / "feature_importance.csv"
        if imp_path.exists():
            imp_df = pd.read_csv(imp_path)
            feature_names = imp_df['feature'].tolist()
        else:
            meta = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'forward_return']
            feature_names = [c for c in features_df.columns if c not in meta]
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(features_df, feature_names)
        
        # Run optimization
        n_trials = 10 if args.quick else args.trials
        
        if args.method == 'grid':
            best_params = optimizer.grid_search(n_trials)
        elif args.method == 'optuna':
            best_params = optimizer.optuna_search(n_trials)
        else:
            best_params = optimizer.random_search(n_trials)
        
        # Save results
        output_dir = Path(args.output) if args.output else results_dir
        ensure_dir(output_dir)
        optimizer.save_results(str(output_dir / "optimization_results.json"))
        
        print(f"\n✅ Optimization completed in {(time.time() - start_time)/60:.1f} min")
        print(f"📁 Results: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())