"""
Run Optimization
================
Hyperparameter optimization for ML Alpha Model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import warnings
import json
from itertools import product
from typing import Dict, List, Tuple, Any

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import settings, print_welcome
from quant_alpha.data import DataLoader
from quant_alpha.features.registry import FactorRegistry
from quant_alpha.models import WalkForwardTrainer, LightGBMModel

warnings.filterwarnings('ignore')

# Try to import optimization libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Using grid search instead.")


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for ML models.
    
    Features:
        - Grid search optimization
        - Optuna-based optimization (if available)
        - Walk-forward validation
        - Feature selection optimization
        - Model parameter tuning
    """
    
    def __init__(self, features_df: pd.DataFrame, feature_names: List[str]):
        """
        Initialize optimizer.
        
        Args:
            features_df: DataFrame with features and targets
            feature_names: List of feature column names
        """
        self.features_df = features_df
        self.feature_names = feature_names
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
        print(f"🎯 Hyperparameter Optimizer initialized")
        print(f"   📊 Dataset: {features_df.shape}")
        print(f"   🔧 Features: {len(feature_names)}")
    
    def define_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space."""
        search_space = {
            # Model parameters
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.3, 0.5],
            'reg_lambda': [0.5, 1.0, 2.0, 3.0],
            'min_child_samples': [10, 20, 30, 50],
            
            # Feature selection
            'max_features': [15, 20, 25, 30, None],
            
            # Validation parameters
            'train_months': [12, 15, 18, 21],
            'test_months': [2, 3, 4],
        }
        
        return search_space
    
    def grid_search_optimization(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run grid search optimization.
        
        Args:
            n_trials: Maximum number of trials
            
        Returns:
            Best parameters found
        """
        print("\n" + "="*70)
        print("🔍 GRID SEARCH OPTIMIZATION")
        print("="*70)
        
        search_space = self.define_search_space()
        
        # Generate parameter combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        
        # Sample random combinations to limit trials
        np.random.seed(42)
        all_combinations = list(product(*param_values))
        
        if len(all_combinations) > n_trials:
            selected_indices = np.random.choice(len(all_combinations), n_trials, replace=False)
            combinations = [all_combinations[i] for i in selected_indices]
        else:
            combinations = all_combinations
        
        print(f"🎯 Testing {len(combinations)} parameter combinations...")
        
        for i, param_values in enumerate(combinations):
            params = dict(zip(param_names, param_values))
            
            try:
                score = self._evaluate_parameters(params)
                
                result = {
                    'trial': i,
                    'score': score,
                    'params': params.copy()
                }
                self.results.append(result)
                
                # Update best parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"   ✓ Completed {i + 1}/{len(combinations)} trials (Best IC: {self.best_score:.4f})")
                
            except Exception as e:
                print(f"   ⚠️ Trial {i} failed: {e}")
                continue
        
        self._print_optimization_summary()
        return self.best_params
    
    def optuna_optimization(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run Optuna-based optimization.
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available. Using grid search instead.")
            return self.grid_search_optimization(n_trials)
        
        print("\n" + "="*70)
        print("🔍 OPTUNA OPTIMIZATION")
        print("="*70)
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'max_features': trial.suggest_categorical('max_features', [15, 20, 25, 30, None]),
                'train_months': trial.suggest_int('train_months', 12, 24),
                'test_months': trial.suggest_int('test_months', 2, 6),
            }
            
            return self._evaluate_parameters(params)
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n🏆 Optuna optimization completed!")
        print(f"   📈 Best IC: {self.best_score:.4f}")
        print(f"   🎯 Best parameters: {len(self.best_params)} params")
        
        return self.best_params
    
    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a set of parameters using walk-forward validation.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Average Information Coefficient
        """
        # Extract validation parameters
        train_months = params.pop('train_months', 18)
        test_months = params.pop('test_months', 3)
        max_features = params.pop('max_features', None)
        
        # Feature selection
        feature_names = self.feature_names.copy()
        if max_features and max_features < len(feature_names):
            # Simple feature selection - use first N features
            # In practice, you might want more sophisticated selection
            feature_names = feature_names[:max_features]
        
        # Update validation config temporarily
        original_train_months = settings.validation.train_months
        original_test_months = settings.validation.test_months
        
        settings.validation.train_months = train_months
        settings.validation.test_months = test_months
        
        try:
            # Create trainer with custom parameters
            trainer = WalkForwardTrainer(feature_names)
            
            # Override model parameters
            original_params = settings.model.lgb_params.copy()
            settings.model.lgb_params.update(params)
            
            # Run validation (limited folds for speed)
            results_df = trainer.train_and_validate(self.features_df, save_models=False)
            
            # Calculate average IC
            if len(results_df) > 0 and 'test_ic' in results_df.columns:
                avg_ic = results_df['test_ic'].mean()
                return avg_ic if not np.isnan(avg_ic) else -1.0
            else:
                return -1.0
                
        except Exception as e:
            print(f"   ⚠️ Evaluation failed: {e}")
            return -1.0
            
        finally:
            # Restore original settings
            settings.validation.train_months = original_train_months
            settings.validation.test_months = original_test_months
            settings.model.lgb_params = original_params
    
    def _print_optimization_summary(self):
        """Print optimization results summary."""
        if not self.results:
            print("❌ No optimization results!")
            return
        
        print(f"\n📊 OPTIMIZATION SUMMARY")
        print("="*50)
        
        results_df = pd.DataFrame(self.results)
        
        print(f"🎯 Total trials: {len(results_df)}")
        print(f"🏆 Best IC: {self.best_score:.4f}")
        print(f"📈 Mean IC: {results_df['score'].mean():.4f}")
        print(f"📊 Std IC: {results_df['score'].std():.4f}")
        
        # Top 5 results
        top_results = results_df.nlargest(5, 'score')
        print(f"\n🏆 Top 5 Results:")
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"   {i}. IC: {row['score']:.4f} (Trial {row['trial']})")
        
        # Best parameters
        print(f"\n🎯 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param:20s}: {value}")
        
        print("="*50)
    
    def save_results(self, path: str):
        """Save optimization results."""
        results_data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Optimization results saved: {path}")
    
    def apply_best_parameters(self):
        """Apply best parameters to global settings."""
        if not self.best_params:
            print("❌ No best parameters found!")
            return
        
        # Update model parameters
        model_params = {k: v for k, v in self.best_params.items() 
                       if k not in ['train_months', 'test_months', 'max_features']}
        
        settings.model.lgb_params.update(model_params)
        
        # Update validation parameters
        if 'train_months' in self.best_params:
            settings.validation.train_months = self.best_params['train_months']
        if 'test_months' in self.best_params:
            settings.validation.test_months = self.best_params['test_months']
        if 'max_features' in self.best_params and self.best_params['max_features']:
            settings.features.max_features = self.best_params['max_features']
        
        print("✅ Best parameters applied to global settings!")


def main():
    """Run hyperparameter optimization pipeline."""
    
    start_time = time.time()
    
    print_welcome()
    print("\n🎯 HYPERPARAMETER OPTIMIZATION PIPELINE")
    print("="*70)
    
    try:
        # Check if features data exists
        features_path = settings.data.processed_dir / "features_dataset.pkl"
        
        if not features_path.exists():
            print("❌ Missing features dataset! Run research pipeline first:")
            print("   python scripts/run_research.py")
            return False
        
        # Load features data
        print("📊 Loading features dataset...")
        features_df = pd.read_pickle(features_path)
        
        # Load feature names
        importance_path = settings.results_dir / "feature_importance.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            feature_names = importance_df['feature'].tolist()
        else:
            # Fallback: detect feature names
            meta_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'forward_return']
            feature_names = [col for col in features_df.columns if col not in meta_cols]
        
        print(f"✅ Loaded features dataset: {features_df.shape}")
        print(f"✅ Feature names: {len(feature_names)} features")
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(features_df, feature_names)
        
        # Run optimization
        if OPTUNA_AVAILABLE:
            print("🔍 Using Optuna optimization...")
            best_params = optimizer.optuna_optimization(n_trials=50)
        else:
            print("🔍 Using grid search optimization...")
            best_params = optimizer.grid_search_optimization(n_trials=30)
        
        # Save results
        results_path = settings.results_dir / "optimization_results.json"
        optimizer.save_results(str(results_path))
        
        # Apply best parameters
        optimizer.apply_best_parameters()
        
        # Save updated config
        config_path = settings.results_dir / "optimized_config.json"
        optimized_config = {
            'model_params': settings.model.lgb_params,
            'validation_params': {
                'train_months': settings.validation.train_months,
                'test_months': settings.validation.test_months,
                'embargo_days': settings.validation.embargo_days
            },
            'feature_params': {
                'max_features': settings.features.max_features,
                'forward_return_days': settings.features.forward_return_days
            },
            'optimization_summary': {
                'best_score': optimizer.best_score,
                'total_trials': len(optimizer.results),
                'optimization_time': time.time() - start_time
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(optimized_config, f, indent=2, default=str)
        
        print(f"\n✅ Optimization completed successfully!")
        print(f"📊 Results saved:")
        print(f"   • {results_path.name}")
        print(f"   • {config_path.name}")
        
        # Print improvement summary
        if optimizer.best_score > 0:
            baseline_ic = 0.0149  # Your current model IC
            improvement = ((optimizer.best_score - baseline_ic) / baseline_ic) * 100
            print(f"\n📈 Performance Improvement:")
            print(f"   Baseline IC: {baseline_ic:.4f}")
            print(f"   Optimized IC: {optimizer.best_score:.4f}")
            print(f"   Improvement: {improvement:+.1f}%")
        
        # Recommendations
        print(f"\n💡 Next Steps:")
        print(f"   1. Review optimized parameters in {config_path.name}")
        print(f"   2. Re-run research pipeline with optimized settings:")
        print(f"      python scripts/run_research.py")
        print(f"   3. Compare results with baseline performance")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total optimization time: {total_time/60:.1f} minutes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_optimization():
    """Run quick optimization with fewer trials."""
    print("🚀 Running quick optimization (10 trials)...")
    
    try:
        # Load data
        features_path = settings.data.processed_dir / "features_dataset.pkl"
        features_df = pd.read_pickle(features_path)
        
        importance_path = settings.results_dir / "feature_importance.csv"
        if importance_path.exists():
            importance_df = pd.read_csv(importance_path)
            feature_names = importance_df['feature'].tolist()[:15]  # Use top 15 features
        else:
            meta_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'forward_return']
            feature_names = [col for col in features_df.columns if col not in meta_cols][:15]
        
        # Quick optimization
        optimizer = HyperparameterOptimizer(features_df, feature_names)
        best_params = optimizer.grid_search_optimization(n_trials=10)
        
        print(f"✅ Quick optimization complete!")
        print(f"🏆 Best IC: {optimizer.best_score:.4f}")
        
        return best_params
        
    except Exception as e:
        print(f"❌ Quick optimization failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--quick', action='store_true', help='Run quick optimization (10 trials)')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--method', choices=['grid', 'optuna'], default='auto', help='Optimization method')
    parser.add_argument('--features', type=int, help='Maximum number of features to use')
    
    args = parser.parse_args()
    
    # Override settings if provided
    if args.features:
        settings.features.max_features = args.features
    
    if args.quick:
        quick_optimization()
    else:
        # Override optimization method if specified
        if args.method == 'grid':
            # Force grid search by setting OPTUNA_AVAILABLE to False
            import sys
            current_module = sys.modules[__name__]
            current_module.OPTUNA_AVAILABLE = False
        
        success = main()
        if success:
            print("\n🎯 Hyperparameter optimization completed successfully!")
        else:
            print("\n💥 Hyperparameter optimization failed!")
            sys.exit(1)