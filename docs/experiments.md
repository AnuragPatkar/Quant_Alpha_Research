# Experiments & Workflow Guide

> **Purpose**: Step-by-step guide for running research experiments, comparing models, logging metrics, and validating factors before production deployment.

---

## 1. Factor Research Workflow

### 1.1 Engineer & Validate a New Factor (2-3 Hours)

**Goal**: Develop, test, and validate a new alpha factor.

#### Step 1: Implement Factor Class

Create a new Python file in `quant_alpha/features/custom/`:

```python
# quant_alpha/features/custom/my_momentum_factor.py
from quant_alpha.features import BaseFactor, FactorRegistry
import pandas as pd
import numpy as np

@FactorRegistry.register()
class CustomMomentumFactor(BaseFactor):
    """
    Multi-horizon momentum factor combining 1M, 3M, 6M signals.
    
    Theory:
        Captures intermediate-term price trends. Combines multiple horizons
        to reduce noise and improve signal robustness.
    
    Parameters:
        d1 (int): Short-term lookback (days). Default: 21
        d2 (int): Medium-term lookback (days). Default: 63
        d3 (int): Long-term lookback (days). Default: 252
    """
    
    def __init__(self, d1=21, d2=63, d3=252, name='custom_momentum'):
        super().__init__(
            name=name,
            category='technical',
            lookback_period=max(d1, d2, d3),
            normalize=True,
            winsorize=True
        )
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute multi-horizon momentum signal.
        
        Args:
            data (pd.DataFrame): MultiIndex(date, ticker) DataFrame with 'close'
        
        Returns:
            pd.Series: Momentum scores, same index as input
        """
        # Validate required columns
        if 'close' not in data.columns:
            raise ValueError("'close' column required")
        
        # Group by ticker, compute returns at multiple horizons
        returns = data.groupby('ticker')['close'].pct_change()
        
        # 1-month momentum
        m1 = returns.rolling(window=self.d1).sum()
        
        # 3-month momentum
        m2 = returns.rolling(window=self.d2).sum()
        
        # 6-month momentum
        m3 = returns.rolling(window=self.d3).sum()
        
        # Composite: weighted average (equal weights)
        composite = (m1 + m2 + m3) / 3
        
        return composite
```

#### Step 2: Load & Compute Factor

```python
# In notebook: 02_factor_research.ipynb Cell 1

from quant_alpha.data import DataManager
from quant_alpha.features import FactorRegistry

# Load data
dm = DataManager()
data = dm.get_data(start_date='2023-01-01', end_date='2024-12-31')

# Compute all factors (includes new CustomMomentumFactor)
registry = FactorRegistry()
factors = registry.compute_all(data, max_workers=4)

print(f"Total factors: {len(factors.columns)}")
print(f"New factor column: 'custom_momentum' in {list(factors.columns)}")
```

#### Step 3: Analyze Factor Efficacy

```python
# Cell 2: Compute Information Coefficient

from quant_alpha.research import FactorAnalyzer
import numpy as np

# Prepare targets (5-day forward returns)
targets = data.groupby('ticker')['close'].pct_change(5).shift(-5)

# Analyze custom factor
analyzer = FactorAnalyzer(data=data)
ic_timeseries = analyzer.compute_ic(
    factor=factors['custom_momentum'],
    returns=targets,
    correlation_type='spearman'
)

# Print summary
ic_mean = ic_timeseries.mean()
ic_std = ic_timeseries.std()
ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_timeseries))) * np.sqrt(252)

print(f"Mean IC: {ic_mean:.6f}")
print(f"Std IC: {ic_std:.6f}")
print(f"IC t-statistic: {ic_tstat:.2f}")
print(f"Pass Alpha Gate (t > 2.5)? {'✓ YES' if ic_tstat > 2.5 else '✗ NO'}")
```

**Expected output**:
```
Mean IC: 0.004250
Std IC: 0.008340
IC t-statistic: 2.87
Pass Alpha Gate (t > 2.5)? ✓ YES
```

#### Step 4: Analyze Decay Profile

```python
# Cell 3: Factor Alpha Decay

decay = analyzer.compute_alpha_decay(
    factor=factors['custom_momentum'],
    returns=targets,
    max_days=20
)

# Plot decay curve
decay.plot(figsize=(12, 6), title='Custom Momentum Alpha Decay')
plt.xlabel('Days Forward')
plt.ylabel('Forward Return Prediction')
plt.grid(True, alpha=0.3)
plt.show()

# Print half-life
half_value = decay.iloc[0] / 2
half_life = (decay - half_value).abs().idxmin()
print(f"Factor half-life: {half_life} days")
```

#### Step 5: Quantile Analysis

```python
# Cell 4: Long-Short Spread

quantile_returns = analyzer.compute_quantile_returns(
    factor=factors['custom_momentum'],
    returns=targets,
    n_quantiles=10
)

# Plot quantile returns
quantile_returns.plot(kind='bar', figsize=(12, 6), title='Quantile Returns')
plt.ylabel('Average Daily Return')
plt.xlabel('Decile (1=Lowest, 10=Highest)')
plt.grid(True, alpha=0.3)

# Long-short spread
ls_spread = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
print(f"Long-Short Daily Spread: {ls_spread:.4%}")
print(f"Long-Short Annual Spread: {ls_spread * 252:.2%}")
```

**Interpretation**:
- Long-short spread > 0.02% → Strong signal ✓
- Long-short spread 0.01-0.02% → Moderate signal
- Long-short spread < 0.01% → Weak signal, consider refinement ✗

#### Step 6: Decision Point

**If factor passes gates**:
- Mean IC t-stat > 2.5 ✓
- Long-short spread > 0.01% ✓
- Half-life > 1 day ✓

→ **Proceed to model training** (Step 2 below)

**If factor fails gates**:
- Return to Step 1 and refine factor logic
- Try different lookback periods
- Combine with other signals
- Consider alternative universe (not all S&P 500)

---

## 2. Model Development & Comparison

### 2.1 Train Multiple Models (3-4 Hours)

**Objective**: Compare GBDT base models and select the best ensemble.

#### Step 1: Initialize Trainer

```python
# In notebook: 03_model_development.ipynb Cell 1

from quant_alpha.models import WalkForwardTrainer, XGBoostModel, LightGBMModel, CatBoostModel
import pandas as pd

# Load features and targets
features = factors.dropna(axis=1, how='all')  # Drop all-NaN columns
targets = data.groupby('ticker')['close'].pct_change(5).shift(-5)

# Ensure alignment
common_index = features.index.intersection(targets.index)
features = features.loc[common_index]
targets = targets.loc[common_index]

print(f"Training set: {targets.shape[0]} samples × {features.shape[1]} features")
print(f"Missing targets: {targets.isna().sum()}")
```

#### Step 2: Train Individual Models

```python
# Cell 2: Train with different GBDT implementations (long running)

models_to_train = [
    ('lightgbm', LightGBMModel),
    ('xgboost', XGBoostModel),
    ('catboost', CatBoostModel),
]

results = {}
for model_name, model_class in models_to_train:
    print(f"\n🚀 Training {model_name.upper()}...")
    
    trainer = WalkForwardTrainer(
        model_class=model_class,
        min_train_months=36,
        test_months=6,
        step_months=3,
        embargo_days=21,
        n_jobs=4
    )
    
    oos_pred, models, metrics = trainer.train(X=features, y=targets)
    
    results[model_name] = {
        'predictions': oos_pred,
        'models': models,
        'metrics': metrics
    }
    
    print(f"✓ {model_name.upper()} complete")
    print(f"   IC: {metrics['information_coefficient']:.6f}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   t-statistic: {metrics['t_statistic']:.2f}")
```

**Expected output**:
```
🚀 Training LIGHTGBM...
✓ LIGHTGBM complete
   IC: 0.021450
   Sharpe Ratio: 1.95
   t-statistic: 2.67

🚀 Training XGBOOST...
✓ XGBOOST complete
   IC: 0.019320
   Sharpe Ratio: 1.78
   t-statistic: 2.41

🚀 Training CATBOOST...
✓ CATBOOST complete
   IC: 0.022100
   Sharpe Ratio: 2.11
   t-statistic: 2.78
```

#### Step 3: Compare Results

```python
# Cell 3: Compare model performance

comparison_df = pd.DataFrame({
    'LightGBM': results['lightgbm']['metrics'],
    'XGBoost': results['xgboost']['metrics'],
    'CatBoost': results['catboost']['metrics'],
}).T

print("Model Comparison (Out-of-Sample):")
print(comparison_df[['information_coefficient', 'sharpe_ratio', 't_statistic']])

# Best model by IC
best_model = comparison_df['information_coefficient'].idxmax()
print(f"\n🏆 Best model by IC: {best_model}")
```

#### Step 4: Create Ensemble

```python
# Cell 4: Blend predictions using rank averaging

from quant_alpha.models import EnsembleModel

# Create ensemble of all three models
ensemble = EnsembleModel(
    base_models=[
        results['lightgbm']['models'],
        results['xgboost']['models'],
        results['catboost']['models'],
    ],
    blending_method='rank_average',  # Rank-based averaging for robustness
    weights=[0.5, 0.3, 0.2]  # Optional: weight by performance
)

# Get ensemble predictions (average of ranks)
ensemble_predictions = ensemble.predict(X=features)

# Evaluate ensemble
analyzer = FactorAnalyzer(data=data)
ensemble_ic = analyzer.compute_ic(
    factor=ensemble_predictions,
    returns=targets,
    correlation_type='spearman'
)

print(f"Ensemble IC: {ensemble_ic.mean():.6f}")
print(f"Ensemble t-stat: {(ensemble_ic.mean() / ensemble_ic.std() * np.sqrt(252)):.2f}")
```

**Ensemble benefits**:
- ✓ Reduces single-model risk
- ✓ Captures different feature interactions
- ✓ More robust to market regime changes

---

## 3. Backtesting & Optimization Comparison

### 3.1 Compare Portfolio Methods (2-3 Hours)

**Goal**: Test multiple portfolio construction methods and select optimal allocation strategy.

```python
# In notebook: 04_backtest_analysis.ipynb

from quant_alpha.backtest import BacktestEngine
import matplotlib.pyplot as plt

# Initialize engine with standard parameters
engine = BacktestEngine(
    initial_capital=1_000_000,
    commission_bps=10,
    spread_bps=5,
    slippage_bps=2,
    rebalance_frequency='daily',
    target_volatility=0.15,
    max_adv_participation=0.02,
    trailing_stop_pct=0.10,
    max_drawdown_trigger=-0.20
)

# Test different optimization strategies
strategies = {
    'Top-N (Equal Weight)': {
        'optimization_method': None,
        'top_n': 50,
        'is_weights': False
    },
    'Mean-Variance': {
        'optimization_method': 'mean_variance',
        'top_n': None,
        'is_weights': True
    },
    'Kelly Criterion': {
        'optimization_method': 'kelly',
        'top_n': None,
        'is_weights': True
    },
    'Risk Parity (ERC)': {
        'optimization_method': 'risk_parity',
        'top_n': None,
        'is_weights': True
    },
}

results = {}
for strategy_name, params in strategies.items():
    print(f"\n🎯 Backtesting: {strategy_name}...")
    
    results[strategy_name] = engine.run(
        signals=ensemble_predictions,
        prices=prices[['open', 'high', 'low', 'close', 'volume']],
        **params
    )
    
    metrics = results[strategy_name]['metrics']
    print(f"   CAGR: {metrics['cagr']:.2%}")
    print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"   Max DD: {metrics['max_drawdown']:.2%}")
    print(f"   Win Rate: {metrics['win_rate']:.2%}")
```

### 3.2 Visualization & Comparison

```python
# Cell 2: Plot equity curves and metrics

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Equity curves
ax = axes[0, 0]
for name, result in results.items():
    ax.plot(result['equity_curve'].index, result['equity_curve'].values / 1_000_000, label=name)
ax.set_title('Equity Curves')
ax.set_ylabel('Portfolio Value ($M)')
ax.legend()
ax.grid(True, alpha=0.3)

# Metrics comparison
ax = axes[0, 1]
metrics_data = {name: result['metrics']['cagr'] for name, result in results.items()}
ax.bar(metrics_data.keys(), metrics_data.values())
ax.set_title('CAGR Comparison')
ax.set_ylabel('CAGR (%)')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Sharpe ratios
ax = axes[1, 0]
sharpe_data = {name: result['metrics']['sharpe_ratio'] for name, result in results.items()}
ax.bar(sharpe_data.keys(), sharpe_data.values(), color='green')
ax.set_title('Sharpe Ratio Comparison')
ax.set_ylabel('Sharpe Ratio')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Maximum drawdowns
ax = axes[1, 1]
dd_data = {name: result['metrics']['max_drawdown'] * 100 for name, result in results.items()}
ax.bar(dd_data.keys(), dd_data.values(), color='red')
ax.set_title('Maximum Drawdown Comparison')
ax.set_ylabel('Max DD (%)')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/backtest_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary table
summary = pd.DataFrame({
    name: result['metrics'] 
    for name, result in results.items()
}).T

print("\n📊 Summary Statistics:")
print(summary[['cagr', 'sharpe_ratio', 'max_drawdown', 'win_rate']])
```

---

## 4. Model Monitoring & Validation

### 4.1 Out-of-Sample Performance Tracking

```python
# In notebook: 05_production_monitoring.ipynb

from quant_alpha.monitoring import PerformanceTracker, ModelDriftDetector
import pandas as pd

# Initialize trackers
perf_tracker = PerformanceTracker()
drift_detector = ModelDriftDetector(rolling_window_days=30)

# Load recent predictions and actuals
predictions = pd.read_csv('results/predictions/latest_signals.csv')
recent_returns = data.groupby('ticker')['close'].pct_change(5).shift(-5)

# Daily updates
for date in predictions['date'].unique():
    date_preds = predictions[predictions['date'] == date]['prediction']
    date_returns = recent_returns.loc[date]
    
    # Update trackers
    perf_tracker.update(date, date_preds, date_returns)
    drift_detector.update(date, date_preds, date_returns)
    
    # Check drift
    if drift_detector.detect_drift():
        print(f"⚠️ MODEL DRIFT DETECTED on {date}")
        print(f"   Z-score: {drift_detector.z_score:.2f}")
        print(f"   Recommendation: Schedule retraining")
```

### 4.2 Factor Importance Tracking

```python
# Cell 2: Monitor feature importance over time

feature_importance_over_time = []

for fold_idx, model in enumerate(models):
    fold_importance = model.feature_importance
    fold_importance['fold'] = fold_idx
    feature_importance_over_time.append(fold_importance)

importance_df = pd.concat(feature_importance_over_time)
top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10)

print("Top 10 Most Important Features:")
print(top_features)

# Check for stability
variance = importance_df.groupby('feature')['importance'].std()
unstable_features = variance[variance > variance.mean()].index.tolist()

if unstable_features:
    print(f"\n⚠️ Unstable features (high variance): {unstable_features}")
    print("   These features may need data quality review")
```

---

## 5. Deployment Workflow

### 5.1 Pre-Deployment Checklist

Before deploying a new model to production:

```python
# In notebook: 06_research_tearsheet.ipynb Cell 1

deployment_checklist = {
    'Data Quality': {
        'No NaN columns': features.isnull().sum().max() == 0,
        'No infinite values': not np.any(np.isinf(features.values)),
        'Date range valid': data.index.get_level_values('date').is_monotonic_increasing,
    },
    'Model Performance': {
        'OOS IC > 0.005': results['lightgbm']['metrics']['information_coefficient'] > 0.005,
        'OOS t-stat > 2.5': results['lightgbm']['metrics']['t_statistic'] > 2.5,
        'Sharpe > 1.0': results['lightgbm']['metrics']['sharpe_ratio'] > 1.0,
    },
    'Backtest Results': {
        'CAGR > 10%': backtest_results['metrics']['cagr'] > 0.10,
        'Sharpe > 1.5': backtest_results['metrics']['sharpe_ratio'] > 1.5,
        'Max DD < 15%': abs(backtest_results['metrics']['max_drawdown']) < 0.15,
        'Win rate > 51%': backtest_results['metrics']['win_rate'] > 0.51,
    },
    'Risk Controls': {
        'Position limits set': config.MAX_POSITION_SIZE <= 0.10,
        'Concentration cap enforced': config.HHI_LIMIT <= 0.05,
        'Trailing stop enabled': config.TRAILING_STOP_PCT > 0,
        'Drawdown trigger set': config.MAX_DRAWDOWN_TRIGGER < -0.15,
    },
}

# Print checklist
print("DEPLOYMENT PRE-FLIGHT CHECKLIST:")
print("=" * 60)

all_pass = True
for category, checks in deployment_checklist.items():
    print(f"\n{category}:")
    for check_name, check_result in checks.items():
        status = "✓ PASS" if check_result else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not check_result:
            all_pass = False

print("\n" + "=" * 60)
if all_pass:
    print("✓ ALL CHECKS PASSED - Ready for deployment")
else:
    print("✗ SOME CHECKS FAILED - Address issues before deploying")
```

### 5.2 Deploy to Production

```bash
# After checklist passes:

# 1. Archive current production model
python scripts/deploy_model.py --archive-current --backup

# 2. Promote new model to production
python scripts/deploy_model.py \
  --model lightgbm \
  --ensemble-weights lightgbm=0.5,xgboost=0.3,catboost=0.2 \
  --environment production \
  --enable-monitoring

# 3. Verify deployment
python scripts/deploy_model.py --test

# 4. Start monitoring
python main.py monitor --live-stream
```

---

## 6. Experiment Logging & Reproducibility

### 6.1 Experiment Tracking

Record experiments for reproducibility:

```python
# At end of notebook, save metadata

experiment_metadata = {
    'experiment_id': 'exp_custom_momentum_20240315',
    'timestamp': pd.Timestamp.now().isoformat(),
    'features_used': list(features.columns),
    'model_type': 'GBDT Ensemble',
    'training_dates': f"{training_start} to {training_end}",
    'backtest_dates': f"{backtest_start} to {backtest_end}",
    'metrics': {
        'oos_ic': results['lightgbm']['metrics']['information_coefficient'],
        'oos_sharpe': results['lightgbm']['metrics']['sharpe_ratio'],
        'backtest_cagr': backtest_results['metrics']['cagr'],
        'backtest_sharpe': backtest_results['metrics']['sharpe_ratio'],
        'max_drawdown': backtest_results['metrics']['max_drawdown'],
    },
    'notes': 'Custom momentum factor combining 4 horizons. Passed all alpha gates.',
}

# Save metadata
import json
with open('results/experiment_metadata.json', 'w') as f:
    json.dump(experiment_metadata, f, indent=2, default=str)

print("✓ Experiment logged:", experiment_metadata['experiment_id'])
```

### 6.2 Reproducible Runs

```bash
# Reproduce an exact past run:
git checkout [COMMIT_HASH]
python main.py pipeline --seed 42 --log-level INFO

# Or use saved experiment config:
python main.py pipeline --load-config results/experiment_config.yaml
```

---

## Next Steps

1. **Run factor development cycle**: Follow [Factor Research Workflow](#1-factor-research-workflow)
2. **Train and compare models**: Follow [Model Development](#2-model-development--comparison)
3. **Backtest optimizations**: Follow [Backtesting](#3-backtesting--optimization-comparison)
4. **Monitor performance**: Follow [Monitoring](#4-model-monitoring--validation)
5. **Deploy to production**: Follow [Deployment](#5-deployment-workflow)
