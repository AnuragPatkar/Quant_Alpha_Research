# Usage Guide & Examples

> **Purpose**: Comprehensive guide to using the platform via CLI, Python API, and Jupyter notebooks with copy-paste-ready examples.

---

## 1. Quick Start (5 Minutes)

### 1.1 Run the Full Pipeline

```bash
# Activate environment
source venv/bin/activate  # or: conda activate quant-alpha

# Execute full production pipeline (Data → Inference → Backtest → Report)
python main.py pipeline --all

# Or run specific phases
python main.py pipeline --skip-data --train --backtest
```

**What happens**:
1. ✓ Downloads latest S&P 500 price, fundamental, and earnings data
2. ✓ Computes 110+ factors using parallel thread pool
3. ✓ Trains walk-forward GBDT ensemble (3+ hours)
4. ✓ Generates out-of-sample alpha signals
5. ✓ Backtests signals with transaction costs
6. ✓ Optimizes portfolios (Mean-Variance, Kelly, Risk Parity)
7. ✓ Generates equity curves and performance reports

**Output files**:
```
results/
├── predictions/latest_signals.csv      # Alpha signals for next rebalance
├── backtests/backtest_results.pkl      # Full simulation state
└── reports/performance_report.html     # Visual dashboard
```

---

## 2. Command-Line Interface (CLI)

### 2.1 Main Entry Point: `main.py`

Unified orchestrator for all pipeline phases:

```bash
python main.py [COMMAND] [OPTIONS]
```

#### Available Commands

| Command | Purpose |
|---------|---------|
| `pipeline` | Execute full or partial research DAG |
| `update-data` | Fetch latest market data only |
| `train` | Train or retrain ML models |
| `backtest` | Run historical backtest simulation |
| `optimize` | Generate portfolio weights |
| `report` | Generate performance analysis |
| `monitor` | Start live monitoring dashboard |
| `validate` | Run data quality and model validation checks |

#### Global Options

```bash
-h, --help              # Show help
--env {dev, prod}       # Environment (default: development)
--log-level {DEBUG, INFO, WARNING, ERROR}  # Logging verbosity
--workers N             # Number of parallel workers (default: 4)
--seed SEED             # Random seed for reproducibility (default: 42)
```

### 2.2 Pipeline Execution

**Full Pipeline (All Phases)**:

```bash
python main.py pipeline --all
```

**Selective Phases**:

```bash
# Skip data download, only train and backtest
python main.py pipeline --skip-data --train --backtest

# Only generate predictions (skip training)
python main.py pipeline --skip-data --skip-train --predict

# Only run backtest on existing predictions
python main.py pipeline --skip-data --skip-train --backtest

# Only generate report from existing results
python main.py pipeline --skip-data --skip-train --skip-backtest --report
```

**Data Download Only**:

```bash
python main.py update-data --start-date 2024-01-01 --fundamentals --earnings
```

### 2.3 Model Training

```bash
# Full walk-forward training (12+ hours for full S&P 500)
python main.py train --full

# Quick training on subset (test mode, 1 hour)
python main.py train --sample-size 0.1 --max-tickers 50

# Retrain single model
python main.py train --model lightgbm

# Retrain with specific hyperparameters
python main.py train --params '{"max_depth": 8, "num_leaves": 128}'
```

### 2.4 Backtesting

```bash
# Backtest with default settings (Mean-Variance optimization)
python main.py backtest --start-date 2024-01-01

# Backtest with specific optimization method
python main.py backtest --opt-method risk_parity
python main.py backtest --opt-method kelly
python main.py backtest --opt-method mean_variance

# Backtest with custom parameters
python main.py backtest \
  --initial-capital 2000000 \
  --commission-bps 5 \
  --max-positions 20 \
  --rebalance-freq weekly

# Compare multiple portfolio construction methods
python main.py backtest --opt-method all
```

### 2.5 Portfolio Optimization

```bash
# Generate weights for next rebalance
python main.py optimize --method mean_variance

# Generate using different methods for comparison
python main.py optimize --method kelly
python main.py optimize --method risk_parity
python main.py optimize --method black_litterman

# Save weights to CSV
python main.py optimize --output results/weights_next_rebalance.csv
```

### 2.6 Generate Reports

```bash
# Generate comprehensive performance report
python main.py report --type performance

# Generate factor attribution report
python main.py report --type attribution

# Generate all reports
python main.py report --all

# Save to specific format
python main.py report --format html  # or json, csv, pdf
```

### 2.7 Validation & Diagnostics

```bash
# Run data quality checks
python main.py validate --type data

# Check model drift
python main.py validate --type drift

# Validate factor efficacy
python main.py validate --type factors

# Run all validations
python main.py validate --all
```

---

## 3. Python API Usage

For programmatic access in Python scripts or notebooks:

### 3.1 Data Access

```python
from quant_alpha.data import DataManager

# Initialize data manager
dm = DataManager()

# Get unified price + fundamentals matrix
data = dm.get_data(
    start_date='2023-01-01',
    end_date='2024-12-31',
    include_fundamentals=True,
    include_alternative=True
)

print(data.shape)           # (252d × 500t, 120 features)
print(data.columns)         # ['open', 'high', 'low', 'close', 'volume', 'eps', ..., 'momentum_252', ...]
print(data.index)           # MultiIndex(date, ticker)

# Get specific ticker time series
aapl_prices = data.loc[pd.IndexSlice[:, 'AAPL'], 'close']

# Get specific date cross-section (all tickers)
latest_prices = data.xs(data.index.get_level_values('date').max(), level='date')
```

### 3.2 Feature Engineering

```python
from quant_alpha.features import FactorRegistry

# Initialize registry (auto-discovers all registered factors)
registry = FactorRegistry()

# Compute all 110+ factors (multi-threaded, ~5 minutes)
factors_df = registry.compute_all(
    data=data,
    max_workers=4,
    verbose=True
)

print(f"Computed {len(factors_df.columns)} factors")
# Output: Computed 112 factors

# Get specific factor
momentum_factor = factors_df['momentum_252']

# Get all factors in category
technical_factors = {col: factors_df[col] for col in factors_df.columns 
                    if 'momentum' in col or 'mean_reversion' in col}
```

### 3.3 Model Training

```python
from quant_alpha.models import XGBoostModel, CatBoostModel, WalkForwardTrainer

# Prepare targets (e.g., 5-day forward returns)
targets = data['close'].pct_change(5).shift(-5)  # Use correct alignment!

# Initialize trainer with institutional walk-forward parameters
trainer = WalkForwardTrainer(
    model_class=XGBoostModel,
    min_train_months=36,      # 3 years minimum
    test_months=6,             # 6-month test windows
    step_months=3,             # Quarterly rebalance
    embargo_days=21,           # 21-day embargo vs. test set
    inner_val_fraction=0.2,    # 20% validation within train
    n_jobs=4
)

# Train (expanding-window, ~3 hours)
oos_predictions, trained_models, metrics = trainer.train(
    X=factors_df,
    y=targets
)

print(f"Information Coefficient: {metrics['information_coefficient']:.4f}")
print(f"Sharpe Ratio (OOS): {metrics['sharpe_ratio']:.2f}")
print(f"Model trained on {len(trained_models)} folds")
```

### 3.4 Portfolio Optimization

```python
from quant_alpha.optimization import Allocator
import numpy as np

# Expected returns and covariance matrix from model predictions
mu = oos_predictions.mean()  # Cross-sectional mean
cov = oos_predictions.cov()  # Covariance matrix

# Initialize allocator
allocator = Allocator(
    max_weight=0.10,          # 10% position limit
    min_weight=0.0,           # No shorts for this example
    target_leverage=1.0,      # 100% invested
    sector_bounds=0.05        # ±5% relative to index
)

# Generate weights using different methods
weights_mv = allocator.optimize(
    method='mean_variance',
    expected_returns=mu,
    cov_matrix=cov,
    risk_aversion=2.0
)

weights_kelly = allocator.optimize(
    method='kelly',
    expected_returns=mu,
    cov_matrix=cov,
    kelly_fraction=0.25  # 1/4 Kelly to reduce volatility
)

weights_erc = allocator.optimize(
    method='risk_parity',
    cov_matrix=cov
)

print(f"Mean-Variance sum: {weights_mv.sum():.2f}")
print(f"Kelly sum: {weights_kelly.sum():.2f}")
print(f"ERC sum: {weights_erc.sum():.2f}")
print(f"MV top 5: {weights_mv.nlargest(5)}")  # Top 5 holdings
```

### 3.5 Backtesting

```python
from quant_alpha.backtest import BacktestEngine

# Run backtest with equity curve and trade log
engine = BacktestEngine(
    initial_capital=1_000_000,
    commission_bps=10,
    spread_bps=5,
    slippage_bps=2,
    rebalance_frequency='daily',
    target_volatility=0.15,
    max_adv_participation=0.02
)

# Run simulation
results = engine.run(
    signals=oos_predictions,          # Model predictions (alpha signals)
    prices=data[['open', 'high', 'low', 'close', 'volume']],
    optimization_method='mean_variance',
    top_n=50,                         # Top 50 long positions
    is_weights=False                  # Signals are rankings (not explicit weights)
)

# Access results
equity_curve = results['equity_curve']
trades = results['trade_log']
metrics = results['metrics']

print(f"CAGR: {metrics['cagr']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Total trades: {len(trades)}")
```

### 3.6 Factor Analysis & Research

```python
from quant_alpha.research import FactorAnalyzer

# Analyze factor efficacy
analyzer = FactorAnalyzer(data=data)

# Compute Information Coefficient (cross-sectional daily)
ic_timeseries = analyzer.compute_ic(
    factor=factors_df['momentum_252'],
    returns=data['close'].pct_change(5).shift(-5),
    correlation_type='spearman'
)

print(f"Mean IC: {ic_timeseries.mean():.4f}")
print(f"IC t-stat: {ic_timeseries.mean() / ic_timeseries.std() * np.sqrt(252):.2f}")

# Compute quantile returns (long-short spread)
quantile_returns = analyzer.compute_quantile_returns(
    factor=factors_df['value_pe'],
    returns=data['close'].pct_change(5).shift(-5),
    n_quantiles=10
)

print(f"Long-Short spread (L/S daily): {quantile_returns[9] - quantile_returns[0]:.4%}")
```

### 3.7 Model Monitoring

```python
from quant_alpha.monitoring import ModelDriftDetector, DataQualityMonitor

# Initialize drift detector
drift_detector = ModelDriftDetector(rolling_window_days=30)

# Update daily with new predictions and actuals
for date, predictions, actuals in daily_data:
    drift_detector.update(date, predictions, actuals)
    
    # Check for concept drift
    if drift_detector.detect_drift():
        print(f"⚠️ Drift detected on {date}: z_score={drift_detector.z_score:.2f}")

# Initialize data quality monitor
quality_monitor = DataQualityMonitor()
quality_monitor.set_reference_data(historical_data)

# Validate incoming data
status = quality_monitor.check_incoming_data(new_data, data_type='prices')
print(f"Data quality: {status['overall_status']}")  # PASS, WARNING, or FAIL
```

---

## 4. Notebook-Based Research Workflow

The `/notebooks` directory contains 6 research notebooks organized as a workflow:

### 4.1 01_data_exploration.ipynb

Explore raw data structure, coverage, and distributions:

```python
# Cell 1: Load data
from quant_alpha.data import DataManager
dm = DataManager()
data = dm.get_data(start_date='2023-01-01')

# Cell 2: Data quality summary
print(data.describe())
print(f"Missing values:\n{data.isnull().sum(axis=0)}")

# Cell 3: Visualize price patterns
import matplotlib.pyplot as plt
data['close'].xs('AAPL', level='ticker').plot(figsize=(12, 6))
plt.title('AAPL Historical Prices')
plt.show()

# Cell 4: Correlation heatmap
import seaborn as sns
corr = data[['open', 'high', 'low', 'close', 'volume']].corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title('Price Component Correlations')
plt.show()
```

### 4.2 02_factor_research.ipynb

Engineer and validate new factors:

```python
# Cell 1: Compute a new factor
from quant_alpha.features import BaseFactor, FactorRegistry

@FactorRegistry.register()
class CustomMomentum(BaseFactor):
    def compute(self, data):
        return data.groupby('ticker')['close'].pct_change(252)

# Cell 2: Analyze factor efficacy
from quant_alpha.research import FactorAnalyzer
analyzer = FactorAnalyzer(data)
ic_ts = analyzer.compute_ic(factor, returns)
print(f"Factor IC: {ic_ts.mean():.4f} (t-stat: {ic_ts.mean()/ic_ts.std():.2f})")

# Cell 3: Visualize decay
decay = analyzer.compute_alpha_decay(factor, returns, max_days=20)
decay.plot(title='Factor Alpha Decay')
```

### 4.3 03_model_development.ipynb

Train and tune ML models:

```python
# Cell 1: Initialize trainer
from quant_alpha.models import WalkForwardTrainer, XGBoostModel
trainer = WalkForwardTrainer(model_class=XGBoostModel)

# Cell 2: Train models (long cell, ~3 hours)
oos_preds, models, metrics = trainer.train(X=features, y=targets)

# Cell 3: Analyze out-of-sample performance
print(metrics)
# Information Coefficient: 0.0234
# Sharpe Ratio: 1.87
# t-statistic: 2.62 ✓ Passes alpha gate (t > 2.5)
```

### 4.4 04_backtest_analysis.ipynb

Historical backtesting and optimization:

```python
# Cell 1: Run backtest
from quant_alpha.backtest import BacktestEngine
engine = BacktestEngine(initial_capital=1_000_000)
results = engine.run(signals=oos_preds, prices=prices)

# Cell 2: Plot equity curve
results['equity_curve'].plot(figsize=(14, 6), title='Backtest Equity Curve')
plt.axhline(y=1_000_000, color='r', linestyle='--', label='Starting Capital')

# Cell 3: Compare optimization methods
for method in ['mean_variance', 'kelly', 'risk_parity']:
    results = engine.run(..., optimization_method=method)
    results['equity_curve'].plot(label=method)
plt.legend()
```

### 4.5 05_production_monitoring.ipynb

Monitor live/test set performance:

```python
# Cell 1: Track recent predictions
from quant_alpha.monitoring import PerformanceTracker
tracker = PerformanceTracker()

# Cell 2: Check for drift
tracker.update_with_actuals(recent_predictions, recent_returns)
if tracker.detect_drift():
    print("⚠️ Model drift detected - consider retraining")

# Cell 3: Performance metrics by date
tracker.rolling_metrics(window=21)  # 21-day rolling Sharpe, IC
```

### 4.6 06_research_tearsheet.ipynb

Generate research summary and publication-ready visuals:

```python
# Cell 1: Load results
import pandas as pd
results = pd.read_pickle('results/backtest_results.pkl')

# Cell 2: Factor attribution
attribution = engine.attribute_returns(
    returns=results['returns'],
    factor_exposures=features
)
attribution.plot(kind='barh')

# Cell 3: Summary statistics table
stats = {
    'CAGR': results['metrics']['cagr'],
    'Sharpe': results['metrics']['sharpe_ratio'],
    'Max DD': results['metrics']['max_drawdown'],
    'Win Rate': results['metrics']['win_rate'],
}
pd.DataFrame(stats, index=['Value']).to_csv('results/summary_stats.csv')
```

---

## 5. Advanced Usage Patterns

### 5.1 Custom Feature Development

```python
from quant_alpha.features import BaseFactor, FactorRegistry
import pandas as pd
import numpy as np

@FactorRegistry.register()
class MyCustomFactor(BaseFactor):
    """
    Custom research factor for testing.
    """
    def __init__(self, lookback=20, name='my_custom_factor'):
        super().__init__(
            name=name,
            category='custom',
            lookback_period=lookback,
            normalize=True,
            winsorize=True
        )
        self.lookback = lookback
    
    def compute(self, data):
        """
        Compute the factor.
        
        Args:
            data (pd.DataFrame): MultiIndex(date, ticker) with 'close', 'volume'
        
        Returns:
            pd.Series: Factor values with same index as input
        """
        # Example: Volume-price correlation
        returns = data.groupby('ticker')['close'].pct_change()
        vol_changes = data.groupby('ticker')['volume'].pct_change()
        
        # Rolling correlation
        corr = returns.rolling(window=self.lookback).corr(vol_changes)
        return corr

# Register and compute
registry = FactorRegistry()
factor_df = registry.compute_all(data)
# Now 'my_custom_factor' column available in output
```

### 5.2 Custom Model Configuration

```python
from quant_alpha.models import LightGBMModel, WalkForwardTrainer

# Define custom hyperparameters
custom_params = {
    'num_leaves': 128,
    'max_depth': 10,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'objective': 'regression',
    'metric': 'rmse',
    'boost_from_average': True,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
}

# Initialize with custom params
model = LightGBMModel(params=custom_params)

# Use in trainer
trainer = WalkForwardTrainer(model_class=LightGBMModel)
trainer.model_params = custom_params
results = trainer.train(X=features, y=targets)
```

### 5.3 Custom Optimization Constraints

```python
from quant_alpha.optimization import MeanVarianceOptimizer

# Sector constraints (example: limit tech to ±3% vs SPY)
sector_bounds = {
    'technology': {'lower': 0.27, 'upper': 0.33},  # SPY ~30%
    'healthcare': {'lower': 0.10, 'upper': 0.16},  # SPY ~13%
}

optimizer = MeanVarianceOptimizer(
    max_weight=0.12,
    sector_bounds=sector_bounds,
    min_positions=25,
    max_positions=75
)

weights = optimizer.optimize(
    expected_returns=mu,
    cov_matrix=cov,
    sector_mapping=ticker_sectors
)
```

### 5.4 Rolling Window Analysis

```python
from datetime import timedelta
import pandas as pd

# Analyze factor efficacy over rolling 252-day windows
all_ics = []
for date in pd.date_range(start='2023-01-01', end='2024-12-31', freq='M'):
    window_start = date - timedelta(days=252)
    window_data = data.loc[window_start:date]
    
    # Compute IC for this window
    analyzer = FactorAnalyzer(data=window_data)
    ic = analyzer.compute_ic(factor, returns)
    all_ics.append({'date': date, 'ic': ic.mean()})

ic_df = pd.DataFrame(all_ics)
ic_df.set_index('date').plot(title='Rolling Factor IC')
```

---

## 6. Troubleshooting & Debugging

### 6.1 Enable Debug Logging

```bash
# Run with DEBUG logging
ENV=development LOG_LEVEL=DEBUG python main.py pipeline --all

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 6.2 Check Intermediate Results

```python
# Save intermediate DataFrames for inspection
features_df.to_parquet('_debug_features.parquet')
oos_predictions.to_csv('_debug_predictions.csv')

# Inspect in another session
import pandas as pd
features = pd.read_parquet('_debug_features.parquet')
print(features.describe())
print(features.isnull().sum())
```

### 6.3 Memory Profiling

```bash
# Monitor memory during execution
pip install memory-profiler
python -m memory_profiler scripts/run_pipeline.py --sample-size 0.1
```

---

## 7. Next Steps

- **Implement a new factor**: See [02_factor_research.ipynb](../notebooks/02_factor_research.ipynb)
- **Optimize a new portfolio method**: See [04_backtest_analysis.ipynb](../notebooks/04_backtest_analysis.ipynb)
- **Deploy to production**: See [Contributing Guidelines](contributing.md)
- **Understand data flow**: See [Architecture Guide](architecture.md)
