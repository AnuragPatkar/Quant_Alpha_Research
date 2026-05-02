# Frequently Asked Questions (FAQ)

> **Purpose**: Quick answers to common setup, usage, and troubleshooting questions.

---

## Installation & Environment

### Q: I get `ModuleNotFoundError: No module named 'quant_alpha'`
**A**: The package isn't installed in editable mode. Run:
```bash
cd quant_alpha_research
pip install -e .
```
Verify with:
```python
import quant_alpha
print(quant_alpha.__file__)
```

### Q: I get `ImportError: libomp.dylib not found` on Mac
**A**: OpenMP library issue. Install via Homebrew:
```bash
brew install libomp
```
Or use conda (simpler):
```bash
conda create -n quant-alpha python=3.11
conda install -c conda-forge libopenblas mkl
pip install -e .[dev]
```

### Q: How much RAM do I need?
**A**: Minimum 16 GB. For reference:
- 500 tickers × 2500 days × 120 features = ~5.2 GB in memory
- During training, add ~2-3 GB for model state
- Gradient boosting models can spike to 10-15 GB temporarily

**If memory-constrained**: Use `--sample-size 0.1` to test on 50 tickers (~500 MB).

### Q: Can I use Python 3.10 instead of 3.11?
**A**: Yes, the package supports 3.9+. But 3.11 is recommended for performance (faster interpreter, better type checking). If using 3.10:
```bash
# Update setup.py
python_requires=">=3.9"
```

### Q: How do I switch Python versions?
**A**: With conda:
```bash
conda create -n quant-alpha-py310 python=3.10
conda activate quant-alpha-py310
pip install -e .[dev]

# Switch back anytime
conda activate quant-alpha
```

---

## Data & Setup

### Q: Where do I get the S&P 500 constituents file?
**A**: You need the historical S&P 500 components CSV. Sources:
1. **S&P Indices** (official): https://www.spindices.com/indices/equity/sp-500
2. **Wikipedia**: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies (has history)
3. **Yahoo Finance**: Scrape via `yfinance`

Then run:
```bash
python scripts/create_membership_mask.py \
  --constituents-csv "S&P 500 Historical Components & Changes.csv"
```

### Q: How do I download historical data?
**A**: Use the download script (first-time, slow ~30-60 minutes):
```bash
python scripts/download_data.py \
  --universe sp500 \
  --start-date 2015-01-01 \
  --fundamentals --earnings
```

For incremental updates (fast, ~5 minutes):
```bash
python scripts/download_data.py --incremental
```

### Q: Can I use data other than S&P 500?
**A**: Yes, modify `config/settings.py`:
```python
# Change universe
UNIVERSE = ['AAPL', 'MSFT', 'GOOG', 'AMZN']  # Custom list

# Or use a different index
UNIVERSE_FILE = 'data/nasdaq_100.txt'
```

But the pipeline expects `(date, ticker)` MultiIndex. You'll need to:
1. Download data for your tickers
2. Create a membership mask for your universe
3. Update factor logic if needed

### Q: Why do I get different prices than Yahoo Finance?
**A**: `yfinance` returns adjusted prices (dividend-adjusted, split-adjusted). This is correct. If you see discrepancies:
1. Use `Close` not `Adj Close` (they should be identical)
2. Check the download date (yfinance updates daily at ~4 PM ET)
3. Verify ticker symbol (e.g., `BRK.B` not `BRK-B`)

### Q: What if I have missing data?
**A**: The pipeline forward-fills for up to 5 trading days. Beyond that:
- **Price data**: Drops ticker-date pair (NaN removed)
- **Fundamentals**: Uses prior quarter (up to 90 days stale)
- **Earnings**: Uses consensus estimate until actual reported

Check:
```python
from quant_alpha.data import DataManager
dm = DataManager()
data = dm.get_data()
print(data.isnull().sum())
```

---

## Model Training & Backtesting

### Q: How long does model training take?
**A**: Depends on data size:
- **Sample (50 tickers)**: ~15 minutes
- **Full S&P 500**: 3-6 hours (depends on CPU cores)
- **With hyperparameter tuning**: Add 2-4 hours

Speed it up:
```bash
# Use fewer cores
python scripts/train_models.py --n-jobs 2

# Use sample data
python scripts/train_models.py --sample-size 0.1

# Skip expensive steps
python scripts/train_models.py --skip-feature-selection
```

### Q: Why is my backtest Sharpe ratio unrealistically high?
**A**: Possible causes:
1. **Look-ahead bias**: Fundamental data date-stamped incorrectly
2. **Survivorship bias**: Used current S&P 500 instead of historical constituents (fix: use membership mask)
3. **Too many parameters**: Overfitted to backtest period (use walk-forward CV)
4. **Slippage underestimated**: Set `commission_bps=15`, `spread_bps=10`, `slippage_bps=5`

Check:
```python
# Verify no forward data leakage
features['earnings_date'] <= backtest_date  # Should all be True
fundamentals['report_date'] + 90 <= backtest_date  # Should all be True
```

### Q: Can I use intraday signals (minute data)?
**A**: Not directly—the pipeline uses daily data. Workaround:
1. Compute intraday signal for a day
2. Use as of market close (after signal fully formed)
3. Execute next day at open

The current pipeline closes on signals at next-day open already, so it's compatible with same-day intraday signals.

### Q: Why do my model predictions have no variance?
**A**: Possible causes:
1. **Feature columns all NaN**: Check `factors_df.isnull().sum()`
2. **Target all same value**: Verify `targets.nunique() > 1`
3. **Model underfitting**: Try increasing `max_depth`, `num_leaves`
4. **Extremely stable market**: Unlikely but possible in sideways market (verify cross-sectional variance)

Debug:
```python
from quant_alpha.models import WalkForwardTrainer
features_clean = features.dropna(axis=1, how='all')
targets_clean = targets.dropna()
print(f"Features: {features_clean.shape}")
print(f"Target variance: {targets_clean.var():.8f}")  # Should be > 1e-6
```

### Q: How do I compare two models?
**A**: See [Experiments Guide](experiments.md) → [Model Development](experiments.md#21-train-multiple-models-3-4-hours). Quick check:
```python
# Train two models
from quant_alpha.models import WalkForwardTrainer, XGBoostModel, LightGBMModel

for model_class in [XGBoostModel, LightGBMModel]:
    trainer = WalkForwardTrainer(model_class=model_class)
    _, _, metrics = trainer.train(X=features, y=targets)
    print(f"{model_class.__name__} IC: {metrics['information_coefficient']:.6f}")
```

---

## Production & Deployment

### Q: Can I update the model without stopping the system?
**A**: You can deploy in parallel:
```bash
# Current production: model_v1
# New model: model_v2 (being trained)

# Once v2 ready:
python scripts/deploy_model.py --switch-to model_v2

# This swaps at next rebalance (doesn't interrupt execution)
```

For zero-downtime, use a rolling deployment (advanced).

### Q: What's the difference between staging and production?
**A**: 
- **Staging**: Test environment, can break without consequences
- **Production**: Live environment, real capital at risk

Before deploying to production:
1. Backtest on staging environment
2. Soak-test for 7+ days (live inference, no trading)
3. Monitor data quality and model drift
4. All pre-flight checks passing (see [Contributing Guide](contributing.md#43-deployment-checklist))

### Q: How do I monitor model performance live?
**A**: Use the monitoring dashboard:
```bash
python main.py monitor --live-stream
```

Or programmatically:
```python
from quant_alpha.monitoring import PerformanceTracker, ModelDriftDetector

tracker = PerformanceTracker()
drift = ModelDriftDetector(rolling_window_days=30)

# Update daily
for date, predictions, actuals in live_feed:
    tracker.update(date, predictions, actuals)
    drift.update(date, predictions, actuals)
    
    if drift.detect_drift():
        print(f"⚠️ Drift detected: {drift.z_score:.2f}")
```

### Q: What should I do if model drift is detected?
**A**: 
1. **Assess severity**: Is z-score > 3.0? Is IC degraded > 50%?
2. **Check data quality**: Run `python scripts/diagnose_data.py`
3. **Retrain if needed**: `python scripts/train_models.py --force`
4. **Notify stakeholders**: Send alert via Slack/email

See [Monitoring & Validation](experiments.md#4-model-monitoring--validation) for detailed workflow.

---

## Factor Development

### Q: How do I know if my factor is good?
**A**: It must pass these tests (see [Factor Research Workflow](experiments.md#11-engineer--validate-a-new-factor-2-3-hours)):
1. **IC t-statistic > 2.5**: Statistical significance
2. **Long-short spread > 0.01% daily**: Economic significance
3. **Minimal autocorrelation**: Signal isn't just lagged price
4. **Monotonic ranking power**: Deciles show linear return pattern

Check:
```python
from quant_alpha.research import FactorAnalyzer

analyzer = FactorAnalyzer(data)
ic = analyzer.compute_ic(factor, returns)
print(f"IC t-stat: {ic.mean() / ic.std() * np.sqrt(252):.2f}")  # Need > 2.5

ls = analyzer.compute_quantile_returns(factor, returns, n_quantiles=10)
print(f"Long-short spread: {(ls.iloc[-1] - ls.iloc[0]) * 100:.4f}%")  # Need > 0.01%
```

### Q: Can I use earnings announcement dates directly?
**A**: No, that's look-ahead bias! Earnings aren't available until after announcement. Workaround:
```python
# Load earnings announcement dates
from datetime import timedelta
announcement_date = earnings.loc['2024-01-20']

# You can only use data AFTER announcement
available_date = announcement_date + timedelta(days=1)

# So on 2024-01-20, you don't know if AAPL beat or missed
# Only on 2024-01-21 onwards can you trade on the news
```

### Q: Why is my factor correlated with other known factors?
**A**: High correlation (>0.8) is normal—most factors correlate with value, momentum, or quality. But:
1. **Is correlation > 0.95?** Then it's essentially a duplicate—merge or discard
2. **Is correlation 0.8-0.95?** Creates redundancy—consider removing lower t-stat factor
3. **Is correlation \< 0.8?** Signals are sufficiently orthogonal—keep both

Check:
```python
import pandas as pd

# Compute correlation across all factors
factor_corr = factors_df.corr()
high_corr_pairs = (factor_corr.abs().stack()
                   .sort_values(ascending=False)
                   .drop_duplicates()
                   .head(10))
print(high_corr_pairs)
```

---

## Troubleshooting & Errors

### Q: I get `ValueError: could not convert string to float` during data load
**A**: Malformed CSV file. Check:
```python
import pandas as pd

# Try loading the file
try:
    df = pd.read_csv('data/raw/sp500_prices/AAPL.csv')
except ValueError as e:
    print(f"Error: {e}")
    # Check for non-numeric values, extra quotes, bad lines
    df = pd.read_csv('data/raw/sp500_prices/AAPL.csv', on_bad_lines='skip')
```

### Q: I get `KeyError: 'date'` after loading data
**A**: Data doesn't have 'date' column or wrong index name. Fix:
```python
# Check column names
print(data.columns)

# Rename if needed
data = data.rename(columns={'Date': 'date', 'Ticker': 'ticker'})

# Or set index
data = data.set_index(['date', 'ticker'])
```

### Q: GPU memory error during training
**A**: Models are using GPU. Either:
1. Reduce batch size
2. Use CPU only (slower but less RAM)
3. Upgrade GPU memory

Disable GPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only

from quant_alpha.models import LightGBMModel
model = LightGBMModel(params={'device': 'cpu'})
```

### Q: Test suite hangs / timeout
**A**: Long-running tests. Run specific tests:
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run with timeout
pytest tests/ --timeout=300  # 5 minute timeout

# Or run unit tests only (integration tests are slow)
pytest tests/unit/ -v
```

---

## Performance & Optimization

### Q: How do I speed up factor computation?
**A**: Increase workers:
```python
from quant_alpha.features import FactorRegistry

registry = FactorRegistry()
factors = registry.compute_all(data, max_workers=8)  # Default is 4
```

Or skip expensive factors:
```python
# Disable slow factors
registry = FactorRegistry()
registry.skip_factors(['technical.advanced_pattern_recognition'])  # Too slow

factors = registry.compute_all(data)
```

### Q: How do I reduce memory usage?
**A**:
1. **Use smaller universe**: 50 tickers uses ~10x less memory than 500
2. **Use float32 instead of float64**: 50% memory savings, negligible accuracy loss
3. **Process in rolling windows**: Don't load entire 10 years at once

```python
# Load in 1-year chunks
for year in range(2015, 2026):
    data = dm.get_data(
        start_date=f'{year}-01-01',
        end_date=f'{year}-12-31'
    )
    # Process year's data
```

### Q: Can I parallelize backtest runs?
**A**: Yes, use multiprocessing:
```python
from multiprocessing import Pool

def backtest_config(params):
    engine = BacktestEngine(**params)
    return engine.run(signals, prices)

configs = [
    {'method': 'mean_variance'},
    {'method': 'kelly'},
    {'method': 'risk_parity'},
]

with Pool(3) as p:
    results = p.map(backtest_config, configs)
```

---

## Additional Resources

- **Setup Help**: See [Setup Guide](setup.md)
- **CLI Commands**: See [Usage Guide](usage.md)
- **Data Questions**: See [Data Guide](data.md)
- **Development**: See [Contributing Guide](contributing.md)
- **Architecture**: See [Architecture Guide](architecture.md)
- **Experiments**: See [Experiments Guide](experiments.md)

**Still stuck?** Open a GitHub issue with:
1. What you tried
2. Error message (full traceback)
3. Your environment (OS, Python version, installed packages)
4. Minimal reproducible example
