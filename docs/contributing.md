# Contributing Guidelines

> **Purpose**: Standards and procedures for collaboration, code quality, testing, and deployment to production.

---

## 1. Development Workflow

### 1.1 Branching Strategy

We use **Git Flow** with feature branches (GitHub Flow variant):

```
main (production-ready)
├─ release/v0.2.0 (pre-production release branches)
├─ feature/new-factor (feature branches)
└─ fix/data-loader (bugfix branches)
```

### 1.2 Creating a Feature Branch

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b feature/my-new-factor

# Or use descriptive names:
git checkout -b feature/add-insider-trading-factor
git checkout -b fix/fundamental-loader-bug
git checkout -b refactor/optimize-memory-usage
```

### 1.3 Commit Standards

**Format**: Follow conventional commits for clarity

```bash
# Good commits
git commit -m "feat: add insider trading factor with winsorization"
git commit -m "fix: correct reporting lag application in fundamentals"
git commit -m "test: add validation for factor IC t-statistic"
git commit -m "docs: update architecture guide with new modules"
git commit -m "refactor: consolidate duplicate feature aggregation code"

# Avoid
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "asdf"
```

**Commit Types**:
- `feat:` — New feature
- `fix:` — Bug fix
- `test:` — Test additions/improvements
- `docs:` — Documentation
- `refactor:` — Code refactoring (no functional change)
- `perf:` — Performance improvement
- `style:` — Code style (formatting, linting)
- `chore:` — Build, dependency, or tooling changes

### 1.4 Pull Request Process

**Before submitting**:

```bash
# 1. Update local main
git checkout main
git pull origin main

# 2. Rebase feature onto main (clean history)
git checkout feature/my-new-factor
git rebase main

# 3. Run full test suite
pytest tests/ -v --tb=short

# 4. Run type checking
mypy quant_alpha/ --config-file mypy.ini

# 5. Format code
black quant_alpha/
flake8 quant_alpha/

# 6. Push to remote
git push origin feature/my-new-factor
```

**Pull Request Template** (on GitHub):

```markdown
## Description
Brief explanation of changes and why they're needed.

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Closes #123

## Testing
- [ ] Unit tests added
- [ ] Integration tests passing
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide (black, flake8)
- [ ] Type hints added (mypy passes)
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Changes backtest-reproducible (same seed → same results)
```

### 1.5 Merge & Deploy

**After PR approval**:

```bash
# Merge to main (via GitHub: "Squash and merge" recommended)
# Deletes feature branch automatically

# On local machine:
git checkout main
git pull origin main
git branch -d feature/my-new-factor
```

**Trigger CI/CD**:
- GitHub Actions automatically runs tests on PR
- Merge to main triggers deployment to staging
- Manual approval needed for production deployment

---

## 2. Code Quality Standards

### 2.1 Style Guide (PEP 8 + Black)

**Use Black for auto-formatting**:

```bash
# Format all Python files
black quant_alpha/

# Check formatting without changes
black --check quant_alpha/

# Format with line length 100
black --line-length 100 quant_alpha/
```

**Code style example**:

```python
# Good: Clear, well-documented
@FactorRegistry.register()
class MomentumFactor(BaseFactor):
    """
    Time-series momentum factor.
    
    Computes N-day returns as a mean-reversion signal.
    
    Parameters
    ----------
    lookback : int, default=252
        Number of days to compute momentum.
    normalize : bool, default=True
        Cross-sectional z-score normalization.
    """
    
    def __init__(self, lookback: int = 252, normalize: bool = True):
        super().__init__(
            name='momentum',
            category='technical',
            lookback_period=lookback,
            normalize=normalize,
            winsorize=True
        )
        self.lookback = lookback
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute momentum factor."""
        return data.groupby('ticker')['close'].pct_change(self.lookback)
```

### 2.2 Type Hints (mypy)

**All public functions must have type hints**:

```python
# Required
def optimize_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    method: str = 'mean_variance'
) -> Dict[str, float]:
    """
    Generate portfolio weights from signals.
    
    Parameters
    ----------
    signals : pd.DataFrame
        MultiIndex(date, ticker) with 'prediction' column.
    prices : pd.DataFrame
        MultiIndex(date, ticker) with 'close' column.
    method : str
        Optimization method ('mean_variance', 'kelly', or 'risk_parity').
    
    Returns
    -------
    Dict[str, float]
        Ticker → optimal weight mapping.
    """
    pass

# Run type checking
mypy quant_alpha/ --config-file mypy.ini
```

**Type checking config** (`mypy.ini`):

```ini
[mypy]
python_version = 3.11
ignore_missing_imports = True
disallow_untyped_defs = False  # Gradually enforce
show_error_codes = True
```

### 2.3 Linting (flake8)

**Check code quality issues**:

```bash
# Run flake8
flake8 quant_alpha/ --max-line-length=100 --ignore=E203,W503

# Expected output (or empty = all good)
# quant_alpha/features/momentum.py:45:5: E501 line too long
# quant_alpha/models/trainer.py:120:1: F841 local variable 'x' is assigned but never used
```

**Auto-fix some issues**:

```bash
pip install autopep8
autopep8 --in-place --aggressive quant_alpha/models/trainer.py
```

### 2.4 Documentation Standards

**All modules must have docstrings**:

```python
"""
Module: quant_alpha.features.technical.momentum
================================================
Implements time-series and cross-sectional momentum factors.

This module provides several momentum-based alpha factors designed to capture
intermediate-term price trends. Factors are orthogonalized via cross-sectional
normalization to isolate idiosyncratic signals.

Classes:
    MomentumFactor: Time-series momentum signal
    CrossSectionMomentum: Cross-sectional momentum relative to sector
    
Functions:
    compute_returns: Helper to compute multi-horizon returns
    
Usage:
    >>> from quant_alpha.features import FactorRegistry
    >>> registry = FactorRegistry()
    >>> factors = registry.compute_all(data)
    >>> momentum_signal = factors['momentum']
"""

import pandas as pd
import numpy as np
from typing import Optional
```

**Docstring format** (NumPy style):

```python
def compute_ic(factor: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Compute Information Coefficient (IC) time-series.
    
    Information Coefficient measures the predictive power of a factor as
    the Spearman correlation between factor values and subsequent returns,
    computed on a rolling daily basis.
    
    Parameters
    ----------
    factor : pd.Series
        Factor values (MultiIndex: date, ticker).
    returns : pd.Series
        Forward returns (MultiIndex: date, ticker). Typically 5-day returns.
    
    Returns
    -------
    pd.Series
        Daily IC time-series with datetime index.
        
    Raises
    ------
    ValueError
        If inputs have mismatched lengths or indexes.
        
    See Also
    --------
    compute_alpha_decay : Analyze signal decay over horizons.
    compute_quantile_returns : Analyze long-short spreads.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from quant_alpha.research import FactorAnalyzer
    >>> analyzer = FactorAnalyzer(data)
    >>> ic = analyzer.compute_ic(factor, returns)
    >>> print(f"Mean IC: {ic.mean():.6f}")
    """
    pass
```

---

## 3. Testing Standards

### 3.1 Test Structure

```
tests/
├── unit/
│   ├── test_features.py         # Factor computation tests
│   ├── test_models.py           # Model fitting/prediction tests
│   ├── test_data_loaders.py     # Data loading tests
│   ├── test_optimization.py     # Portfolio optimization tests
│   ├── test_backtest.py         # Backtesting engine tests
│   └── test_utils.py            # Utility function tests
├── integration/
│   ├── test_pipeline.py         # Full pipeline execution
│   ├── test_data_workflow.py    # Data loading → feature → model
│   └── test_backtest_end2end.py # Full backtest simulation
├── performance/
│   ├── test_scalability.py      # Large dataset handling
│   └── test_memory.py           # Memory efficiency
└── conftest.py                  # Pytest fixtures and config
```

### 3.2 Writing Unit Tests

```python
# tests/unit/test_features.py

import pytest
import pandas as pd
import numpy as np
from quant_alpha.features import MomentumFactor, FactorRegistry

class TestMomentumFactor:
    """Test suite for MomentumFactor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOG']
        index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
        
        data = pd.DataFrame({
            'close': np.random.randn(len(index)).cumsum() + 100,
        }, index=index)
        return data
    
    def test_momentum_computation(self, sample_data):
        """Test basic momentum computation."""
        factor = MomentumFactor(lookback=21)
        result = factor.compute(sample_data)
        
        # Assertions
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.dtype == np.float64
    
    def test_momentum_nan_handling(self, sample_data):
        """Test momentum with missing data."""
        sample_data_with_nans = sample_data.copy()
        sample_data_with_nans.iloc[50:60, 0] = np.nan
        
        factor = MomentumFactor(lookback=21)
        result = factor.compute(sample_data_with_nans)
        
        # First lookback_period values per ticker should be NaN
        assert result.iloc[:21].isna().sum() > 0
    
    def test_momentum_normalization(self):
        """Test cross-sectional normalization."""
        factor = MomentumFactor(normalize=True)
        
        # After normalization, cross-sectional mean should be ~0
        # (This is tested at the end of compute() pipeline)
        pass
    
    @pytest.mark.slow
    def test_momentum_performance(self, sample_data):
        """Test computation speed (marked as slow)."""
        factor = MomentumFactor()
        
        # Should compute in < 100ms for sample data
        import time
        start = time.time()
        result = factor.compute(sample_data)
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Momentum computation too slow: {elapsed:.2f}s"
```

### 3.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_features.py -v

# Run specific test class
pytest tests/unit/test_features.py::TestMomentumFactor -v

# Run specific test method
pytest tests/unit/test_features.py::TestMomentumFactor::test_momentum_computation -v

# Run with coverage report
pytest tests/ --cov=quant_alpha --cov-report=html

# Run excluding slow tests
pytest tests/ -v -m "not slow"

# Run with detailed output
pytest tests/ -v --tb=long
```

### 3.4 Test Fixtures (conftest.py)

```python
# tests/conftest.py

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_prices_multiindex():
    """Create sample OHLCV data."""
    dates = pd.date_range('2023-01-01', periods=252)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, len(index)),
        'high': np.random.uniform(100, 200, len(index)),
        'low': np.random.uniform(100, 200, len(index)),
        'close': np.random.uniform(100, 200, len(index)),
        'volume': np.random.randint(1_000_000, 100_000_000, len(index)),
    }, index=index)
    return data

@pytest.fixture
def sample_signals():
    """Create sample model predictions."""
    dates = pd.date_range('2023-01-01', periods=252)
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    signals = pd.Series(
        np.random.randn(len(index)),
        index=index,
        name='prediction'
    )
    return signals
```

### 3.5 Test Coverage Goals

**Minimum coverage by module**:

| Module | Target Coverage |
|--------|-----------------|
| `quant_alpha/data/` | 85% |
| `quant_alpha/features/` | 80% |
| `quant_alpha/models/` | 90% (critical) |
| `quant_alpha/backtest/` | 90% (critical) |
| `quant_alpha/optimization/` | 85% |
| `quant_alpha/research/` | 75% |

**Run coverage check**:

```bash
pytest tests/ --cov=quant_alpha --cov-report=term-missing --cov-fail-under=80
```

---

## 4. Deployment Process

### 4.1 Staging Deployment (Automated)

**Triggered on merge to main**:

```yaml
# .github/workflows/ci-staging.yml (example)

name: CI - Staging Deployment
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e .[dev]
      - run: pytest tests/ --cov=quant_alpha
      - run: mypy quant_alpha/
      - run: black --check quant_alpha/
      - run: flake8 quant_alpha/
      
  deploy-staging:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: docker build -t quant-alpha:staging .
      - run: docker push quant-alpha:staging
      - run: kubectl set image staging quant-alpha=quant-alpha:staging
```

### 4.2 Production Deployment (Manual)

**Only after staging validation**:

```bash
# 1. Tag release
git tag v0.2.0
git push origin v0.2.0

# 2. Create GitHub Release with notes
# (List all changes, highlight breaking changes)

# 3. Trigger production deployment
# (Via GitHub Actions or manual approval)

# 4. Monitor in production
python main.py monitor --live-stream

# 5. If rollback needed:
git tag v0.1.9
git push origin v0.1.9
# (Trigger deployment to prior version)
```

### 4.3 Deployment Checklist

**Before deploying to production**:

```markdown
## Pre-Deployment Checklist

- [ ] All tests passing (100% coverage maintained)
- [ ] Code review approved by 2+ senior engineers
- [ ] Backtest results replicated on staging environment
- [ ] No breaking changes to data contracts
- [ ] No changes to hyperparameters without approval
- [ ] Monitoring dashboards configured
- [ ] Rollback procedure documented
- [ ] Staging soak-test: 7+ days of live inference
- [ ] Data quality gates configured and passing
- [ ] Model drift detection enabled

**Sign-off**:
- Code Owner: _______________
- Tech Lead: _______________
- Risk Officer: _______________
```

---

## 5. Issue & Roadmap Management

### 5.1 Issue Template

When opening a GitHub issue:

```markdown
## Description
Clear description of the problem or feature request.

## Type
- [ ] Bug report
- [ ] Feature request
- [ ] Documentation
- [ ] Performance issue

## Steps to Reproduce (for bugs)
1. ...
2. ...
3. ...

## Expected vs. Actual Behavior
Expected: ...
Actual: ...

## Environment
- OS: Ubuntu 22.04
- Python: 3.11
- Branch: feature/my-factor

## Additional Context
Screenshots, error traces, or relevant code snippets.
```

### 5.2 Roadmap

Managed via GitHub Projects. Current priorities:

- **Q1 2024**: Insider trading factor + earnings surprise factor
- **Q2 2024**: Multi-period ensemble + Kelly criterion optimization
- **Q3 2024**: Production monitoring dashboard + drift alerts
- **Q4 2024**: Risk decomposition and factor attribution improvements

---

## 6. Code Review Checklist

**Reviewers should verify**:

- [ ] Code follows style guide (black, flake8, mypy passes)
- [ ] Tests added and passing (coverage maintained)
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] No hardcoded paths or credentials
- [ ] No new dependencies without justification
- [ ] Algorithm/math verified if new model or factor
- [ ] Backtest reproducibility maintained (deterministic)
- [ ] Data validation and error handling appropriate
- [ ] Performance impact acceptable
- [ ] No breaking changes to public APIs

---

## 7. Resources

- **Style Guide**: [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- **Docstring Format**: [NumPy Style](https://numpydoc.readthedocs.io/en/latest/format.html)
- **Type Hints**: [Python typing docs](https://docs.python.org/3/library/typing.html)
- **Git Flow**: [Atlassian Git Flow Tutorial](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- **Testing**: [Pytest Documentation](https://docs.pytest.org/)

---

## Getting Help

- **Quick questions**: Open discussion in GitHub Discussions
- **Bug reports**: Create issue with full reproduction steps
- **Feature requests**: Discuss in Q&A before implementing
- **Code review**: Tag maintainers (@AnuragPatkar) for review
- **Onboarding**: See [Setup Guide](setup.md) for environment setup

Thank you for contributing to Quant Alpha Research! 🚀
