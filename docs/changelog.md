# Changelog

All notable changes to the Quant Alpha Research Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Interactive Jupyter widgets for real-time factor analysis
- GPU acceleration via CuPy for large-scale feature computation
- Insider trading activity factor (Form 4 filings)
- Bayesian hyperparameter optimization with Optuna (replacing grid search)
- Multi-period ensemble with confidence weighting
- Production monitoring dashboard (Streamlit)
- Factor orthogonalization via PCA and ICA
- Automated report generation (PDF, email delivery)

### Changed
- Refactored `DataManager` for better caching and incremental updates
- Improved fundamental preprocessor with column name mapping (see [Raw Columns Validation](memories/user/quant_alpha_raw_columns_fix.md))
- Walk-forward trainer now supports expanding and rolling windows
- Optimization: Ledoit-Wolf shrinkage replaced with DCC-GARCH for volatility models

### Deprecated
- `_compute_legacy_features()` → Use `FactorRegistry` pattern
- Pickle-based model storage → Migrate to ONNX format

### Removed
- Support for Python 3.8 (now requires 3.9+)
- Old proprietary risk model (replaced with open-source CVXPY)

### Fixed
- [CRITICAL] Reporting lag not enforced in fundamental data extraction (Issue #142)
- [HIGH] Survivorship bias in backtests when using current S&P 500 (Issue #138)
- NaN columns in fundamental extraction due to key name mismatches (see [Raw Columns Fix](memories/user/quant_alpha_raw_columns_fix.md))
- Memory leak in factor registry during parallel computation (Issue #156)
- Incorrect date alignment in walk-forward cross-validation (Issue #151)

### Security
- Credentials now stored in `.env` file (secrets not in code)
- API keys rotated and moved to GitHub Secrets for CI/CD

---

## [0.1.0] - 2026-03-10

### Added
- **Core Data Pipeline**: `DataManager` with caching, `BaseLoader` abstraction for price/fundamental/earnings data
- **Feature Engineering**: `BaseFactor` abstract class, `FactorRegistry` with 110+ implemented factors
  - Technical: momentum, mean reversion, volatility, volume
  - Fundamental: value (P/E, EV/EBITDA), quality (ROE, margins), growth, financial health
  - Earnings: surprises, revisions, consensus estimates
  - Alternative: macro indicators (VIX, rates), sentiment
- **Machine Learning Pipeline**:
  - Walk-forward trainer with expanding window CV (36-month min train, 6-month test, 21-day embargo)
  - GBDT base models: LightGBM, XGBoost, CatBoost with custom loss functions
  - Ensemble via rank-averaging
  - Feature selection via permutation importance
- **Portfolio Optimization**:
  - Mean-Variance (Markowitz) with Ledoit-Wolf shrinkage
  - Kelly Criterion (fractional Kelly for risk control)
  - Risk Parity (Equal Risk Contribution)
  - Constraints: position limits, concentration caps (HHI), sector bounds, turnover limits
- **Backtesting Engine**:
  - Daily execution simulation with trade-by-trade logging
  - Realistic transaction costs: commission (10 bps), spread (5 bps), slippage (2 bps)
  - Almgren-Chriss market impact model
  - Risk controls: trailing stops, drawdown limits, concentration enforcement
  - Attribution analysis by factor exposure
  - Performance metrics: CAGR, Sharpe, Sortino, Max DD, Calmar, Win Rate
- **Research & Validation Tools**:
  - Information Coefficient (IC) computation and significance testing
  - Alpha decay analysis (forward-looking signal decay curves)
  - Factor quantile analysis (long-short spreads by decile)
  - Correlation analysis (pairwise factor orthogonality)
  - Regime detection (market state identification)
  - Significance testing (t-stats, p-values, effect sizes)
- **Monitoring & Alerts**:
  - Data quality monitoring with PSI (Population Stability Index) drift detection
  - Model drift detection (concept drift, prediction bias, label shift)
  - Performance tracking vs. live benchmark
  - Alert system for data failures and model degradation
- **Docker & Deployment**:
  - Multi-stage Dockerfile for containerized execution
  - Docker Compose for local and cloud deployment
  - Environment-based configuration (development vs. production)
- **Documentation**:
  - Comprehensive architecture guide with data flow diagrams
  - Setup & installation instructions (venv, conda, Docker)
  - Usage guide with CLI and Python API examples
  - Experiments workflow for factor development and model training
  - Data guide with schemas and validation rules
  - Contributing guidelines with testing standards
  - FAQ with troubleshooting

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## Release Naming Convention

- **Major (X.0.0)**: Breaking changes, incompatible API changes, major methodology shifts
- **Minor (0.Y.0)**: New features, backward-compatible additions
- **Patch (0.0.Z)**: Bug fixes, documentation, minor improvements

---

## Backlog (Future Releases)

### Planned for v0.2.0 (Q2 2026)
- [ ] Multi-period ensemble with confidence weighting
- [ ] Insider trading factor (Form 4 filings)
- [ ] Bayesian optimization (Optuna)
- [ ] Production monitoring dashboard
- [ ] Batch prediction API
- [ ] Factor orthogonalization (PCA/ICA)

### Planned for v0.3.0 (Q3 2026)
- [ ] GPU acceleration (CuPy)
- [ ] Distributed backtesting (Ray)
- [ ] ONNX model export
- [ ] Real-time streaming data support
- [ ] Model calibration and uncertainty quantification

### Planned for v1.0.0 (Q4 2026)
- [ ] Production API (REST + gRPC)
- [ ] Kubernetes deployment with auto-scaling
- [ ] Multi-asset class support (equities, fixed income, commodities)
- [ ] Advanced risk models (factors-based VaR, ES)
- [ ] Live trading execution via broker APIs

---

## Known Issues

### Current (v0.1.0)
- **[Medium]** Walk-forward retraining takes 3+ hours for full S&P 500 (parallelization in progress)
- **[Low]** Factor registry discovery can take 10+ seconds on first import (caching in progress)
- **[Low]** Memory usage can spike to 15+ GB during ensemble training (chunking in v0.2.0)

### Resolved
- ~~**[CRITICAL]** Reporting lag not enforced~~ → Fixed in latest commit
- ~~**[HIGH]** Survivorship bias in backtests~~ → Fixed with membership mask
- ~~**[HIGH]** Fundamental data extraction missing columns~~ → Fixed with key name mapping

---

## Contributors

- **Anurag Patkar** (@AnuragPatkar) — Core architecture, research infrastructure
- **Community Contributors** — Welcome! See [Contributing Guide](docs/contributing.md)

---

## Migration Guides

### From v0.0.1 to v0.1.0
No prior releases. Safe to upgrade from development version.

### Upcoming: v0.1.0 to v0.2.0
```python
# Feature extraction API change
# OLD (v0.1.0):
from quant_alpha.features import compute_features
features = compute_features(data)

# NEW (v0.2.0):
from quant_alpha.features import FactorRegistry
registry = FactorRegistry()
features = registry.compute_all(data)
```

---

## How to Report Issues

Found a bug? Please report with:
1. Exact error message and traceback
2. Minimal reproducible example
3. Your environment (OS, Python version, package versions)
4. Steps to recreate the issue

Open issue on [GitHub Issues](https://github.com/AnuragPatkar/quant_alpha_research/issues)

---

## How to Request Features

Feature requests welcome! Please:
1. Check existing issues/discussions for duplicates
2. Describe use case and expected behavior
3. Provide code example or sketch of implementation

Request on [GitHub Discussions](https://github.com/AnuragPatkar/quant_alpha_research/discussions)

---

## References

- **Quantitative Finance**:
  - Prado, M. L. D. (2018). *Advances in Financial Machine Learning*
  - Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
  - Sharpe, W. F. (1966). "Mutual fund performance"
  
- **Machine Learning**:
  - Breiman, L. (2001). "Random Forests"
  - Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
  
- **Optimization**:
  - Markowitz, H. (1952). "Portfolio Selection"
  - Thorp, E. O. (2008). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
  
- **Implementation Libraries**:
  - [LightGBM](https://github.com/microsoft/LightGBM)
  - [XGBoost](https://github.com/dmlc/xgboost)
  - [CVXPY](https://www.cvxpy.org/)
  - [Numba](https://numba.readthedocs.io/)
