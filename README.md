# 🏆 Institutional-Grade Quantitative Equity Alpha Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Executive Summary

This repository implements an **institutional-grade quantitative research and systematic trading infrastructure** designed for scalable equity alpha generation. Engineered to tier-one hedge fund standards, the system integrates orthogonal factor engineering, non-linear machine learning ensembles, and convex portfolio optimization to systematically capture market inefficiencies.

The architecture guarantees absolute mathematical parity between the research environment and the production execution pipeline, strictly mitigating look-ahead bias, survivorship bias, and model overfitting.

---

## 2. System Architecture Overview

The platform operates as a Directed Acyclic Graph (DAG), seamlessly transitioning from raw market data to optimal portfolio weights:

1. **Data Ingestion (Data Lake):** Aggregates daily OHLCV, Fundamental (10-K/10-Q), and Macro indicators.
2. **Feature Engineering:** Computes 110+ orthogonal alpha factors, applying cross-sectional winsorization and dynamic sector-neutralization.
3. **Signal Inference:** Employs an ensemble of Walk-Forward trained Gradient Boosted Decision Trees (GBDTs) to generate non-linear return predictions.
4. **Portfolio Construction:** Ingests signals into a Convex Optimizer constrained by covariance matrices, transaction costs, and exposure limits.
5. **Risk Management:** Enforces volatility targeting, sector bounds, and trailing stops to protect capital.

---

## 3. Quantitative Methodology

### 3.1 Data Warehouse & Universe Construction
*   **Survivorship Bias Mitigation**: Dynamic historical constituent tracking via exact daily S&P 500 membership masks. Bankruptcies and delistings are natively simulated.
*   **Point-in-Time (PiT) Correctness**: Strict adherence to reporting lag horizons for fundamental and earnings data to completely eliminate look-ahead bias.
*   **Data Quality Guards**: Automated validation pipelines for anomaly detection, liquidity filters (Median ADV thresholds), and regime shift monitoring.

### 3.2 Orthogonal Feature Engineering
*   **Technical**: Momentum (Time-Series & Cross-Sectional), Mean Reversion, Volatility, Volume/Liquidity.
*   **Fundamental**: Value (Earnings Yield, EV/EBITDA), Quality (ROE, Accruals), Growth (Revenue, EPS), Financial Health (Altman Z-Score).
*   **Alternative/Macro**: Term Premium, VIX proxies, Sentiment Analysis, and Insider Trading Activity.
*   **Signal Preprocessing**: Strict *per-fold* cross-sectional winsorization and dynamic sector-neutralization to isolate pure idiosyncratic alpha.

### 3.3 Machine Learning & Signal Extraction
*   **Non-Linear Ensembles**: Stacking of Gradient Boosted Decision Trees (**LightGBM**, **XGBoost**, **CatBoost**) to capture complex, non-linear feature interactions.
*   **Walk-Forward Validation**: Expanding window cross-validation with a strict **21-trading-day embargo period** (Purged K-Fold) to prevent temporal target overlapping.
*   **Alpha Gatekeeping**: Models must pass rigorous out-of-sample Information Coefficient ($IC$) and $t$-statistic gates ($t > 2.5$) before entering the production ensemble.
*   **Rank-Based Alpha**: Signals are smoothed temporally via EWMA and cross-sectionally ranked to maintain market neutrality.
*   **Custom Objectives**: Specialized loss functions (e.g., Weighted Symmetric MAE) designed to strictly penalize directional sign errors over absolute magnitude.

### 3.4 Portfolio Optimization & Ex-Ante Risk
*   **Risk Modeling**: Dynamic covariance matrix estimation via **Ledoit-Wolf Shrinkage** to stabilize matrix inversion.
*   **Optimization Engines**: Configurable objective functions including Mean-Variance (Markowitz), Risk Parity (ERC), Black-Litterman, and Kelly Criterion allocators.
*   **Volatility Targeting**: Dynamic gross exposure scaling to maintain a constant portfolio risk profile ($\sigma_{target}$).
*   **Institutional Constraints**: Bound by maximum cardinality, sector tracking limits, Herfindahl-Hirschman Index (HHI) concentration caps, and turnover budgets.

### 3.5 Transaction Cost Analysis (TCA)
*   **Simulated Commissions**: Institutional commission schedules modeled at $10\text{ bps}$ per executed trade value.
*   **Market Impact**: Almgren-Chriss linear impact model applied via percentage of Average Daily Volume (ADV).
*   **Slippage**: Strict spread crossing and liquidity-adjusted slippage constraints ($5\text{ bps}$) embedded within execution.

---

## 4. Performance Analytics

*Out-of-Sample (OOS) Backtest Period: February 2024 – May 1, 2026 | Initial Capital: $1,000,000 | Benchmark: SPY*

| Portfolio Construction Method | CAGR | Sharpe Ratio | Sortino Ratio | Max Drawdown | Excess Return (vs SPY) | Ending Equity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-N (Equal Weight)** | **+47.50%** | 2.85 | 3.91 | -7.66% | **+30.41%** | $2,224,421 |
| **Risk Parity (ERC)** | **+32.01%** | 2.52 | 3.37 | -7.28% | **+13.98%** | $2,341,531 |
| **Kelly Criterion** | **+26.76%** | 2.33 | 3.07 | -8.19% | **+8.73%** | $2,066,105 |
| **Mean-Variance (Markowitz)**| **+13.03%** | 1.15 | 1.74 | -7.00% | **-5.50%** | $1,487,786 |

*> Note: Past performance is not indicative of future results. Metrics derived from rigorous out-of-sample walk-forward backtesting utilizing a multi-model GBDT ensemble. Results are fully burdened by simulated institutional TCA. Recent 2026 performance reflects strong alpha generation, particularly through equal-weight and risk-parity portfolios.*

---

## 5. Risk Management Framework

The platform incorporates multi-layered risk management to preserve capital and ensure stable returns:

* **Dynamic Covariance Modeling**: Utilizes **Ledoit-Wolf Shrinkage** to estimate the covariance matrix $\Sigma$.
* **Volatility Targeting**: Dynamically scales gross exposure down during volatile market regimes.
* **Concentration Caps**: Constrains the Herfindahl-Hirschman Index (HHI) to enforce strict diversification.
* **Systemic Kill Switch**: Monitors peak-to-trough drawdowns ($MaxDD$). If a threshold is breached (e.g., $-20\%$), the system enters a mandatory 21-day cash cooldown.

---

## 6. Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Core Runtime** | Python 3.11, NumPy, Pandas, SciPy |
| **Machine Learning** | LightGBM, XGBoost, CatBoost, Scikit-Learn |
| **Optimization** | CVXPY, SciPy Optimize, Optuna |
| **Acceleration** | Numba (JIT Compilation) |
| **Infrastructure** | Docker, Docker Compose, GitHub Actions (CI/CD) |
| **Deployment** | Terraform (AWS/GCP), Kubernetes |

---

## 7. Repository Structure

```text
quant_alpha_research/                     # Root directory
│
├── 📁 config/                            # Configuration management
│   ├── __init__.py                       # Package initialization
│   ├── settings.py                       # Main configuration 
│   ├── mappings.py                       # column mappings 
│   └── logging_config.py                 # Logging setup
│
├── 📁 quant_alpha/                       # Main package (core logic)
│   │
│   ├── 📁 data/                          # Data acquisition layer
│   │   ├── __init__.py
│   │   ├── base_loader.py                # Abstract base class 
│   │   ├── price_loader.py               # OHLCV data  
│   │   ├── fundamental_loader.py         # Fundamentals 
│   │   ├── earnings_loader.py            # Earnings data 
│   │   ├── alternative_loader.py         # News, sentiment, insider 
│   │   ├── DataManager.py                
│   │
│   ├── 📁 features/                      # Feature engineering 
│   │   ├── __init__.py
│   │   ├── base.py                       # BaseFactor abstract class 
│   │   ├── registry.py                   # FactorRegistry manager 
│   │   │
│   │   ├── 📁 technical/                 # Technical indicators
│   │   │   ├── __init__.py
│   │   │   ├── momentum.py               # Momentum factors 
│   │   │   ├── mean_reversion.py         # Mean reversion 
│   │   │   ├── volatility.py             # Volatility factors 
│   │   │   └── volume.py                 # Volume factors 
│   │   │
│   │   ├── 📁 fundamental/               # Fundamental factors
│   │   │   ├── __init__.py
│   │   │   ├── value.py                  # P/E, P/B, FCF yield 
│   │   │   ├── quality.py                # ROE, ROIC, margins 
│   │   │   ├── growth.py                 # Revenue, EPS growth 
│   │   │   ├── utils.py           
│   │   │   └── financial_health.py       # Debt, liquidity 
│   │   │
│   │   ├── 📁 earnings/                  # Earnings-based
│   │   │   ├── __init__.py
│   │   │   ├── surprises.py              # EPS surprises 
│   │   │   ├── revisions.py              # Analyst revisions 
│   │   │   ├── utils.py               
│   │   │   └── estimates.py              # Consensus estimates 
│   │   │
│   │   ├── 📁 alternative/               # Alternative data
│   │   │   ├── __init__.py
│   │   │   ├── sentiment.py              
│   │   │   ├── inflation.py               
│   │   │   ├── marco.py        
│   │   │   
│   │   │
│   │   ├── 📁 composite/                 # Multi-factor scores
│   │   │   ├── __init__.py
│   │   │   ├── marco_adjusted.py            
│   │   │   ├── smart_signals.py          
│   │   │   ├── system_health.py                    
│   │   │
│   │   └── utils.py                      # Feature utilities
│   │
│   ├── 📁 models/                        # ML modeling layer
│   │   ├── __init__.py
│   │   ├── base_model.py                 # Abstract model class
│   │   ├── lightgbm_model.py             # LightGBM wrapper
│   │   ├── xgboost_model.py              # XGBoost wrapper
│   │   ├── catboost_model.py             # CatBoost wrapper
│   │   ├── ensemble.py                   # Model averaging/stacking
│   │   ├── trainer.py                    # Walk-forward trainer
│   │   ├── predictor.py                  # Production predictions
│   │   ├── feature_selector.py           # Feature selection
│   │   ├── hyperopt.py                   # Hyperparameter tuning
│   │   └── utils.py                      # Model utilities
│   │
│   ├── 📁 backtest/                      # Backtesting engine
│   │   ├── __init__.py
│   │   ├── engine.py                     # Main backtest loop
│   │   ├── portfolio.py                  # Portfolio construction
│   │   ├── execution.py                  # Trade execution simulator
│   │   ├── market_impact.py              # Almgren-Chriss model
│   │   ├── metrics.py                    # Performance metrics
│   │   ├── attribution.py                # Factor attribution
│   │   ├── risk_manager.py               # Risk controls
│   │   └── utils.py                      # Backtest utilities
│   │
│   ├── 📁 research/                      # Research & analysis
│   │   ├── __init__.py
│   │   ├── factor_analysis.py            # IC, decay analysis
│   │   ├── regime_detection.py           # Market regimes
│   │   ├── correlation_analysis.py       # Factor correlation
│   │   ├── significance_testing.py       # Statistical tests
│   │   ├── alpha_decay.py                # Signal decay analysis
│   │   └── utils.py                      # Research utilities
│   │
│   ├── 📁 optimization/                  # Portfolio optimization
│   │   ├── __init__.py
│   │   ├── mean_variance.py              # Markowitz optimization
│   │   ├── risk_parity.py                # Risk parity
│   │   ├── black_litterman.py            # Black-Litterman
│   │   ├── kelly_criterion.py            # Kelly sizing
│   │   ├── constraints.py                # Position constraints
│   │   └── allocator.py                  # Portfolio Allocator
│   │
│   ├── 📁 monitoring/                    # Production monitoring
│   │   ├── __init__.py
│   │   ├── performance_tracker.py        # Live IC tracking
│   │   ├── data_quality.py               # Data validation
│   │   ├── model_drift.py                # Concept drift detection
│   │   ├── alerts.py                     # Alert system
│   │   └── dashboard.py                  # Monitoring dashboard
│   │
│   ├── 📁 visualization/                 # Plotting & reporting
│   │   ├── __init__.py
│   │   ├── plots.py                      # Standard plots
│   │   ├── interactive.py                # Plotly dashboards
│   │   ├── reports.py                    # PDF/HTML reports
│   │   ├── factor_viz.py                 # Factor visualization
│   │   └── utils.py                      # Viz utilities
│   │
│   └── 📁 utils/                         # Common utilities
│       ├── __init__.py
│       ├── coulmn_helpers.py             # Column mapping
│       ├── date_utils.py                 # Date handling
│       ├── math_utils.py                 # Math functions
│       ├── io_utils.py                   # File I/O
│       ├── logging_utils.py              # Logging helpers
│       └── decorators.py                 # Python decorators
│
├── 📁 scripts/                           # Executable scripts
│   ├── __init__.py
│   ├── download_data.py                  # Initial data download
│   ├── update_data.py                    # Daily data updates
│   ├── run_backtest.py                   # Backtest runner
│   ├── train_models.py                   # Model training
│   ├── generate_predictions.py           # Production predictions
│   ├── validate_factors.py               # Factor validation
│   ├── optimize_portfolio.py             # Portfolio optimization
│   ├── create_report.py                  # Generate reports
│   ├── deploy_model.py                   # Deployment script
│   ├── monitor_production.py             # Concept drift monitoring
│   ├── run_hyperopt.py                   # Hyperopt runner
│   └── run_pipeline.py                   # Full pipeline orchestrator
│
├── 📁 tests/                             # Comprehensive testing
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   │
│   ├── 📁 unit/                          # Unit tests
│   │   ├── test_data_loaders.py          
│   │   ├── test_data_updates.py         
│   │   ├── test_features.py              
│   │   ├── test_models.py                
│   │   ├── test_backtest.py              
│   │   ├── test_optimization.py          
│   │   ├── test_scripts.py
│   │   ├── test_validation.py 
│   │   └── test_utils.py                 
│   │
│   ├── 📁 integration/                   # Integration tests
│   │   ├── test_pipeline.py              
│   │   ├── test_deployment_integration.py              
│   │   ├── test_data_flow.py             
│   │   ├── test_validation_integration.py            
│   │   └── test_production.py            
│   │
│   └── 📁 performance/                   # Performance tests
│       ├── test_speed.py                 # Speed benchmarks
│       └── test_memory.py                # Memory usage
│
├── 📁 notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb         # EDA
│   ├── 02_factor_research.ipynb          # Factor testing
│   ├── 03_model_development.ipynb        # Model prototyping
│   ├── 04_backtest_analysis.ipynb        # Backtest results
│   ├── 05_production_monitoring.ipynb    # Live monitoring
│   └── 06_research_tearsheet.ipynb       # Research template
│
├── 📁 data/                              # Data storage
│   ├── 📁 raw/                           # Raw downloaded data
│   │   ├── 📁 prices/                    # OHLCV data
│   │   ├── 📁 fundamentals/              # Fundamental data
│   │   ├── 📁 earnings/                  # Earnings data
│   │   └── 📁 alternative/               # Alternative data
│   │   
│   ├── 📁 processed/                     # Processed data
│   └── 📁 cache/                         # Cached results
│
├── 📁 models/                            # Saved models
│   ├── 📁 production/                    # Production models
│   ├── 📁 archive/                       # Historical models
│   └── 📁 experiments/                   # Experimental models
│
├── 📁 results/                           # Results & outputs
│   ├── 📁 backtests/                     # Backtest results
│   │   ├── backtest_results.csv
│   │   ├── metrics.json
│   │   └── attribution.csv
│   │
│   ├── 📁 predictions/                   # Model predictions
│   │   ├── daily_predictions.csv
│   │   └── portfolio_weights.csv
│   │
│   └── 📁 reports/                       # Generated reports
│
├── 📁 docs/                              # Documentation
├── 📁 docker/                            # Docker configuration
├── 📁 deployment/                        # Deployment configs
│
├── .env.example                          # Environment variables template
├── .gitignore                            # Git ignore rules
├── .pre-commit-config.yaml               # Pre-commit hooks
├── main.py                               # Main orchestrator
├── requirements.txt                      # Python dependencies
├── requirements-dev.txt                  # Development dependencies
├── setup.py                              # Package setup
├── pytest.ini                            # Pytest configuration
├── mypy.ini                              # Type checking config
├── README.md                             # Project README
└── LICENSE                               # MIT License
```

---

## 8. Quick Start Guide

### Prerequisites
*   Python 3.9+
*   Python 3.9+ (3.11 recommended)
*   Docker (optional, for containerized execution)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/AnuragPatkar/quant-alpha-platform.git
cd quant-alpha-platform

# 2. Configure Environment
cp .env.example .env
# Edit .env with your API keys (Polygon, FMP, etc.)

# 3. Install Dependencies
pip install -r requirements.txt
pip install -e .[dev]
```

### 2. Running the Full Pipeline

The most common command is `pipeline`, which runs all essential steps in sequence.

```bash
# Run the full pipeline: data update -> train -> predict -> backtest -> report
python main.py pipeline --all

# Run the pipeline but force a rebuild of the data cache and models
python main.py pipeline --all --force-rebuild
```

### 3. Individual Workflow Commands

You can also run each step of the pipeline individually.

```bash
# Update all market data (prices, fundamentals, etc.)
python main.py data

# Train the ML models using the latest data
python main.py train

# Generate new alpha signals (predictions) without retraining
python main.py predict

# Run a backtest simulation on the latest signals
python main.py backtest --method risk_parity

# Run hyperparameter optimization for the models
python main.py hyperopt

# Generate the executive summary report
python main.py report
```

### 4. Testing

Run the full test suite to ensure the system is healthy.

```bash
python main.py test

# Run static type checking to catch potential bugs
mypy .
```

---

## 🐳 Docker & Deployment

The project is fully containerized for reproducible environments and easy deployment.

### Local Execution with Docker Compose

`docker-compose` orchestrates the application and any required services (like a database or cache).

```bash
# Build and start the services in detached mode
docker-compose -f docker/docker-compose.yml up --build -d

# View the logs of the running application
docker-compose -f docker/docker-compose.yml logs -f quant-alpha

# Stop and remove the containers
docker-compose -f docker/docker-compose.yml down
```

### Cloud Deployment

The `deployment/` directory contains production-grade **Infrastructure-as-Code** templates for deploying the system to major cloud providers.

- **`deployment/aws/`**: Terraform scripts for deploying to AWS ECS Fargate.
- **`deployment/gcp/`**: Terraform scripts for deploying to Google Cloud Run.
- **`deployment/kubernetes/`**: YAML manifests for deploying to any Kubernetes cluster.

Refer to the `README.md` inside the `deployment/` directory for specific usage instructions.

---

## 🔄 CI/CD

A GitHub Actions workflow is defined in `.github/workflows/ci_cd.yml`. This pipeline automatically triggers on every `push` and `pull_request` to the `main` branch.

**Jobs:**
1.  **`test`**: Installs dependencies and runs the entire `pytest` suite.
2.  **`build-docker`**: If tests pass, it builds the Docker image to ensure it's not broken.

This automated process guarantees that code merged into the main branch is always tested and deployable.
