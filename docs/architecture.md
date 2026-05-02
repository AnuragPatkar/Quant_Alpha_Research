# Architecture & System Design

> **Purpose**: This document describes the high-level architecture, module responsibilities, data flows, and design patterns used in the Quant Alpha Research Platform.

---

## 1. Architecture Overview

The Quant Alpha platform is designed as a **Directed Acyclic Graph (DAG)** of independent, composable modules that collectively transform raw market data into optimal portfolio weights. The architecture enforces a strict separation of concerns, enabling researchers to iterate on features and models independently while guaranteeing reproducibility and operational parity between research and production environments.

### High-Level Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION LAYER                       │
│  (CSV → Parquet Cache, Point-in-Time Validation, Survivorship Bias) │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ├─ Price Loader (OHLCV)
             ├─ Fundamental Loader (10-K/10-Q quarterly snapshots)
             └─ Alternative Loader (Macro, VIX, sentiment)
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA MANAGER (ORCHESTRATION)                   │
│     • Caches raw CSVs as Parquet for fast access                   │
│     • Merges price + fundamentals via asynchronous left-joins      │
│     • Returns unified (date, ticker) MultiIndex DataFrame           │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                         │
│  (Registry Pattern + Parallel Computation of 110+ Orthogonal Factors)│
│  • Technical: Momentum, Mean Reversion, Volatility, Volume         │
│  • Fundamental: Value, Quality, Growth, Financial Health           │
│  • Earnings: Surprises, Revisions, Estimates                       │
│  • Alternative: Macro, Sentiment, Insider Activity                 │
│  • Composite: Multi-factor aggregations (sigma, macro_adjusted)    │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING LAYER                           │
│  • Walk-Forward Trainer: Expanding-window CV with 21-day embargo   │
│  • Base Models: XGBoost, LightGBM, CatBoost with custom losses     │
│  • Ensemble: Rank-averaging of GBDT predictions                    │
│  • Feature Selection: Automatic selection of predictive signals    │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    SIGNAL PROCESSING & RANKING                      │
│  • Rank-based (cross-sectional) signal normalization               │
│  • Temporal smoothing via Exponential Weighted Moving Average      │
│  • Market-neutrality enforcement                                   │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  PORTFOLIO OPTIMIZATION LAYER                       │
│  • Mean-Variance (Markowitz): Risk minimization                    │
│  • Kelly Criterion: Geometric growth maximization                  │
│  • Risk Parity (ERC): Equal risk contribution                      │
│  • Volatility Targeting: Dynamic gross exposure scaling            │
│  • Constraints: Cardinality, HHI concentration, sector limits      │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   BACKTESTING ENGINE (SIMULATION)                   │
│  • Daily rebalancing with trade-by-trade execution simulation      │
│  • Transaction Cost Analysis: Commission, spread, slippage, impact │
│  • Portfolio State Management: Cash, positions, realized P&L       │
│  • Risk Controls: Trailing stops, drawdown limits, concentration   │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  RESEARCH & VALIDATION LAYER                        │
│  • Factor Analysis: IC (Information Coefficient), decays            │
│  • Model Drift Detection: Concept drift, prediction bias, label shift
│  • Attribution Analysis: Factor contribution to returns            │
│  • Performance Metrics: Sharpe, Sortino, Max DD, CAGR              │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       RESULTS & MONITORING                          │
│  • Equity curves, trade logs, performance dashboards               │
│  • Drift alerts, data quality checks                               │
│  • Production model archival and versioning                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities & Design Patterns

### 2.1 Data Layer (`quant_alpha/data/`)

**Abstract Pattern**: Template Method + Strategy Pattern

| Module | Responsibility | Input | Output |
|--------|----------------|-------|--------|
| `BaseLoader` | Abstract interface for all data sources | Raw CSV | Processed pd.DataFrame |
| `PriceLoader` | OHLCV daily price data | S&P 500 price CSVs | (date, ticker, OHLCV) |
| `FundamentalLoader` | Quarterly balance sheet, cash flow, income statement | SimFin CSVs | Point-in-time fundamental metrics |
| `EarningsLoader` | Earnings surprise, revisions, consensus estimates | Event-driven CSVs | Earnings event data |
| `AlternativeLoader` | Macro indicators, VIX, sentiment, insider | External APIs/CSVs | Alternative alpha signals |
| `DataManager` | Orchestration layer combining all sources | Raw loader outputs | Unified (date, ticker) MultiIndex DataFrame |

**Key Design Principles**:
- **Cache-first**: All loaders serialize to Parquet for fast reloads
- **Point-in-Time**: Fundamental data includes reporting lag (90 days) to prevent look-ahead bias
- **Survivorship**: Dynamic S&P 500 membership masks exclude delisted/bankrupt companies
- **Quality guards**: NaN validation, volume filtering, data range checks

---

### 2.2 Feature Engineering (`quant_alpha/features/`)

**Abstract Pattern**: Registry Pattern + Template Method + Factory

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| `BaseFactor` | Abstract lifecycle: validate → compute → align → sanitize → normalize | Abstract |
| `FactorRegistry` | Singleton registry; manages factor discovery via `@register()` decorator | Singleton |
| `technical/*` | Price-based signals: momentum, mean reversion, volatility, volume | 15+ implementations |
| `fundamental/*` | Balance sheet, profitability, growth, financial health | 20+ implementations |
| `earnings/*` | EPS surprises, revisions, consensus estimate beats | 8+ implementations |
| `alternative/*` | Macro indicators, sentiment, insider trading | 10+ implementations |
| `composite/*` | Multi-factor aggregations (sigma, macro-adjusted, smart signals) | 5+ implementations |

**Computation Flow**:
1. **Registry instantiation**: Factory pattern creates all registered factors
2. **Parallel execution**: ThreadPoolExecutor with max_workers=4 computes factors in parallel
3. **Validation**: Each factor validates input schema, handles edge cases
4. **Sanitization**: NaN filling, winsorization, cross-sectional normalization
5. **Merging**: Union of all factor columns into a wide matrix (date, ticker, 110+ features)

**Key Constraints**:
- `lookback_period`: Historical window required (e.g., 252 days for momentum)
- `normalize`: Boolean flag for cross-sectional Z-score normalization
- `winsorize`: Trim extreme outliers (default 5th/95th percentile)

---

### 2.3 Machine Learning (`quant_alpha/models/`)

**Abstract Pattern**: Abstract Base Class + Strategy + Trainer

| Module | Responsibility | Algorithm |
|--------|----------------|-----------|
| `BaseModel` | Abstract fit/predict contract; feature validation | Abstract |
| `XGBoostModel` | Gradient boosting via SKLearn XGBRegressor wrapper | Tree ensemble |
| `LightGBMModel` | Fast GBDT with custom loss functions (weighted symmetric MAE) | Tree ensemble |
| `CatBoostModel` | Categorical feature support; native GPU acceleration | Tree ensemble |
| `FeatureSelector` | Automatic feature selection (permutation importance, mutual info) | Filtering/RFE |
| `WalkForwardTrainer` | Expanding-window cross-validation with embargo period | Temporal CV |
| `EnsembleModel` | Rank-averaging of multiple GBDT base models | Meta-learner |
| `Predictor` | Production inference pipeline; handles batch predictions, caching | Serving |

**Walk-Forward Training Parameters** (Institutional Standard):
- `min_train_months: 36` — Minimum 3 years of training data
- `test_months: 6` — Out-of-sample test window
- `step_months: 3` — Quarterly retraining cadence
- `embargo_days: 21` — Prevent look-ahead bias (21 trading days)
- `window_type: 'expanding'` — Growing training set (not rolling)
- `purged_k_fold: True` — Purged K-fold for overlapping predictions

**Model Output**:
- Per-fold out-of-sample predictions (predictions_oos.csv)
- Per-fold trained models (pickled checkpoints)
- Performance metrics per fold (sharpe, ic, etc.)
- Feature importance rankings

---

### 2.4 Backtesting Engine (`quant_alpha/backtest/`)

**Abstract Pattern**: Strategy Pattern + State Machine

| Module | Responsibility | Input | Output |
|--------|----------------|-------|--------|
| `BacktestEngine` | Main simulation loop; daily execution orchestration | signals, prices, parameters | equity_curve, trade_log, metrics |
| `Portfolio` | State machine tracking positions, cash, P&L | orders | updated portfolio state |
| `ExecutionSimulator` | Trade execution with realistic friction costs | order, market state | fill_price |
| `MarketImpact` | Almgren-Chriss linear impact model | order_size, adv, volatility | impact_cost |
| `BacktestMetrics` | Performance analytics (Sharpe, DD, CAGR, etc.) | equity_curve | metrics_dict |
| `RiskManager` | Enforce position limits, concentration caps, trailing stops | portfolio | alerts |
| `FactorAttribution` | Decompose returns by factor exposure | signals, returns | attribution |

**Key Backtesting Parameters** (Fully Burdened):
- `commission_rate: 10 bps` — Institutional commissions
- `spread_bps: 5` — Bid-ask crossing costs
- `slippage_bps: 2` — Execution slippage
- `max_position: 10%` — Concentration limit per stock
- `max_adv_participation: 2%` — Liquidity constraint
- `rebalance_frequency: 'daily'` — Daily execution
- `target_volatility: 15%` — Volatility targeting sigma
- `max_drawdown_trigger: -20%` — System kill-switch threshold

**Daily Simulation Steps** (Pseudo-code):
```
For each trading date:
    1. Fetch model predictions (signals) for universe
    2. Generate target portfolio weights (optimize or rank-based)
    3. Compute position changes (rebalance deltas)
    4. Execute trades with cost simulation (commission, impact, slippage)
    5. Update portfolio state (positions, cash, cost basis)
    6. Mark-to-market all holdings (close prices)
    7. Compute daily metrics (return, volatility, sharpe YTD)
    8. Check risk controls (concentration, drawdown, stops)
    9. Log all transactions and state updates
```

---

### 2.5 Portfolio Optimization (`quant_alpha/optimization/`)

**Abstract Pattern**: Strategy Pattern with Solver Cascading

| Module | Objective Function | Constraint Set | Solver |
|--------|-------------------|-----------------|--------|
| `MeanVarianceOptimizer` | Minimize $\lambda w^T \Sigma w - w^T \mu$ | Box, leverage, HHI | OSQP→ECOS→SCS→CLARABEL |
| `KellyCriterion` | Maximize $E[\ln(1+r)]$ (geometric growth) | Leverage, bounds | QP / closed-form heuristic |
| `RiskParityOptimizer` | Minimize $\frac{1}{2} y^T \Sigma y - \sum_i b_i \ln(y_i)$ | Risk budgets | L-BFGS-B |
| `BlackLittermanOptimizer` | Bayesian blending of priors and views | Expert views | Closed-form Bayes |
| `Allocator` | Unified facade for all optimizers | Method-agnostic | Dispatches to above |

**Institutional Constraints** (enforced):
- **Position limits**: `[0, 0.10]` per stock (maximum 10% concentration)
- **Sector bounds**: ±5% relative to index weighting
- **Gross exposure**: ≤ 150% (1.5x leverage cap)
- **HHI concentration**: ≤ 0.05 (Hirschman-Herfindahl Index)
- **Turnover**: ≤ 20% per rebalance date
- **Cardinality**: 20 to 80 active positions

---

### 2.6 Data Validation & Preprocessing (`quant_alpha/preprocessing/`)

**Pipelines**: ETL + Data Quality Gates

| Module | Responsibility | Processing Steps |
|--------|----------------|------------------|
| `FundamentalPreprocessor` | Standardize financial metrics from multiple sources | Extract → Align PiT → Forward-fill → Bind to daily |
| `PreprocessingIntegration` | Merge fundamentals with daily prices via asynchronous joins | Merge-asof (backward-looking temporal join) |
| `DataValidator` | Schema, nullness, distributional validation | Type checking → NaN bounds → Outlier detection |

**Point-in-Time Alignment** (Critical for removing look-ahead bias):
- Fundamental data includes 90-day reporting lag
- Earnings events aligned to announcement dates
- Quarterly metrics forward-filled to daily (stale until next quarter)
- Prevents using information not available at feature computation date

---

### 2.7 Research & Monitoring (`quant_alpha/research/`, `quant_alpha/monitoring/`)

| Module | Responsibility | Key Metrics |
|--------|----------------|-------------|
| `FactorAnalyzer` | Alpha validation, signal efficacy testing | IC (correlation), quantile spreads, autocorr, monotonicity |
| `FactorAlphaDecay` | Forward-looking decay curves showing signal half-life | IC decay over 1-20 day horizons |
| `CorrelationAnalyzer` | Pairwise factor correlations, orthogonality testing | Rolling correlation matrices |
| `RegimeDetector` | Market regime identification (bull/bear/sideways) | Hidden Markov Model or threshold-based |
| `SignificanceTester` | Hypothesis testing for factor power | t-stats, p-values, effect sizes |
| `DataQualityMonitor` | First-line data validation and covariate drift detection | PSI, schema checks, anomaly flags |
| `ModelDriftDetector` | Concept drift detection (out-of-sample degradation) | Rolling MSE, prediction bias, label shift |
| `PerformanceTracker` | Live model performance monitoring | Predictions vs. actuals, rolling aggregates |

---

## 3. Data Warehouse Schema & Interfaces

### 3.1 Price Data Format

```
Date         | Ticker | Open   | High   | Low    | Close  | Volume
2024-02-01   | AAPL   | 189.50 | 191.00 | 188.75 | 190.25 | 52M
2024-02-02   | AAPL   | 191.00 | 192.50 | 190.00 | 192.00 | 48M
...
```

**Storage**: Parquet partitioned by (date, ticker)  
**Index**: MultiIndex(date, ticker)  
**Frequency**: Daily (trading days only)

### 3.2 Fundamental Data Format (Post-Alignment)

```
Date       | Ticker | EPS | P/E | ROE | Total_Assets | FCF | ...
2024-02-01 | AAPL   | 6.0 | 28  | 85% | 350B         | 98B | ...
2024-02-02 | AAPL   | 6.0 | 28  | 85% | 350B         | 98B | ... (forward-filled)
...
```

**Alignment**: Backward-filled quarterly snapshots to daily  
**Reporting lag**: 90 days (data from quarter Q-1 appears on day Q+90)

### 3.3 Signal/Prediction Format

```
Date       | Ticker | ModelPred | Rank | Weight
2024-02-01 | AAPL   | 0.0234    | 2    | 0.0812
2024-02-01 | MSFT   | 0.0189    | 5    | 0.0623
2024-02-01 | GOOG   | -0.0045   | 245  | 0.0001
...
```

**Index**: MultiIndex(date, ticker) matching price data  
**Columns**: Raw predictions + normalized rank + optimized weight

---

## 4. Critical Design Constraints

### 4.1 Look-Ahead Bias Prevention

**Violation Detection Mechanisms**:
1. **Reporting lags**: Fundamental data includes 90-day embargo
2. **Embargo periods**: Walk-forward CV enforces 21-day separation between training and testing
3. **Point-in-Time enforcement**: Merge-asof uses backward-looking only joins
4. **Audit trails**: All data transformations logged with timestamps

### 4.2 Survivorship Bias Mitigation

**S&P 500 Constituent Tracking**:
- Dynamic daily membership mask (file: `data/processed/sp500_membership_mask.pkl`)
- Reconstructs historical S&P 500 exactly as it existed on each date
- Includes delisted, bankrupt, and de-indexed companies in historical analysis
- Prevents overstating returns by excluding poor performers that dropped out

### 4.3 Model Overfitting Prevention

**Multi-Layer Defense**:
1. **Expanding-window CV**: Growing training set (not rolling) to use all available history
2. **Embargo periods**: Purged K-fold with no temporal overlap between fold training and test sets
3. **Out-of-sample testing**: Public division into train/test (no validation set leakage)
4. **Alpha gates**: Models must pass $IC > 0.01$ and $t\text{-stat} > 2.5$ before production
5. **Rank normalization**: Removes scale dependency and reduces overfitting to price regimes

---

## 5. Execution Flow Examples

### 5.1 Full Pipeline Execution (Data → Inference → Backtest)

```python
# /scripts/run_pipeline.py orchestrates this DAG:

1. Download Data
   └─› update_data.py (fetch latest OHLCV, fundamentals, earnings)

2. Validate Data
   └─› diagnose_data.py (schema checks, NaN distribution, anomalies)

3. Feature Engineering
   └─› FactorRegistry.compute_all() (parallel computation of 110+ factors)

4. Train Models
   └─› train_models.py (walk-forward GBDT ensemble training)

5. Generate Predictions
   └─› generate_predictions.py (out-of-sample alpha signals)

6. Backtest Signals
   └─› run_backtest.py (daily simulation with TCA)

7. Optimize Portfolios
   └─› optimize_portfolio.py (mean-variance / Kelly / Risk Parity allocation)

8. Monitor & Report
   └─› create_report.py (equity curves, performance dashboards)
```

### 5.2 Research Workflow (Factor Development Cycle)

```python
# /notebooks/01_data_exploration.ipynb → 06_research_tearsheet.ipynb

1. Load Data
   └─› DataManager.get_data() returns (date, ticker) matrix

2. Engineer Features
   └─› FactorRegistry.register() new factor class
   └─› Compute and validate IC (Information Coefficient)

3. Analyze Decays
   └─› FactorAnalyzer.compute_quantile_returns() (long-short spreads)
   └─› FactorAnalyzer.compute_alpha_decay() (signal half-life)

4. Model Development
   └─› WalkForwardTrainer.train() → out-of-sample predictions
   └─› Feature importance rankings

5. Portfolio Construction
   └─› Allocator.optimize() (multiple methods)
   └─› Compare MV vs. Kelly vs. Risk Parity outputs

6. Backtest & Report
   └─› BacktestEngine.run() → FactorAttribution → metrics
```

---

## 6. Key Dependencies (Technology Stack)

| Domain | Libraries | Purpose |
|--------|-----------|---------|
| **Numerical Computing** | NumPy, SciPy | Linear algebra, optimization |
| **Data Processing** | Pandas, Polars | Tabular data manipulation |
| **Machine Learning** | Scikit-Learn | Feature scaling, preprocessing |
| **Gradient Boosting** | LightGBM, XGBoost, CatBoost | Ensemble GBDT models |
| **Optimization** | CVXPY, SciPy.optimize | Convex optimization, portfolio construction |
| **Hyperparameter Tuning** | Optuna | Bayesian optimization |
| **Acceleration** | Numba | JIT compilation for hot loops |
| **Visualization** | Matplotlib, Seaborn, Plotly | Charts and dashboards |
| **Deployment** | Docker, Kubernetes | Containerization and orchestration |

---

## 7. Design Philosophy & Principles

### 7.1 Separation of Concerns
Each module has a **single responsibility** (SOLID):
- **Data layer**: Raw ingestion only (no feature computation)
- **Feature layer**: Feature computation only (no model logic)
- **Model layer**: Model training only (no backtesting logic)
- **Backtest layer**: Simulation only (no optimization logic)

### 7.2 Composability & Modularity
- Modules are **loosely coupled** and interface via data frames
- Researchers can swap feature sets, models, or optimization methods independently
- Test coverage ensures each module's contract is maintained

### 7.3 Reproducibility & Determinism
- All random seeds fixed at entry points
- Configuration singleton ensures identical hyperparameters across runs
- Data loading cached in Parquet format for fast, identical reloads
- Walk-forward CV strictly ordered; no random shuffling

### 7.4 Institutional Grade
- **TCA-burdened**: All costs (commission, spread, slippage, impact) explicitly modeled
- **Risk management**: Position limits, concentration caps, trailing stops enforced
- **Monitoring**: Continuous drift detection, data quality checks, alert systems
- **Auditability**: Full execution trails logged with timestamps

---

For more details on specific modules, see:
- [Setup & Installation](setup.md)
- [Usage Guide](usage.md)
- [Data Schemas & Sources](data.md)
- [Contributing Guidelines](contributing.md)
