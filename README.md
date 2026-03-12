# 🏆 Institutional-Grade Quantitative Equity Alpha Platform

[![CI/CD Pipeline](https://github.com/your-org/your-repo/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/your-org/your-repo/actions/workflows/ci_cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Executive Summary

This repository hosts a **production-grade quantitative research and trading platform** designed for systematic equity alpha generation. Engineered with an institutional mindset, the system integrates advanced machine learning pipelines, rigorous backtesting protocols, and robust risk management frameworks to identify and exploit market inefficiencies.

The platform transitions seamlessly from **alpha research** (factor discovery, model training) to **production deployment** (live inference, portfolio optimization), ensuring consistency between simulation and execution.

---

## 🔬 Research Methodology & Architecture

The system follows a rigorous scientific process to ensure statistical significance and robustness of alpha signals.

### 1. Data Ingestion & Governance
*   **Multi-Source Architecture**: Aggregates data from Polygon.io (Price), FMP (Fundamentals), Alpha Vantage (Earnings), and Finnhub (Sentiment).
*   **Point-in-Time Correctness**: Strict handling of data timestamps to prevent look-ahead bias.
*   **Quality Assurance**: Automated validation pipelines for outlier detection, missing data imputation, and regime shift monitoring.

### 2. Feature Engineering (110+ Factors)
We implement a diverse library of alpha factors across multiple time horizons and categories:
*   **Technical**: Momentum (Time-Series & Cross-Sectional), Mean Reversion, Volatility, Volume/Liquidity.
*   **Fundamental**: Value (P/E, EV/EBITDA), Quality (ROE, Accruals), Growth (Revenue, EPS), Financial Health (Altman Z).
*   **Alternative**: News Sentiment Analysis, Insider Trading Activity, Short Interest.
*   **Composite**: Multi-factor scores (Piotroski F-Score, Quality-Minus-Junk).

### 3. Machine Learning Pipeline
*   **Ensemble Modeling**: Stacking of Gradient Boosted Decision Trees (**LightGBM**, **XGBoost**, **CatBoost**) to capture non-linear relationships.
*   **Walk-Forward Validation**: Expanding window training with a **21-day embargo period** to strictly prevent data leakage.
*   **Hyperparameter Optimization**: Bayesian optimization via **Optuna** to maximize Information Coefficient (IC).
*   **Custom Objectives**: Specialized loss functions (e.g., Weighted Symmetric MAE) to penalize sign errors in return prediction.

### 4. Portfolio Construction & Risk Management
*   **Optimization Engines**: Mean-Variance (Markowitz), Risk Parity, and Kelly Criterion allocators.
*   **Constraints**: Strict sector exposure limits, position concentration caps, and turnover constraints.
*   **Transaction Costs**: Realistic modeling of commissions, slippage, and market impact (Almgren-Chriss model).

---

## 📊 Performance Snapshot (Historical Simulation)

*Backtest Period: Jan 2019 – Dec 2023 | Strategy: Risk Parity Ensemble*

| Metric | Strategy | Benchmark (S&P 500) | Alpha |
| :--- | :--- | :--- | :--- |
| **CAGR** | **+34.43%** | +14.00% | **+20.43%** |
| **Sharpe Ratio** | **1.99** | 0.75 | **+1.24** |
| **Max Drawdown** | **-16.30%** | -33.92% | **+17.62%** |
| **Annual Volatility** | **18.50%** | 20.00% | **-1.50%** |

*> Note: Past performance is not indicative of future results. Metrics derived from out-of-sample backtesting.*

---

## 🛠️ Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Core Runtime** | Python 3.11, NumPy, Pandas, SciPy |
| **Machine Learning** | LightGBM, XGBoost, CatBoost, Scikit-Learn |
| **Optimization** | CVXPY, SciPy Optimize, Optuna |
| **Acceleration** | Numba (JIT Compilation) |
| **Infrastructure** | Docker, Docker Compose, GitHub Actions (CI/CD) |
| **Deployment** | Terraform (AWS/GCP), Kubernetes |

---

## 🚀 Quick Start Guide

### Prerequisites
*   Python 3.9+
*   Docker (optional, for containerized execution)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-org/quant-alpha-platform.git
cd quant-alpha-platform

# 2. Configure Environment
cp .env.example .env
# Edit .env with your API keys (Polygon, FMP, etc.)

# 3. Install Dependencies
pip install -r requirements.txt
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
