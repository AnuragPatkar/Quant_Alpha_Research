# Setup & Installation Guide

> **Purpose**: Step-by-step instructions for setting up the development environment, installing dependencies, and validating a working installation.

---

## 1. Prerequisites

### 1.1 System Requirements

| Component | Requirement | Note |
|-----------|-------------|------|
| **OS** | Linux, macOS, or Windows (WSL2 recommended for Windows) | Tested on Ubuntu 22.04, M1 Mac, Windows 11 WSL2 |
| **Python** | ≥ 3.9, recommend 3.11 | Contains type hints and modern async features |
| **RAM** | ≥ 16 GB | 32 GB+ recommended for full S&P 500 backtests |
| **Disk** | ≥ 50 GB | Raw data + processed cache + model artifacts |
| **CPU** | Multi-core (4+ cores recommended) | Parallel factor computation and model training |

### 1.2 Optional Tools

| Tool | Purpose |
|------|---------|
| **Git** | Version control for code and config |
| **Docker** | Containerized execution and deployment |
| **Docker Compose** | Multi-container orchestration |
| **VSCode** | IDE with Python extension recommended |
| **Jupyter** | Interactive notebook research |

---

## 2. Installation Steps

### 2.1 Clone the Repository

```bash
# Clone the repository
git clone https://github.com/AnuragPatkar/quant_alpha_research.git
cd quant_alpha_research

# (Optional) Create a feature branch for your work
git checkout -b feature/my-factor-research
```

### 2.2 Create a Python Virtual Environment

**Option A: venv (Built-in)**

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

**Option B: conda (Recommended for data science)**

```bash
# Create environment from specification
conda env create -n quant-alpha python=3.11

# Activate environment
conda activate quant-alpha
```

### 2.3 Install Dependencies

```bash
# Install the package in editable mode with development dependencies
pip install -e .[dev]

# Verify installation
python -c "import quant_alpha; print('✓ Package installed successfully')"
```

**This installs**:
- Core dependencies: NumPy, Pandas, SciPy, Scikit-Learn, LightGBM, XGBoost, CatBoost
- Optimization: CVXPY, Optuna
- Acceleration: Numba
- Development: pytest, mypy, black, flake8

### 2.4 Configure Environment Variables

Create a `.env` file in the repository root:

```bash
# Copy template
cp .env.example .env  # Or create new file with contents below

# Edit .env with your settings
nano .env
```

**Minimum `.env` template**:

```ini
# Environment
ENV=development
LOG_LEVEL=INFO

# Data paths (auto-resolved from config/settings.py, optional override)
DATA_DIR=./data
MODELS_DIR=./models
RESULTS_DIR=./results

# Feature Engineering
ENABLE_FUNDAMENTAL_DATA=true
ENABLE_EARNINGS_DATA=true
ENABLE_ALTERNATIVE_DATA=true

# Model Training
N_JOBS=4
RANDOM_SEED=42

# Backtesting
INITIAL_CAPITAL=1000000
COMMISSION_BPS=10
TARGET_VOLATILITY=0.15

# Monitoring
ENABLE_DRIFT_DETECTION=true
DRIFT_WINDOW_DAYS=30
```

> **Note**: Most paths are auto-resolved by `config/settings.py`. Only set overrides if using non-standard directory layouts.

### 2.5 Validate Installation

Run the validation suite:

```bash
# 1. Test package imports
python -c "
from quant_alpha.data import DataManager
from quant_alpha.features import FactorRegistry
from quant_alpha.models import XGBoostModel
from quant_alpha.backtest import BacktestEngine
print('✓ All imports successful')
"

# 2. Run unit tests
pytest tests/unit/ -v --tb=short

# 3. Run integration tests (slower, requires sample data)
pytest tests/integration/ -v --tb=short

# 4. Type checking (mypy)
mypy quant_alpha/ --config-file mypy.ini
```

**Expected Output**:
```
✓ All imports successful
========================= 10 passed in 2.34s =========================
```

---

## 3. Data Setup

### 3.1 Directory Structure

The pipeline expects the following structure:

```
quant_alpha_research/
├── data/
│   ├── cache/                    # Auto-generated Parquet caches
│   ├── raw/
│   │   ├── sp500_prices/        # OHLCV CSV files (from yfinance)
│   │   ├── fundamentals/        # SimFin fundamental CSVs
│   │   ├── earnings/            # Earnings event CSVs
│   │   └── alternative/         # Macro + VIX CSVs
│   └── processed/
│       └── sp500_membership_mask.pkl  # Survivorship bias correction
├── models/
│   ├── production/              # Current production models
│   └── archive/                 # Historical model versions
├── results/
│   ├── predictions/             # Generated alpha signals
│   ├── backtests/               # Backtest equity curves
│   ├── reports/                 # Performance reports
│   └── validation/              # Factor validation results
└── logs/                        # Execution logs
```

**Create required directories**:

```bash
python -c "
from pathlib import Path
from config.settings import config

# Auto-created by config.settings
print(f'Data directory: {config.DATA_DIR}')
print(f'Models directory: {config.MODELS_DIR}')
print(f'Results directory: {config.RESULTS_DIR}')
"
```

### 3.2 Download Initial Data

**Option A: Automated Download**

```bash
# Download S&P 500 constituents and price data
python scripts/download_data.py --universe sp500 --start-date 2020-01-01

# Download fundamental and earnings data
python scripts/download_data.py --universe sp500 --fundamentals --earnings --start-date 2020-01-01
```

> **Note**: First download may take 30-60 minutes depending on internet speed and API rate limits.

**Option B: Manual CSV Upload**

If you have pre-existing CSV files:

1. **Price data**: Place OHLCV CSVs in `data/raw/sp500_prices/{TICKER}.csv`
   - Expected columns: `Date, Open, High, Low, Close, Volume`

2. **Fundamentals**: Place SimFin exports in `data/raw/fundamentals/`
   - Files: `balance_sheet.csv, income_statement.csv, cashflow_statement.csv`

3. **S&P 500 membership**: Create `data/processed/sp500_membership_mask.pkl` (see [create_membership_mask.py](../scripts/create_membership_mask.py))

### 3.3 Create S&P 500 Membership Mask

This corrects for survivorship bias by tracking historical S&P 500 constituents:

```bash
python scripts/create_membership_mask.py \
  --constituents-csv "S&P 500 Historical Components & Changes(01-17-2026).csv" \
  --output data/processed/sp500_membership_mask.pkl
```

**Expected Output**:
```
Created membership mask for 497 constituents
Date range: 2010-01-01 to 2026-01-17
Saved to: data/processed/sp500_membership_mask.pkl
```

---

## 4. Docker Setup (Optional)

For containerized execution (recommended for production/deployment):

### 4.1 Build Docker Image

```bash
# Build image
docker build -f docker/Dockerfile -t quant-alpha:latest .

# Verify image
docker images | grep quant-alpha
```

### 4.2 Run Container with Docker Compose

```bash
# Start container
docker-compose -f docker/docker-compose.yml up -d

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f quant-alpha

# Stop container
docker-compose -f docker/docker-compose.yml down
```

### 4.3 Execute Commands in Container

```bash
# Run pipeline
docker-compose -f docker/docker-compose.yml exec quant-alpha \
  python main.py pipeline --all

# Run backtest
docker-compose -f docker/docker-compose.yml exec quant-alpha \
  python scripts/run_backtest.py --start-date 2024-01-01
```

---

## 5. Development Environment Setup

### 5.1 Code Quality Tools

**Install pre-commit hooks** (auto-format and lint on commit):

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks (optional)
pre-commit run --all-files
```

### 5.2 IDE Configuration (VSCode)

**Recommended extensions**:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "ms-python.debugpy"
  ]
}
```

**Settings** (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### 5.3 Jupyter Notebook Setup

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Launch Jupyter Lab
jupyter lab

# Navigate to notebooks/ directory
# Start with 01_data_exploration.ipynb
```

---

## 6. Troubleshooting

### 6.1 Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `ModuleNotFoundError: No module named 'quant_alpha'` | Package not installed | Run `pip install -e .` from repo root |
| `ImportError: libomp.dylib not found` (Mac) | OpenMP not linked | `brew install libomp` or use conda |
| `CUDA/GPU not detected` | Numba can't find GPU | Keep CPU mode; GPU optional and non-critical |
| `MemoryError` during backtest | Insufficient RAM | Reduce universe size or use rolling windows |
| `yfinance API rate limited` | Too many simultaneous requests | Reduce `n_jobs` or add delays between requests |

### 6.2 Validate GPU Acceleration (Optional)

```python
# Check if Numba can use GPU
python -c "
import numba
print(f'CUDA available: {numba.cuda.is_available()}')
print(f'GPUs detected: {numba.cuda.cudadrv.devicearray.DeviceNDArray.__doc__}')
"
```

> **Note**: GPU acceleration is optional. All algorithms run correctly on CPU, just slower.

### 6.3 Check Data Integrity

```bash
# Run diagnostic script
python scripts/diagnose_data.py

# Expected output
# ✓ Data directory structure valid
# ✓ S&P 500 prices loaded for 497 tickers
# ✓ Date range: 2020-01-02 to 2026-01-17
# ✗ Fundamental data missing (optional)
# ✗ Earnings data missing (optional)
```

---

## 7. Reproducible Environment Export

### 7.1 Create Dependency Snapshot

**Via pip**:

```bash
# Freeze exact package versions
pip freeze > requirements-lock.txt

# Install from lock file (ensures exact reproducibility)
pip install -r requirements-lock.txt
```

**Via conda**:

```bash
# Export environment
conda env export > environment.yml

# Recreate on another machine
conda env create -f environment.yml
```

### 7.2 Docker for Perfect Reproducibility

The provided `docker/Dockerfile` and `docker-compose.yml` guarantee **identical execution** across all systems.

```bash
# On any machine with Docker:
docker-compose -f docker/docker-compose.yml up
# Runs with exact same Python version, dependencies, and configurations
```

---

## 8. Next Steps

After successful installation:

1. **Run a quick sanity check**:
   ```bash
   python scripts/diagnose_data.py
   ```

2. **Explore sample data**:
   ```bash
   jupyter lab
   # Open notebooks/01_data_exploration.ipynb
   ```

3. **Train a minimal model** (1 hour):
   ```bash
   python scripts/train_models.py --sample-size 0.1 --max-tickers 50
   ```

4. **Run a small backtest**:
   ```bash
   python scripts/run_backtest.py --universe sp500 --start-date 2024-01-01
   ```

5. **Read [Contributing Guidelines](contributing.md)** for development workflow.

---

## 9. Further Help

- **Configuration questions**: See [config/settings.py](../config/settings.py) for all available parameters
- **Data schema questions**: See [Data Guide](data.md)
- **Architecture questions**: See [Architecture Guide](architecture.md)
- **Development workflow**: See [Contributing Guide](contributing.md)
- **Common problems**: See [FAQ](faq.md)
