import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Setup Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Environment Variables (CPU Throttling for Tests)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

# 3. Shared Fixtures
@pytest.fixture(scope="session")
def synthetic_data():
    """
    Creates a synthetic panel dataset for testing models.
    Returns: (df, features, target_col)
    """
    np.random.seed(42)
    n_tickers = 10
    n_days = 100
    n_features = 10
    
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    rows = []
    for ticker in tickers:
        sector = np.random.choice(["Tech", "Finance", "Health"])
        industry = np.random.choice(["Soft", "Bank", "Pharma"])
        
        for date in dates:
            row = {
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "target": np.random.normal(0, 0.02)
            }
            # Add numeric features
            for i in range(n_features):
                row[f"f_{i:03d}"] = np.random.normal()
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # Set dtypes
    df["sector"] = df["sector"].astype("category")
    df["industry"] = df["industry"].astype("category")
    
    features = [c for c in df.columns if c.startswith("f_")] + ["sector", "industry"]
    return df, features, "target"

@pytest.fixture(scope="session")
def sample_covariance_matrix():
    """Creates a dummy covariance matrix for optimization tests."""
    tickers = [f"TICK{i:03d}" for i in range(5)]
    cov = pd.DataFrame(
        np.identity(5) * 0.0004, # Low variance
        index=tickers,
        columns=tickers
    )
    return cov, tickers

@pytest.fixture(scope="session")
def sample_expected_returns(sample_covariance_matrix):
    """Creates dummy expected returns."""
    _, tickers = sample_covariance_matrix
    return {t: 0.01 * (i + 1) for i, t in enumerate(tickers)}

# Custom Objective for Models (Shared)
def weighted_symmetric_mae(y_true, y_pred):
    residuals = y_true - y_pred
    weights = np.where(y_true * y_pred < 0, 2.0, 1.0)
    grad = -weights * np.tanh(residuals)
    hess = np.maximum(weights * (1.0 - np.tanh(residuals) ** 2), 1e-3)
    return grad, hess

# Inject into main for pickle compatibility during tests
try:
    sys.modules["__main__"].weighted_symmetric_mae = weighted_symmetric_mae
except Exception:
    pass