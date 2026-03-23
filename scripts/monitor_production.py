r"""
Production Signal Monitoring & Drift Detection
==============================================
Automated audit pipeline for detecting distributional shifts, signal decay, and
data integrity issues in live alpha predictions.

Purpose
-------
This module serves as the **Model Observability Layer**, executed post-inference
to validate the statistical properties of generated signals. It acts as a 
blocking gate in the CI/CD pipeline, preventing the propagation of corrupted 
or degraded signals to the execution engine.

Key Checks:
1.  **Population Stability Index (PSI)**: Quantifies distributional drift 
    ($D_{KL}$) between the reference baseline and current signals.
2.  **Signal Entropy**: Detects model collapse (variance $\sigma^2 \to 0$).
3.  **Universe Integrity**: Validates the breadth of the tradeable universe ($N_{tickers}$).
4.  **Signal Stability**: Measures day-over-day autocorrelation to detect 
    excessive turnover or regime shifts.

Usage:
------
Executed automatically after `generate_predictions.py`.

.. code-block:: bash

    # Standard execution (Default threshold $\\psi < 0.15$)
    python scripts/monitor_production.py

    # Strict monitoring for stable regimes
    python scripts/monitor_production.py --psi-threshold 0.10 --min-std 0.01

Importance
----------
-   **Risk Mitigation**: Prevents "Silent Failures" where models output valid 
    floats but meaningless signals (e.g., all zeros or constant values).
-   **Drift Detection**: Identifies **Concept Drift** (market regime changes) 
    requiring model retraining.
-   **Operational Safety**: Enforces $O(1)$ sanity checks before capital allocation.

Tools & Frameworks
------------------
-   **NumPy**: Efficient histogram computation and vector math for PSI.
-   **Pandas**: Time-series alignment and Parquet I/O.
-   **Sys/Argparse**: CI/CD integration via exit codes (0=Pass, 1=Fail).
"""

import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import config
from quant_alpha.utils import load_parquet, setup_logging

setup_logging()
logger = logging.getLogger("ProdMonitor")


def calculate_psi(expected, actual, buckets=10, buckettype='quantiles'):
    r"""
    Calculates the Population Stability Index (PSI) to measure distributional drift.

    Mathematical Formulation:
    .. math::
        PSI = \sum_{i=1}^{B} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)
    
    Where:
    - $P_i$: Proportion of expected population in bin $i$.
    - $Q_i$: Proportion of actual population in bin $i$.
    - $B$: Number of buckets.

    A PSI < 0.10 implies no significant shift; PSI > 0.25 implies major drift.
    """
    
    # Data Hygiene: Filter NaNs to prevent propagation errors
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define breakpoints
    if buckettype == 'bins':
        breakpoints = np.linspace(np.min(expected), np.max(expected), buckets + 1)
    else:
        # Quantile bins: Ensures equal-sized buckets in the reference distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    # Boundary Handling: Extend range to -inf/+inf to capture Fat Tail events in actual data
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Degeneracy Handling: Remove duplicate bins caused by low-variance distributions
    breakpoints = np.unique(breakpoints)

    # Histogram Computation
    expected_counts, _ = np.histogram(expected, breakpoints)
    actual_counts, _ = np.histogram(actual, breakpoints)
    
    # Probability Mass Function (PMF) with Epsilon Smoothing
    # $\\epsilon = 10^{-4}$ prevents division by zero / log(0)
    epsilon = 1e-4
    expected_percents = np.maximum(expected_counts / len(expected), epsilon)
    actual_percents = np.maximum(actual_counts / len(actual), epsilon)

    # PSI Summation
    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    psi_total = np.sum(psi_values)
    
    return psi_total


def main():
    parser = argparse.ArgumentParser(description="Production Model Monitoring")
    parser.add_argument("--psi-threshold", type=float, default=0.15, help="PSI drift threshold (default: 0.15)")
    parser.add_argument("--min-std", type=float, default=0.005, help="Minimum signal standard deviation")
    parser.add_argument("--lookback", type=int, default=5, help="Days to look back for reference distribution")
    args = parser.parse_args()

    logger.info("Starting Production Health Checks...")

    # --- Data Ingestion ---
    pred_dir = config.RESULTS_DIR / 'predictions'
    files = sorted(pred_dir.glob('alpha_signals_*.parquet'))

    if not files:
        logger.error("❌ No prediction files found.")
        sys.exit(1)

    # 1. Target Identification (Current Production Signal)
    latest_file = files[-1]
    latest_date = latest_file.stem.replace("alpha_signals_", "")
    logger.info(f"Latest Prediction: {latest_file.name} ({latest_date})")

    df_latest = load_parquet(latest_file)
    
    # 2. Reference Distribution Construction
    # Strategy: Use rolling window of recent history (default 5 days) to define "Normal" behavior.
    ref_files = files[-(args.lookback+1):-1]
    df_ref = pd.DataFrame()

    if ref_files:
        logger.info(f"Loading {len(ref_files)} historical files for reference...")
        df_ref = pd.concat([load_parquet(f) for f in ref_files])
    else:
        # Fallback: Cold Start / Missing History
        # Uses the aggregate cache if individual daily files are insufficient.
        cache_path = config.CACHE_DIR / 'ensemble_predictions.parquet'
        if cache_path.exists():
            logger.info("Daily history missing. Using cumulative cache for reference.")
            df_cache = load_parquet(cache_path)
            if 'date' in df_cache.columns:
                df_cache['date'] = pd.to_datetime(df_cache['date'])
                # Exclude latest date
                if 'date' in df_latest.columns:
                    current_dt = pd.to_datetime(df_latest['date'].iloc[0])
                    df_ref = df_cache[df_cache['date'] < current_dt]
                else:
                    df_ref = df_cache

    if df_ref.empty:
        logger.warning("⚠️ COLD START: No reference data. PSI will be 0.")
        df_ref = df_latest.copy()

    # --- Statistical Audits ---
    issues = []

    # Audit 1: Distributional Drift (PSI)
    psi = calculate_psi(df_ref['ensemble_alpha'].values, df_latest['ensemble_alpha'].values)
    if psi > args.psi_threshold:
        msg = f"HIGH DRIFT: PSI {psi:.4f} > {args.psi_threshold}"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ PSI Check Passed: {psi:.4f}")

    # Audit 2: Signal Entropy / Collapse Check
    std_dev = df_latest['ensemble_alpha'].std()
    if std_dev < args.min_std:
        msg = f"SIGNAL COLLAPSE: Std Dev {std_dev:.5f} < {args.min_std}"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ Variance Check Passed: {std_dev:.5f}")

    # Audit 3: Universe Integrity Check
    n_tickers = df_latest['ticker'].nunique()
    if n_tickers < 100: # Hard constraint for statistical significance
        msg = f"UNIVERSE SHRINKAGE: Only {n_tickers} tickers found"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ Universe Check Passed: {n_tickers} tickers")

    # Audit 4: Temporal Stability (Turnover Proxy)
    # Calculates auto-correlation $\\rho(S_t, S_{t-1})$
    if len(files) >= 2:
        prev_file = files[-2]
        df_prev = load_parquet(prev_file)
        merged = pd.merge(df_latest, df_prev, on='ticker', suffixes=('_curr', '_prev'))
        if not merged.empty:
            corr = merged['ensemble_alpha_curr'].corr(merged['ensemble_alpha_prev'])
            if corr < 0.5:
                logger.warning(f"⚠️ Low Stability: Day-over-day correlation {corr:.4f}")
            else:
                logger.info(f"✅ Stability Check Passed: Corr {corr:.4f}")

    # --- Final Report ---
    print("\n" + "="*40)
    print(f"  MONITORING REPORT: {latest_date}")
    print("="*40)
    print(f"  PSI:        {psi:.4f}")
    print(f"  Signal Std: {std_dev:.5f}")
    print(f"  Tickers:    {n_tickers}")
    print(f"  Status:     {'FAIL' if issues else 'PASS'}")
    print("="*40 + "\n")

    if issues:
        sys.exit(1) # Fail the pipeline
    
    sys.exit(0)

if __name__ == "__main__":
    main()