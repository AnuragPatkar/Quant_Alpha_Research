"""
monitor_production.py
=====================
Automated Health Checks & Drift Detection
-----------------------------------------
Runs the same logic as 05_production_monitoring.ipynb but in a headless script
suitable for CI/CD pipelines or daily cron jobs.

Exits with status code 1 if CRITICAL checks fail (blocking deployment).
Exits with status code 0 if PASS or WARNING.

Usage:
    python scripts/monitor_production.py
    python scripts/monitor_production.py --psi-threshold 0.2
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
    """Calculate the PSI (Population Stability Index) for a single variable"""
    
    # Robustness: Filter NaNs and check for empty data
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Define breakpoints
    if buckettype == 'bins':
        breakpoints = np.linspace(np.min(expected), np.max(expected), buckets + 1)
    else:
        # Quantile bins
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))

    # Guard against out-of-range values (Fat Tails) in actual data
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Handle degenerate distributions (constant values) by removing duplicate bins
    breakpoints = np.unique(breakpoints)

    # Calculate frequencies
    expected_counts, _ = np.histogram(expected, breakpoints)
    actual_counts, _ = np.histogram(actual, breakpoints)
    
    # Convert to probabilities (add epsilon to avoid division by zero)
    epsilon = 1e-4
    expected_percents = np.maximum(expected_counts / len(expected), epsilon)
    actual_percents = np.maximum(actual_counts / len(actual), epsilon)

    # Calculate PSI
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

    # --- Load Data ---
    pred_dir = config.RESULTS_DIR / 'predictions'
    files = sorted(pred_dir.glob('alpha_signals_*.parquet'))

    if not files:
        logger.error("❌ No prediction files found.")
        sys.exit(1)

    # 1. Identify Latest
    latest_file = files[-1]
    latest_date = latest_file.stem.replace("alpha_signals_", "")
    logger.info(f"Latest Prediction: {latest_file.name} ({latest_date})")

    df_latest = load_parquet(latest_file)
    
    # 2. Load Reference
    ref_files = files[-(args.lookback+1):-1]
    df_ref = pd.DataFrame()

    if ref_files:
        logger.info(f"Loading {len(ref_files)} historical files for reference...")
        df_ref = pd.concat([load_parquet(f) for f in ref_files])
    else:
        # Fallback to cache
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

    # --- Checks ---
    issues = []

    # 1. PSI Drift
    psi = calculate_psi(df_ref['ensemble_alpha'].values, df_latest['ensemble_alpha'].values)
    if psi > args.psi_threshold:
        msg = f"HIGH DRIFT: PSI {psi:.4f} > {args.psi_threshold}"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ PSI Check Passed: {psi:.4f}")

    # 2. Variance
    std_dev = df_latest['ensemble_alpha'].std()
    if std_dev < args.min_std:
        msg = f"SIGNAL COLLAPSE: Std Dev {std_dev:.5f} < {args.min_std}"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ Variance Check Passed: {std_dev:.5f}")

    # 3. Universe Size
    n_tickers = df_latest['ticker'].nunique()
    if n_tickers < 100: # Hard floor
        msg = f"UNIVERSE SHRINKAGE: Only {n_tickers} tickers found"
        logger.error(f"❌ {msg}")
        issues.append(msg)
    else:
        logger.info(f"✅ Universe Check Passed: {n_tickers} tickers")

    # 4. Stability (Turnover Proxy)
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
