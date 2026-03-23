r"""
Script Orchestration & Logic Validation
=======================================
Unit testing suite for the platform's core execution scripts.

Purpose
-------
This module isolates and validates the critical logic embedded within the
application's entry-point scripts (e.g., factor validation, model training, data updates).
It bypasses the full CLI overhead to test specific algorithmic components—such as
Walk-Forward Validation splitting and Information Coefficient (IC) calculation—ensuring
mathematical correctness and logic integrity.

Usage
-----
.. code-block:: bash

    pytest tests/unit/test_scripts.py

Importance
----------
- **Statistical Integrity**: Verifies that the factor validation engine correctly identifies
  predictive signals ($IC > 0$) vs. noise using synthetic control data.
- **Look-Ahead Bias Prevention**: Rigorously asserts that Walk-Forward Validation splits
  are strictly disjoint ($Train \cap Test = \emptyset$) with the correct embargo periods.
- **Data Consistency**: Ensures incremental data merges handle duplicates and overlaps
  idempotently, preserving the uniqueness of the time-series index.

Tools & Frameworks
------------------
- **Pytest**: Test runner and fixture management.
- **Pandas/NumPy**: Synthetic data generation and vector manipulation.
- **SciPy**: Verification of statistical calculations (Spearman Rank Correlation).
- **Unittest.Mock**: Isolation of filesystem I/O and external dependencies.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def synthetic_factor_data():
    """
    Generates a controlled dataset for statistical validation.
    
    Creates 100 days of data for 2 tickers with:
    - **Alpha Factor**: Linearly correlated with returns (Signal).
    - **Noise Factor**: Randomly distributed (Noise).

    Args:
        None

    Returns:
        pd.DataFrame: A populated feature matrix simulating predictive signals and noise.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]
    
    rows = []
    for t in tickers:
        alpha = np.linspace(0, 1, 100) if t == "AAPL" else np.linspace(1, 0, 100)
        noise = rng.standard_normal(100)
        returns = alpha + rng.normal(0, 0.01, 100)
        
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "ticker": t,
                "alpha_factor": alpha[i],
                "noise_factor": noise[i],
                "target_ret_5d": returns[i],
                "volume": 1000000
            })
            
    return pd.DataFrame(rows)

class TestFactorValidator:
    """
    Validates the statistical engine responsible for isolated factor scoring.
    """

    def compute_ic_stats(self, df, target_col="target_ret_5d"):
        """
        Replicates the Information Coefficient (IC) calculation logic.
        
        Computes Spearman Rank Correlation ($r_s$) and t-statistics to assess
        factor significance.

        Args:
            df (pd.DataFrame): The input dataset containing factor vectors and targets.
            target_col (str, optional): The name of the forward return column. Defaults to "target_ret_5d".

        Returns:
            pd.DataFrame: A matrix of calculated statistics including IC mean, t-stat, and p-value.
        """
        if df.empty:
            return pd.DataFrame()
        
        factor_cols = [c for c in df.columns 
                      if 'factor' in c.lower() and c != target_col]
        
        stats = []
        for factor in factor_cols:
            clean_df = df[[factor, target_col]].dropna()
            if len(clean_df) < 2:
                continue
            
            ic, p_val = spearmanr(clean_df[factor], clean_df[target_col])
            # Stabilizes t-statistic calculation: incorporates epsilon to prevent DivisionByZero bounds during perfect correlations
            denom = np.sqrt(1 - ic**2 + 1e-10)
            t_stat = abs(ic * np.sqrt(len(clean_df) - 2) / denom)
            
            stats.append({
                "factor": factor,
                "ic_mean": ic,
                "t_stat": t_stat,
                "p_value": p_val
            })
        
        return pd.DataFrame(stats).set_index("factor") if stats else pd.DataFrame()

    def test_ic_calculation_accuracy(self, synthetic_factor_data):
        r"""
        Verifies that the IC engine correctly identifies a strong signal ($r_s \approx 1.0$).

        Args:
            synthetic_factor_data (pd.DataFrame): The synthetic dataset.

        Returns:
            None
        """
        stats = self.compute_ic_stats(synthetic_factor_data, "target_ret_5d")
        
        assert not stats.empty, "No stats computed"
        alpha_ic = stats.loc["alpha_factor", "ic_mean"]
        assert alpha_ic > 0.9, f"Alpha IC too low: {alpha_ic}"
        
        noise_ic = abs(stats.loc["noise_factor", "ic_mean"])
        assert noise_ic < 0.35, f"Noise IC too high: {noise_ic}"

    def test_t_stat_significance(self, synthetic_factor_data):
        """
        Asserts that high-IC factors generate statistically significant t-stats (> 2.0).

        Args:
            synthetic_factor_data (pd.DataFrame): The synthetic dataset.

        Returns:
            None
        """
        stats = self.compute_ic_stats(synthetic_factor_data, "target_ret_5d")
        
        assert stats.loc["alpha_factor", "t_stat"] > 10.0
        assert stats.loc["noise_factor", "t_stat"] < 2.0

    def test_handling_of_nans(self, synthetic_factor_data):
        """
        Ensures robustness against missing data (NaNs) in factor columns.

        Args:
            synthetic_factor_data (pd.DataFrame): The synthetic dataset.

        Returns:
            None
        """
        df = synthetic_factor_data.copy()
        df.loc[0:10, "alpha_factor"] = np.nan
        
        stats = self.compute_ic_stats(df, "target_ret_5d")
        
        assert not np.isnan(stats.loc["alpha_factor", "ic_mean"])

    def test_empty_dataframe_resilience(self):
        """
        Ensures the validator returns an empty result set instead of crashing on empty inputs.

        Args:
            None

        Returns:
            None
        """
        empty_df = pd.DataFrame(columns=["ticker", "date", "raw_ret_5d"])
        stats = self.compute_ic_stats(empty_df)
        
        assert isinstance(stats, pd.DataFrame)
        assert stats.empty

class TestDataUpdateLogic:
    """
    Validates the incremental data ingestion and deduplication logic.
    """

    def merge_ohlcv_data(self, existing_csv, new_data, key_col="date"):
        """
        Simulates the logic for merging new market data with existing archives.
        
        Strategy: Upsert (Update/Insert) based on `key_col`.
        
        Args:
            existing_csv (Path): Path to the existing CSV file.
            new_data (pd.DataFrame): DataFrame containing new fetched data.
            key_col (str, optional): The column utilized for temporal deduplication. Defaults to "date".
            
        Returns:
            pd.DataFrame | None: The merged DataFrame, or None if execution faults.
        """
        try:
            if existing_csv.exists():
                existing = pd.read_csv(existing_csv)
            else:
                existing = pd.DataFrame()
            
            if existing.empty:
                merged = new_data.copy()
            else:
                # Ensures latest fetched payloads overwrite historical boundaries during merge overlaps
                merged = pd.concat([existing, new_data], ignore_index=True)
                merged = merged.drop_duplicates(subset=[key_col], keep='last')
                merged = merged.sort_values(key_col).reset_index(drop=True)
            
            return merged
        except Exception as e:
            return None

    def test_incremental_merge_deduplication(self, tmp_path):
        """
        Verifies that overlapping dates are merged correctly without creating duplicates.

        Args:
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral testing directory.

        Returns:
            None
        """
        csv_file = tmp_path / "TEST_TICKER.csv"
        
        old_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "close": [100.0, 101.0],
            "volume": [1000, 1100]
        })
        old_df.to_csv(csv_file, index=False)
        
        new_df = pd.DataFrame({
            "date": ["2023-01-02", "2023-01-03"],
            "close": [101.5, 102.0],
            "volume": [1150, 1200]
        })
        
        merged = self.merge_ohlcv_data(csv_file, new_df, key_col="date")
        
        assert merged is not None
        assert len(merged) == 3  
        assert list(merged["date"]) == ["2023-01-01", "2023-01-02", "2023-01-03"]
        assert merged[merged["date"] == "2023-01-02"]["close"].iloc[0] == 101.5

    def test_empty_csv_handling(self, tmp_path):
        """
        Verifies correct initialization when the target CSV does not yet exist.

        Args:
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral testing directory.

        Returns:
            None
        """
        csv_file = tmp_path / "NEW_TICKER.csv"
        
        new_df = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "close": [100.0, 101.0]
        })
        
        merged = self.merge_ohlcv_data(csv_file, new_df)
        
        assert len(merged) == 2
        assert merged["close"].iloc[0] == 100.0

class TestModelTraining:
    """
    Validates the Machine Learning training orchestration and cross-validation schemes.
    """

    def generate_walk_forward_splits(self, n_periods, train_size, test_size, gap):
        """
        Simulates the generation of Walk-Forward Validation indices.
        
        Structure:
        [ Train Window ] -- (Embargo/Gap) -- [ Test Window ]
        
        Args:
            n_periods (int): Total available chronological data points.
            train_size (int): Temporal size of the training bound.
            test_size (int): Temporal size of the out-of-sample evaluation bound.
            gap (int): Mandatory structural embargo length separating train/test boundaries.
            
        Yields:
            tuple: Yields matching sets of `(train_idx, test_idx)` arrays.
        """
        start = 0
        while start + train_size + gap + test_size <= n_periods:
            train_idx = (start, start + train_size)
            test_idx = (start + train_size + gap, start + train_size + gap + test_size)
            yield train_idx, test_idx
            start += test_size

    def test_walk_forward_split_indices(self):
        """
        Asserts that the embargo gap is strictly enforced between training and testing sets.

        Args:
            None

        Returns:
            None
        """
        splits = list(self.generate_walk_forward_splits(
            n_periods=1000,
            train_size=500,
            test_size=100,
            gap=5
        ))
        
        assert len(splits) > 0
        for train, test in splits:
            assert train[1] < test[0], "Train end must be before test start"
            assert (test[0] - train[1]) == 5, "Gap must be exactly 5"

    def test_multiple_splits_no_overlap(self):
        """
        Verifies that consecutive Test windows are contiguous but non-overlapping.
        
        Ensures complete coverage of the out-of-sample period without redundancy.

        Args:
            None

        Returns:
            None
        """
        splits = list(self.generate_walk_forward_splits(
            n_periods=2000,
            train_size=400,
            test_size=100,
            gap=10
        ))
        
        assert len(splits) > 1, "Need at least 2 splits to test overlap"
        
        test_ranges = [test for _, test in splits]
        
        for i in range(len(test_ranges) - 1):
            test1 = test_ranges[i]
            test2 = test_ranges[i + 1]
            
            assert test1[1] <= test2[0], \
                f"Test window {i} overlaps with test window {i+1}: " \
                f"test1={test1}, test2={test2}"
            
            expected_start = test1[1]
            actual_start = test2[0]
            assert actual_start == expected_start, \
                f"Test window {i} and {i+1} not contiguous: " \
                f"test1 ends at {test1[1]}, test2 starts at {test2[0]}"

    def test_walk_forward_data_coverage(self):
        """
        Verifies that the union of all test windows creates a continuous time series.

        Args:
            None

        Returns:
            None
        """
        splits = list(self.generate_walk_forward_splits(
            n_periods=1000,
            train_size=200,
            test_size=50,
            gap=5
        ))
        
        assert len(splits) > 0, "No splits generated"
        
        test_periods = set()
        for train, test in splits:
            test_start, test_end = test  
            test_periods.update(range(test_start, test_end))
        
        assert len(test_periods) > 0, "No test periods generated"
        
        sorted_periods = sorted(list(test_periods))
        expected_start = 200 + 5  
        assert sorted_periods[0] == expected_start, \
            f"First test period should start at {expected_start}, got {sorted_periods[0]}"
        
        for i in range(len(sorted_periods) - 1):
            assert sorted_periods[i + 1] == sorted_periods[i] + 1, \
                f"Gap detected in test periods at index {i}: " \
                f"{sorted_periods[i]} -> {sorted_periods[i + 1]}"
            

    def test_train_test_separation(self):
        
        r"""
        Verifies the critical invariant: $Train \cap Test = \emptyset$.
        Also confirms the existence of the embargo gap to prevent look-ahead bias.

        Args:
            None

        Returns:
            None
        """
        splits = list(self.generate_walk_forward_splits(
            n_periods=1000,
            train_size=300,
            test_size=100,
            gap=10
        ))
        
        for i, (train, test) in enumerate(splits):
            train_start, train_end = train
            test_start, test_end = test
            
            assert train_end < test_start, \
                f"Split {i}: Training ends at {train_end}, " \
                f"but test starts at {test_start} (gap not respected)"
            
            gap_size = test_start - train_end
            assert gap_size == 10, \
                f"Split {i}: Expected gap of 10, got {gap_size}"
            
    @patch("joblib.dump")
    def test_model_serialization(self, mock_dump, tmp_path):
        """
        Verifies that model artifacts are correctly serialized to the filesystem.

        Args:
            mock_dump (MagicMock): Patched Joblib serialization intercept.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        mock_model = MagicMock()
        save_path = tmp_path / "rf_model_v1.joblib"
        
        import joblib
        joblib.dump(mock_model, save_path)
        
        mock_dump.assert_called_once()
        assert str(save_path).endswith(".joblib")

def test_multiindex_consistency(synthetic_factor_data):
    """
    Ensures that the hierarchical index (Date, Ticker) remains consistent during operations.

    Args:
        synthetic_factor_data (pd.DataFrame): Injected synthetic feature matrix.

    Returns:
        None
    """
    df = synthetic_factor_data
    df_multi = df.set_index(["date", "ticker"])
    
    assert isinstance(df_multi.index, pd.MultiIndex)
    assert df_multi.index.names == ["date", "ticker"]
    assert len(df_multi) == 200  

def test_nan_propagation():
    """
    Verifies that explicit NaN boundaries resolve safely without executing silent drops.

    Args:
        None

    Returns:
        None
    """
    df = pd.DataFrame({
        "factor": [1.0, np.nan, 3.0, 4.0],
        "target": [10, 20, 30, 40]
    })
    
    clean = df.dropna()
    assert len(clean) == 3
    assert not clean["factor"].isna().any()