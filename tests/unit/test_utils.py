"""
Utility Functions Validation Suite
==================================
Unit testing suite validating the mathematical, temporal, I/O, and architectural 
primitives of the quantitative platform.

Purpose
-------
This module isolates and verifies the core utility functions. It rigorously tests
financial mathematics (e.g., Sharpe, Sortino, Drawdown), time-series date alignments,
Parquet-based persistence resilience, and Aspect-Oriented Programming (AOP) decorators
(e.g., retry backoff algorithms, execution timing).

Role in Quantitative Workflow
-----------------------------
Guarantees the foundational building blocks of the platform perform deterministically
across edge cases, such as zero-variance regimes, absent files, and intermittent
network failures, ensuring stability in both research and production environments.

Dependencies
------------
- **Pytest**: Test execution framework and temporary directory provisioning.
- **Pandas/NumPy**: Synthesis of mathematical boundaries and time-series arrays.
- **Unittest.Mock**: Extensively patches filesystem I/O and latency decorators to
  guarantee deterministic evaluation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from quant_alpha.utils.math_utils import (
    calculate_returns,
    calculate_log_returns,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_sortino,
    calculate_drawdown
)
from quant_alpha.utils.date_utils import (
    get_trading_days,
    align_dates,
    get_previous_trading_day,
    is_trading_day
)
from quant_alpha.utils.io_utils import save_parquet, load_parquet
from quant_alpha.utils.decorators import retry, time_execution

class TestMathUtils:
    """
    Validation suite verifying the numerical stability and accuracy of financial mathematics.
    """

    def test_calculate_returns(self):
        """
        Validates the derivation of discrete period-over-period simple returns.

        Args:
            None

        Returns:
            None
        """
        prices = pd.Series([100.0, 110.0, 99.0])
        returns = calculate_returns(prices)
        
        assert np.isnan(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(0.10)
        assert returns.iloc[2] == pytest.approx(-0.10)

    def test_calculate_log_returns(self):
        """
        Validates the derivation of continuous logarithmic returns.

        Args:
            None

        Returns:
            None
        """
        prices = pd.Series([100.0, 110.0, 99.0])
        log_rets = calculate_log_returns(prices)
        
        assert np.isnan(log_rets.iloc[0])
        assert log_rets.iloc[1] == pytest.approx(np.log(110/100))
        assert log_rets.iloc[2] == pytest.approx(np.log(99/110))

    def test_calculate_sharpe_zero_volatility(self):
        """
        Ensures resilience of the Sharpe ratio against zero-variance regimes.

        Evaluates perfectly flat return profiles to assert that division-by-zero 
        singularities are bypassed gracefully.

        Args:
            None

        Returns:
            None
        """
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        
        assert sharpe == 0.0, f"Expected 0.0 for zero vol, got {sharpe}"

    def test_calculate_sharpe_normal(self):
        """
        Validates standard Sharpe ratio scaling against randomized Gaussian return streams.

        Args:
            None

        Returns:
            None
        """
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sharpe = calculate_sharpe(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_sharpe_negative(self):
        """
        Verifies arithmetic logic preserves directional negativity in risk-adjusted performance.

        Args:
            None

        Returns:
            None
        """
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.01, 100))
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        assert sharpe < 0

    def test_calculate_max_drawdown_positive_curve(self):
        """
        Confirms monotonic asset growth architectures resolve to absolute zero drawdown.

        Args:
            None

        Returns:
            None
        """
        equity = pd.Series(np.linspace(100, 200, 50))
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_calculate_max_drawdown_crash(self):
        """
        Verifies severe discrete peak-to-trough magnitude extractions across nonlinear trajectories.

        Args:
            None

        Returns:
            None
        """
        equity = pd.Series([100, 80, 50, 60, 75])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.5)

    def test_calculate_drawdown_series(self):
        """
        Evaluates high-water mark vector boundaries constructing sequential drawdown arrays.

        Args:
            None

        Returns:
            None
        """
        equity = pd.Series([100, 110, 99, 120], index=[1, 2, 3, 4])
        dd = calculate_drawdown(equity)
        expected = pd.Series([0.0, 0.0, -0.1, 0.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(dd, expected)

    def test_calculate_max_drawdown_recovery(self):
        """
        Validates the strict retention of historical peak contractions despite subsequent total recovery.

        Args:
            None

        Returns:
            None
        """
        equity = pd.Series([100, 50, 100, 150])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.5)

    def test_calculate_sharpe_short_history(self):
        """
        Verifies defensive constraints trigger safely against structurally insufficient longitudinal depths.

        Args:
            None

        Returns:
            None
        """
        returns = pd.Series([0.01])
        assert calculate_sharpe(returns) == 0.0

    def test_calculate_sortino_no_downside(self):
        """
        Confirms mathematical stability evaluating positive-only variance in Sortino calculations.

        Args:
            None

        Returns:
            None
        """
        returns = pd.Series([0.01, 0.02, 0.01, 0.03]) 
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        assert sortino == np.inf or sortino > 1000

    def test_calculate_sortino_normal(self):
        """
        Validates standard Sortino scaling against randomized Gaussian return streams.

        Args:
            None

        Returns:
            None
        """
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sortino = calculate_sortino(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        assert sortino != np.inf

    def test_calculate_sortino_mixed(self):
        """
        Verifies arithmetic bounding across structurally mixed upside and downside return regimes.

        Args:
            None

        Returns:
            None
        """
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        assert isinstance(sortino, float)
        assert sortino > -100 and sortino < 100

class TestDateUtils:
    """
    Validation suite verifying temporal resolution and alignment paradigms.
    """

    def test_get_trading_days(self):
        """
        Confirms accurate structural derivation of business day calendar arrays.

        Args:
            None

        Returns:
            None
        """
        days = get_trading_days("2023-01-01", "2023-01-05")
        
        assert len(days) == 3
        expected_days = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
        assert all(days == expected_days)

    def test_align_dates(self):
        """
        Verifies explicit index intersection behaviors mapping temporally disjoint architectures.

        Args:
            None

        Returns:
            None
        """
        df1 = pd.DataFrame({"A": [1, 2]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        df2 = pd.DataFrame({"B": [3, 4]}, index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
        
        a1, a2 = align_dates(df1, df2)
        
        assert len(a1) == 1
        assert len(a2) == 1
        assert a1.index[0] == pd.Timestamp("2023-01-02")
        assert a2.index[0] == pd.Timestamp("2023-01-02")

class TestIOUtils:
    """
    Validation suite verifying transient data serialization and physical IO bindings.
    """

    def test_save_and_load_parquet(self, tmp_path):
        """
        Evaluates exact physical schema persistency across Apache Parquet writes and reads.

        Args:
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        file_path = tmp_path / "test.parquet"
        
        save_parquet(df, file_path)
        
        assert file_path.exists(), f"File {file_path} was not created"
        
        loaded_df = load_parquet(file_path)
        
        pd.testing.assert_frame_equal(df, loaded_df)

    @patch("quant_alpha.utils.io_utils.Path.exists")
    @patch("quant_alpha.utils.io_utils.pd.read_parquet")
    def test_load_parquet_with_mock(self, mock_read, mock_exists, tmp_path):
        """
        Verifies deserialization routing using isolated mock dependency constraints.

        Args:
            mock_read (MagicMock): Intercepted Pandas Parquet reader logic.
            mock_exists (MagicMock): Intercepted Pathlib existence flag.
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        df = pd.DataFrame({"col1": [1, 2, 3]})
        file_path = tmp_path / "mock_test.parquet"
        
        mock_exists.return_value = True
        mock_read.return_value = df
        
        loaded_df = load_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded_df)
        mock_read.assert_called_once()

    @patch("quant_alpha.utils.io_utils.Path.mkdir")
    @patch("quant_alpha.utils.io_utils.pd.DataFrame.to_parquet")
    def test_save_parquet_with_mock(self, mock_to_parquet, mock_mkdir, tmp_path):
        """
        Verifies serialization invocation parameters explicitly against Mock assertions.

        Args:
            mock_to_parquet (MagicMock): Intercepted Pandas Parquet writer logic.
            mock_mkdir (MagicMock): Intercepted Pathlib directory constructor.
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        df = pd.DataFrame({"col1": [1, 2, 3]})
        file_path = tmp_path / "mock_save.parquet"
        
        save_parquet(df, file_path)
        
        mock_to_parquet.assert_called()

    def test_load_parquet_missing_file(self, tmp_path):
        """
        Asserts defensive empty DataFrame returns explicitly guarding against missing payloads.

        Args:
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        file_path = tmp_path / "ghost.parquet"
        df = load_parquet(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    @patch("quant_alpha.utils.io_utils.Path.mkdir")
    @patch("pandas.DataFrame.to_parquet")
    def test_save_parquet_mkdir(self, mock_to_parquet, mock_mkdir, tmp_path):
        """
        Ensures nested hierarchical directory provisioning operates structurally.

        Args:
            mock_to_parquet (MagicMock): Intercepted Pandas Parquet writer logic.
            mock_mkdir (MagicMock): Intercepted Pathlib directory constructor.
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        nested_path = tmp_path / "folder" / "subfolder" / "data.parquet"
        df = pd.DataFrame({"A": [1]})
        
        save_parquet(df, nested_path)
        mock_mkdir.assert_called()
        mock_to_parquet.assert_called()

    def test_save_parquet_fallback(self, tmp_path):
        """
        Confirms engine fallback mechanics mathematically triggering upon pyarrow C-extension faults.

        Args:
            tmp_path (pathlib.Path): Injected pytest temporary filesystem boundary.

        Returns:
            None
        """
        df = pd.DataFrame({"A": [1]})
        file_path = tmp_path / "fallback.parquet"
        
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            mock_to_parquet.side_effect = [Exception("Pyarrow fail"), None]
            
            save_parquet(df, file_path)
            
            assert mock_to_parquet.call_count == 2
            assert mock_to_parquet.call_args_list[0][1]['engine'] == 'pyarrow'
            assert mock_to_parquet.call_args_list[1][1]['engine'] == 'fastparquet'

class TestDecorators:
    """
    Validation suite evaluating Aspect-Oriented latency reporting and resiliency logic.
    """

    def test_retry_success(self):
        """
        Validates deterministic decorator passthrough upon immediate success contexts.

        Args:
            None

        Returns:
            None
        """
        mock_func = MagicMock(return_value="success")
        mock_func.__name__ = "mock_func"
        decorated = retry(max_retries=3)(mock_func)
        
        res = decorated()
        assert res == "success"
        assert mock_func.call_count == 1

    def test_retry_failure_then_success(self):
        """
        Evaluates correct backoff state progression routing subsequent execution blocks successfully.

        Args:
            None

        Returns:
            None
        """
        mock_func = MagicMock(side_effect=[ValueError("Fail"), "success"])
        mock_func.__name__ = "mock_func"
        
        decorated = retry(max_retries=3, delay=0.01)(mock_func)
        
        res = decorated()
        assert res == "success"
        assert mock_func.call_count == 2

    def test_retry_max_retries_exceeded(self):
        """
        Validates the terminal propagation of exceptions after exhaustion of the backoff retry pool.

        Args:
            None

        Returns:
            None
        """
        mock_func = MagicMock(side_effect=ValueError("Persistent Fail"))
        mock_func.__name__ = "mock_func"
        
        decorated = retry(max_retries=2, delay=0.01)(mock_func)
        
        with pytest.raises(ValueError, match="Persistent Fail"):
            decorated()
        
        # Validates structural constraint: 1 initial attempt + 1 execution in retry loop = 2 total invocations
        assert mock_func.call_count == 2

    def test_time_execution_logging(self, caplog):
        """
        Ensures precise computational latency payloads route cleanly to global observability metrics.

        Args:
            caplog (pytest.LogCaptureFixture): Pytest native logger boundary injector.

        Returns:
            None
        """
        import logging
        
        @time_execution
        def slow_func():
            time.sleep(0.01)
            return "done"
            
        with caplog.at_level(logging.INFO):
            res = slow_func()
            
        assert res == "done"
        # Asserts structural format conformity of the emitted telemetry payload
        assert "[timer]" in caplog.text
        assert "slow_func" in caplog.text
        assert "completed in" in caplog.text