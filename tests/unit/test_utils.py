"""
UNIT TEST: Utilities
====================
Tests for math_utils and other utility functions.
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

    # -------------------------------------------------------------------------
    # Returns
    # -------------------------------------------------------------------------
    def test_calculate_returns(self):
        """Test simple percentage returns."""
        prices = pd.Series([100.0, 110.0, 99.0])
        returns = calculate_returns(prices)
        
        assert np.isnan(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(0.10)
        assert returns.iloc[2] == pytest.approx(-0.10)

    def test_calculate_log_returns(self):
        """Test log returns."""
        prices = pd.Series([100.0, 110.0, 99.0])
        log_rets = calculate_log_returns(prices)
        
        assert np.isnan(log_rets.iloc[0])
        assert log_rets.iloc[1] == pytest.approx(np.log(110/100))
        assert log_rets.iloc[2] == pytest.approx(np.log(99/110))

    # -------------------------------------------------------------------------
    # Sharpe Ratio
    # -------------------------------------------------------------------------
    def test_calculate_sharpe_zero_volatility(self):
        """
        Sharpe ratio should handle zero volatility (flat returns) gracefully.
        """
        # Flat returns -> std_dev = 0
        returns = pd.Series([0.01] * 100)
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        
        # Should return 0.0 or handle division by zero without crash
        assert sharpe == 0.0, f"Expected 0.0 for zero vol, got {sharpe}"

    def test_calculate_sharpe_normal(self):
        """Standard Sharpe calculation."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sharpe = calculate_sharpe(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_calculate_sharpe_negative(self):
        """Test negative Sharpe ratio."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.001, 0.01, 100))
        sharpe = calculate_sharpe(returns, risk_free_rate=0.0)
        assert sharpe < 0

    # -------------------------------------------------------------------------
    # Drawdown
    # -------------------------------------------------------------------------
    def test_calculate_max_drawdown_positive_curve(self):
        """
        If equity curve is strictly increasing, max drawdown should be 0.
        """
        equity = pd.Series(np.linspace(100, 200, 50))
        dd = calculate_max_drawdown(equity)
        assert dd == 0.0

    def test_calculate_max_drawdown_crash(self):
        """
        Verify drawdown calculation on a crash.
        100 -> 50 (50% drop) -> 75. Max DD should be 0.5.
        """
        equity = pd.Series([100, 80, 50, 60, 75])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.5)

    def test_calculate_drawdown_series(self):
        """Test the full drawdown series output."""
        equity = pd.Series([100, 110, 99, 120], index=[1, 2, 3, 4])
        # 100: 0
        # 110: 0 (new peak)
        # 99: (99-110)/110 = -0.1
        # 120: 0 (new peak)
        dd = calculate_drawdown(equity)
        expected = pd.Series([0.0, 0.0, -0.1, 0.0], index=[1, 2, 3, 4])
        pd.testing.assert_series_equal(dd, expected)

    def test_calculate_max_drawdown_recovery(self):
        """Test drawdown with full recovery."""
        equity = pd.Series([100, 50, 100, 150])
        dd = calculate_max_drawdown(equity)
        assert dd == pytest.approx(0.5)

    # -------------------------------------------------------------------------
    # Sharpe Ratio Edge Cases
    # -------------------------------------------------------------------------
    def test_calculate_sharpe_short_history(self):
        """Sharpe should return 0.0 if history is too short."""
        returns = pd.Series([0.01])
        assert calculate_sharpe(returns) == 0.0

    # -------------------------------------------------------------------------
    # Sortino Ratio
    # -------------------------------------------------------------------------
    def test_calculate_sortino_no_downside(self):
        """
        Sortino ratio with no negative returns (infinite theoretically).
        Implementation usually returns Inf or a large number.
        """
        returns = pd.Series([0.01, 0.02, 0.01, 0.03]) # All positive
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        # math_utils.py returns np.inf if downside_deviation == 0
        assert sortino == np.inf or sortino > 1000

    def test_calculate_sortino_normal(self):
        """Test normal Sortino calculation."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        sortino = calculate_sortino(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        assert sortino != np.inf

    def test_calculate_sortino_mixed(self):
        """Test Sortino with mixed returns."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        sortino = calculate_sortino(returns, risk_free_rate=0.0)
        assert isinstance(sortino, float)
        assert sortino > -100 and sortino < 100


class TestDateUtils:
    """Tests for date_utils.py using mocks for pandas_market_calendars."""

    @pytest.fixture
    def mock_calendar(self):
        with patch("quant_alpha.utils.date_utils.get_market_calendar") as mock_get:
            calendar = MagicMock()
            mock_get.return_value = calendar
            yield calendar

    def test_get_trading_days(self, mock_calendar):
        """Verify it extracts the index from the calendar schedule."""
        # Setup mock return
        # Fix: mcal returns UTC timestamps
        mock_schedule = pd.DataFrame(
            index=pd.to_datetime(["2023-01-03", "2023-01-04"]).tz_localize("UTC")
        )
        mock_calendar.schedule.return_value = mock_schedule

        days = get_trading_days("2023-01-01", "2023-01-05")
        
        assert len(days) == 2
        assert days[0] == pd.Timestamp("2023-01-03").tz_localize("UTC")
        assert days[1] == pd.Timestamp("2023-01-04").tz_localize("UTC")

    def test_align_dates(self):
        """Verify alignment by intersection of indices."""
        df1 = pd.DataFrame({"A": [1, 2]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
        df2 = pd.DataFrame({"B": [3, 4]}, index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
        
        a1, a2 = align_dates(df1, df2)
        
        assert len(a1) == 1
        assert len(a2) == 1
        assert a1.index[0] == pd.Timestamp("2023-01-02")
        assert a2.index[0] == pd.Timestamp("2023-01-02")

    def test_get_previous_trading_day(self, mock_calendar):
        """Verify logic to find previous day."""
        target = pd.Timestamp("2023-01-04").tz_localize("UTC")
        # Schedule includes 2nd and 3rd
        # Fix: mcal returns UTC, and get_previous_trading_day forces UTC comparison
        mock_schedule = pd.DataFrame(
            index=pd.to_datetime(["2023-01-02", "2023-01-03"]).tz_localize("UTC")
        )
        mock_calendar.schedule.return_value = mock_schedule

        prev = get_previous_trading_day(target)
        assert prev == pd.Timestamp("2023-01-03").tz_localize("UTC")

    def test_get_previous_trading_day_fallback(self, mock_calendar):
        """Verify fallback when schedule is empty (date before calendar start)."""
        target = pd.Timestamp("1900-01-01").tz_localize("UTC")
        # Empty schedule
        mock_calendar.schedule.return_value = pd.DataFrame()
        
        # Should fall back to BDay logic (target - 1 BDay)
        # 1900-01-01 is Monday, prev BDay is Friday 1899-12-29
        prev = get_previous_trading_day(target)
        # The implementation returns date - BDay(1).
        expected = target - pd.tseries.offsets.BusinessDay(1)
        assert prev == expected

    def test_is_trading_day(self, mock_calendar):
        """Verify valid_days check."""
        target = pd.Timestamp("2023-01-03")
        # valid_days returns list-like
        mock_calendar.valid_days.return_value = [target]
        
        assert is_trading_day(target) is True
        
        mock_calendar.valid_days.return_value = []
        assert is_trading_day(target) is False


class TestIOUtils:
    """Tests for io_utils.py using temporary directories."""

    def test_save_and_load_parquet(self, tmp_path):
        """Round trip test for parquet I/O."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        file_path = tmp_path / "test.parquet"
        
        # Save
        save_parquet(df, file_path)
        assert file_path.exists()
        
        # Load
        loaded_df = load_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_load_parquet_missing_file(self, tmp_path):
        """Loading a non-existent file should return empty DataFrame."""
        file_path = tmp_path / "ghost.parquet"
        df = load_parquet(file_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_save_parquet_mkdir(self, tmp_path):
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "folder" / "subfolder" / "data.parquet"
        df = pd.DataFrame({"A": [1]})
        
        save_parquet(df, nested_path)
        assert nested_path.exists()

    def test_save_parquet_fallback(self, tmp_path):
        """Verify fallback to fastparquet if pyarrow fails."""
        df = pd.DataFrame({"A": [1]})
        file_path = tmp_path / "fallback.parquet"
        
        # Mock df.to_parquet to raise exception on first call (pyarrow), succeed on second
        with patch.object(pd.DataFrame, 'to_parquet') as mock_to_parquet:
            mock_to_parquet.side_effect = [Exception("Pyarrow fail"), None]
            
            save_parquet(df, file_path)
            
            assert mock_to_parquet.call_count == 2
            assert mock_to_parquet.call_args_list[0][1]['engine'] == 'pyarrow'
            assert mock_to_parquet.call_args_list[1][1]['engine'] == 'fastparquet'


class TestDecorators:
    """Tests for decorators.py."""

    def test_retry_success(self):
        """Function succeeds on first try."""
        mock_func = MagicMock(return_value="success")
        decorated = retry(max_retries=3)(mock_func)
        
        res = decorated()
        assert res == "success"
        assert mock_func.call_count == 1

    def test_retry_failure_then_success(self):
        """Function fails once then succeeds."""
        # Side effect: Raise ValueError first, then return "success"
        mock_func = MagicMock(side_effect=[ValueError("Fail"), "success"])
        
        # Use small delay to speed up test
        decorated = retry(max_retries=3, delay=0.01)(mock_func)
        
        res = decorated()
        assert res == "success"
        assert mock_func.call_count == 2

    def test_retry_max_retries_exceeded(self):
        """Function fails always, should raise exception eventually."""
        mock_func = MagicMock(side_effect=ValueError("Persistent Fail"))
        
        decorated = retry(max_retries=2, delay=0.01)(mock_func)
        
        with pytest.raises(ValueError, match="Persistent Fail"):
            decorated()
        
        # Initial call + 1 retry (since max_retries logic usually means total attempts or retries)
        # Implementation: while mtries > 1 ... mtries -= 1. 
        # If max_retries=2: 
        #   Call 1 (Fail) -> mtries=2 -> loop continues
        #   Sleep -> mtries=1
        #   Call 2 (Fail) -> mtries=1 -> loop terminates
        #   Final Call (Fail) -> Raises
        # Total calls = 2 inside loop? Wait, let's check implementation.
        # Implementation:
        # while mtries > 1:
        #    try: func()
        #    except: sleep; mtries-=1
        # return func() (final attempt)
        # So max_retries=2 means: 1 retry loop + 1 final call = 2 total calls.
        assert mock_func.call_count == 2

    def test_time_execution_logging(self, caplog):
        """Verify time_execution logs the duration."""
        import logging
        
        @time_execution
        def slow_func():
            time.sleep(0.01)
            return "done"
            
        with caplog.at_level(logging.INFO):
            res = slow_func()
            
        assert res == "done"
        # Check log message
        assert "Function 'slow_func' took" in caplog.text
        assert "seconds to execute" in caplog.text