"""
Incremental Data Ingestion Validation Suite
===========================================
Validates the temporal delta-patching and merge deduplication logic for the Data Lake.

Purpose
-------
This module serves as the primary test suite for the `update_data` orchestration script. 
It rigorously evaluates the incremental update procedures for asset pricing, fundamentals, 
and macroeconomic indicators, ensuring that overlapping data boundaries do not induce 
duplicate indices or look-ahead biases.

Role in Quantitative Workflow
-----------------------------
Acts as an automated data quality gatekeeper, guaranteeing that the "Update-in-Place" 
operations correctly identify temporal gaps, resolve stale schema states, and gracefully 
handle missing or corrupt files prior to factor construction.

Dependencies
------------
- **Pytest**: Test execution and synthetic fixture orchestration.
- **Pandas**: Structural mock generation and boundary deduplication assertions.
- **Unittest.Mock**: Extensively patches external APIs (e.g., YFinance) to isolate 
  the internal logic and guarantee deterministic evaluations.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
import importlib
from unittest.mock import patch, MagicMock
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

@pytest.fixture
def update_data_module():
    """
    Provisions a safely isolated environment for evaluating the update_data script.

    Intercepts sys.modules to inject mocked structural dependencies and global 
    configurations prior to importing the target module, ensuring no inadvertent 
    pollution or physical IO occurs on the host filesystem.

    Yields:
        module: The dynamically imported and constrained `scripts.update_data` module.
    """
    with patch.dict(sys.modules):
        m = MagicMock()
        m.__path__ = []
        sys.modules["quant_alpha.utils"] = m
        sys.modules["download_data"] = MagicMock()

        mock_cfg = MagicMock()
        mock_cfg.PRICES_DIR = Path("prices")
        mock_cfg.FUNDAMENTALS_DIR = Path("fundamentals")
        mock_cfg.EARNINGS_DIR = Path("earnings")
        mock_cfg.ALTERNATIVE_DIR = Path("alternative")
        sys.modules["config.settings"] = MagicMock(config=mock_cfg)

        import scripts.update_data
        yield scripts.update_data

class TestDataUpdates:
    """
    Isolated validation suite mapping temporal boundaries and ingestion protocols.
    """

    def test_price_merge_logic(self, update_data_module, tmp_path):
        """
        Validates the strict idempotency of the incremental price merge logic.

        Ensures that when upstream data contains overlapping temporal windows 
        (e.g., due to fetching padding), the system accurately deduplicates 
        by date and retains only the most recent structurally sound updates.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        # Synthesizes an existing on-disk historical archive prior to the update operation
        csv_path = tmp_path / "AAPL.csv"
        old_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "close": [100.0, 101.0],
            "volume": [1000, 1100]
        })
        old_data.to_csv(csv_path, index=False)
        
        # Simulates upstream API response containing overlapping historical indices to validate deduplication bounds
        new_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "close": [101.0, 102.0],
            "volume": [1100, 1200]
        }).set_index("date")
        
        # Overrides global temporal bounds to force the evaluation window
        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            
            with patch("yfinance.download", return_value=new_data):
                status = update_data_module._update_price_ticker(csv_path, today=date(2023, 1, 4))
            
        assert status == "updated"
        
        # Asserts strict topological adherence: 3 contiguous records with the overlapping boundary deduplicated
        result = pd.read_csv(csv_path)
        assert len(result) == 3
        assert result.iloc[-1]["close"] == 102.0
        assert len(result[result["date"] == "2023-01-02"]) == 1

    def test_uptodate_check(self, update_data_module, tmp_path):
        """
        Verifies the optimal short-circuit routing when data caches are already fresh.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        csv_path = tmp_path / "MSFT.csv"
        today = date(2023, 1, 4)
        
        # Establishes a contiguous historical block meeting the target temporal threshold
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2023-01-04"]),
            "close": [100.0, 200.0],
            "volume": [1000, 1000]
        })
        df.to_csv(csv_path, index=False)
        
        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            # Asserts that the network interface is entirely bypassed when caching thresholds are satisfied
            with patch("yfinance.download") as mock_download:
                status = update_data_module._update_price_ticker(csv_path, today=today)
                mock_download.assert_not_called()
            
        assert status == "uptodate"

    def test_missing_history_trigger(self, update_data_module, tmp_path):
        """
        Verifies that deep historical voids trigger targeted retroactive backfill downloads.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        csv_path = tmp_path / "GOOGL.csv"
        # Provisions an incomplete historical boundary missing the critical inception window
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01", "2023-01-04"]),
            "close": [100.0, 150.0],
            "volume": [1000, 1000]
        })
        df.to_csv(csv_path, index=False)

        # Synthesizes the required retroactive payload mapping to the missing timeframe
        hist_data = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2021-12-31"]),
            "close": [50.0, 90.0],
            "volume": [500, 500]
        }).set_index("date")

        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            
            # Asserts proper parameter configuration during the retroactive backfill execution
            with patch("yfinance.download", return_value=hist_data) as mock_download:
                status = update_data_module._update_price_ticker(csv_path, today=date(2023, 1, 4))
                
                assert status == "updated"
                args, kwargs = mock_download.call_args
                assert kwargs['start'] == "2020-01-01"

    def test_corrupt_csv_handling(self, update_data_module, tmp_path):
        """
        Validates the resilient handling of structurally invalid or zero-byte caching artifacts.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        csv_path = tmp_path / "BAD.csv"
        csv_path.touch() 

        status = update_data_module._update_price_ticker(csv_path, today=date(2023, 1, 1))
        assert status == "error"

    def test_earnings_staleness_logic(self, update_data_module, tmp_path):
        """
        Validates the temporal staleness audit logic for earnings event datasets.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        # Binds execution boundaries to an ephemeral path to prevent physical modifications
        with patch("scripts.update_data.EARNINGS_DIR", tmp_path):
            ticker = "NVDA"
            
            # Asserts missing artifacts universally trigger mandatory update cycles
            assert update_data_module._earnings_needs_update(ticker) is True
            
            # Asserts historical event caches lacking forward projections require updates
            csv_path = tmp_path / f"{ticker}.csv"
            old_dates = pd.DataFrame({
                "date": pd.to_datetime(["2020-01-01", "2021-01-01"])
            })
            old_dates.to_csv(csv_path, index=False)
            assert update_data_module._earnings_needs_update(ticker) is True

    def test_macro_update_logic(self, update_data_module, tmp_path):
        """
        Verifies the discrete update and initialization mechanisms for systemic macro indicators.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        # Binds execution boundaries to an ephemeral path to prevent physical modifications
        with patch("scripts.update_data.ALT_DIR", tmp_path):
            ticker = "SPY"
            name = "sp500"
            
            # Validates the primary acquisition flow utilizing the deterministic dd wrapper
            mock_hist = MagicMock()
            mock_hist.history.return_value = pd.DataFrame({
                "Close": [100.0, 101.0],
                "Volume": [1000, 1200]
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
            
            # Injects upstream network shims
            with patch("scripts.update_data.dd") as mock_dd:
                mock_dd._yf_ticker.return_value = mock_hist
                mock_dd._retry.side_effect = lambda f, **k: f()
                
                status = update_data_module._update_macro_series(name, ticker, date(2023, 1, 3))
            
            assert status == "updated", f"Expected 'updated', got '{status}'"
            
            csv_file = tmp_path / f"{name}.csv"
            assert csv_file.exists(), f"Expected {csv_file} to exist"
            
            result_df = pd.read_csv(csv_file)
            assert len(result_df) == 2, f"Expected 2 rows, got {len(result_df)}"
            assert "sp500_close" in result_df.columns, f"Expected 'sp500_close' column, got {result_df.columns.tolist()}"
            
            # Asserts strict operational bypassing for active caching bounds
            df = pd.DataFrame({
                "date": ["2023-01-03"],
                "sp500_close": [102.0],
                "sp500_volume": [1200]
            })
            df.to_csv(tmp_path / f"{name}.csv", index=False)
            
            status = update_data_module._update_macro_series(name, ticker, date(2023, 1, 3))
            assert status == "uptodate"
            
            # Validates alternative resolution protocols when primary ingestion adapters fault
            mock_yf_ticker = MagicMock()
            mock_yf_ticker.history.return_value = pd.DataFrame({
                "Close": [103.0, 104.0],
                "Volume": [1300, 1400]
            }, index=pd.to_datetime(["2023-01-04", "2023-01-05"]))
            
            with patch("scripts.update_data.dd", None):  # Disable dd
                with patch("scripts.update_data.yf") as mock_yf:
                    mock_yf.Ticker.return_value = mock_yf_ticker
                    
                    # Remove the old file to trigger full download
                    (tmp_path / f"{name}.csv").unlink()
                    
                    status = update_data_module._update_macro_series(
                        name, 
                        ticker, 
                        date(2023, 1, 6)
                    )
                    
                    assert status == "updated", f"Expected 'updated' with yfinance, got '{status}'"
                    assert (tmp_path / f"{name}.csv").exists()
            
            mock_yf_ticker = MagicMock()
            mock_yf_ticker.history.return_value = pd.DataFrame({
                "Close": [103.0, 104.0],
                "Volume": [1300, 1400]
            }, index=pd.to_datetime(["2023-01-04", "2023-01-05"]))
            
            with patch("scripts.update_data.dd", None):  # Disable dd
                with patch("scripts.update_data.yf") as mock_yf:
                    mock_yf.Ticker.return_value = mock_yf_ticker
                    
                    (tmp_path / f"{name}.csv").unlink()
                    
                    status = update_data_module._update_macro_series(
                        name, 
                        ticker, 
                        date(2023, 1, 6)
                    )
                    
                    assert status == "updated", f"Expected 'updated' with yfinance, got '{status}'"
                    assert (tmp_path / f"{name}.csv").exists()

    def test_fundamentals_staleness(self, update_data_module, tmp_path):
        """
        Verifies fundamental staleness logic across discrete financial statements.

        Args:
            update_data_module (module): The isolated target script.
            tmp_path (pathlib.Path): Pytest fixture providing an ephemeral directory.

        Returns:
            None
        """
        with patch("scripts.update_data.FUND_DIR", tmp_path):
            ticker = "AAPL"
            ticker_dir = tmp_path / ticker
            ticker_dir.mkdir()
            
            # Asserts foundational metadata absence mathematically mandates a refresh
            assert update_data_module._info_age_days(ticker) == 9999
            
            # Asserts partial corporate statement availability triggers an update
            assert update_data_module._fund_data_incomplete(ticker) is True
            
            # Provisions a fully contiguous and populated structural directory
            for fname in ["financials.csv", "balance_sheet.csv", "cashflow.csv"]:
                # Simulates transposed temporal columns typical of upstream fundamental APIs
                df = pd.DataFrame({"2023-09-30": [1], "2022-09-30": [1]}, index=["Revenue"])
                df.to_csv(ticker_dir / fname)
            
            assert update_data_module._fund_data_incomplete(ticker) is False
