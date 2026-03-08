"""
UNIT TEST: Data Update Logic
============================
Tests the incremental update logic in scripts/update_data.py.
Verifies that new data is correctly merged with existing CSVs without duplication.
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Mock dependencies
if "quant_alpha.utils" not in sys.modules:
    sys.modules["quant_alpha.utils"] = MagicMock()
if "download_data" not in sys.modules:
    sys.modules["download_data"] = MagicMock()

from scripts import update_data

class TestDataUpdates:

    def test_price_merge_logic(self, tmp_path):
        """
        Verify _update_price_ticker correctly merges new data 
        and removes duplicates.
        """
        # 1. Create existing CSV (Old Data)
        csv_path = tmp_path / "AAPL.csv"
        old_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "close": [100.0, 101.0],
            "volume": [1000, 1100]
        })
        old_data.to_csv(csv_path, index=False)
        
        # 2. Mock yfinance to return overlapping + new data
        # Overlap: 2023-01-02 (should dedupe)
        # New: 2023-01-03
        new_data = pd.DataFrame({
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "close": [101.0, 102.0],
            "volume": [1100, 1200]
        }).set_index("date") # yf returns index
        
        # Mock config
        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            
            with patch("yfinance.download", return_value=new_data):
                # Run update
                status = update_data._update_price_ticker(csv_path, today=date(2023, 1, 4))
            
        assert status == "updated"
        
        # 3. Verify Result
        result = pd.read_csv(csv_path)
        assert len(result) == 3 # 01, 02, 03
        assert result.iloc[-1]["close"] == 102.0
        # Check deduplication of Jan 02
        assert len(result[result["date"] == "2023-01-02"]) == 1

    def test_uptodate_check(self, tmp_path):
        """Verify it skips update if data is already fresh."""
        csv_path = tmp_path / "MSFT.csv"
        today = date(2023, 1, 4)
        
        # Data covering the full required range
        # Start: 2020-01-01 (matches config)
        # End:   2023-01-04 (matches today)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2023-01-04"]),
            "close": [100.0, 200.0],
            "volume": [1000, 1000]
        })
        df.to_csv(csv_path, index=False)
        
        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            # Ensure yfinance is NOT called (if logic works, it returns early)
            with patch("yfinance.download") as mock_download:
                status = update_data._update_price_ticker(csv_path, today=today)
                mock_download.assert_not_called()
            
        assert status == "uptodate"

    def test_missing_history_trigger(self, tmp_path):
        """Verify it triggers download if history is missing (start date gap)."""
        csv_path = tmp_path / "GOOGL.csv"
        # File starts late (2022), config wants 2020
        df = pd.DataFrame({
            "date": pd.to_datetime(["2022-01-01", "2023-01-04"]),
            "close": [100.0, 150.0],
            "volume": [1000, 1000]
        })
        df.to_csv(csv_path, index=False)

        # Mock history download return
        hist_data = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2021-12-31"]),
            "close": [50.0, 90.0],
            "volume": [500, 500]
        }).set_index("date")

        with patch("scripts.update_data.config") as mock_config:
            mock_config.BACKTEST_START_DATE = "2020-01-01"
            
            # We expect yf.download to be called for history
            with patch("yfinance.download", return_value=hist_data) as mock_download:
                # Set today same as last date to avoid triggering 'recent update' logic
                # This ensures only the history download is called, making assertion simple.
                status = update_data._update_price_ticker(csv_path, today=date(2023, 1, 4))
                
                # Should return updated
                assert status == "updated"
                # Verify download called with correct start date
                args, kwargs = mock_download.call_args
                assert kwargs['start'] == "2020-01-01"

    def test_corrupt_csv_handling(self, tmp_path):
        """Verify it returns 'error' for empty/corrupt CSVs."""
        csv_path = tmp_path / "BAD.csv"
        csv_path.touch() # Empty file

        status = update_data._update_price_ticker(csv_path, today=date(2023, 1, 1))
        assert status == "error"

    def test_earnings_staleness_logic(self, tmp_path):
        """Verify _earnings_needs_update logic handles missing/stale files."""
        # Patch the module-level EARNINGS_DIR variable
        with patch("scripts.update_data.EARNINGS_DIR", tmp_path):
            ticker = "NVDA"
            
            # Case 1: File missing -> Needs update
            assert update_data._earnings_needs_update(ticker) is True
            
            # Case 2: File exists but old (no future dates)
            csv_path = tmp_path / f"{ticker}.csv"
            old_dates = pd.DataFrame({
                "date": pd.to_datetime(["2020-01-01", "2021-01-01"])
            })
            old_dates.to_csv(csv_path, index=False)
            assert update_data._earnings_needs_update(ticker) is True

    def test_macro_update_logic(self, tmp_path):
        """Verify macro data update logic (append vs full download)."""
        # Mock ALT_DIR
        with patch("scripts.update_data.ALT_DIR", tmp_path):
            ticker = "SPY"
            name = "sp500"
            
            # Case 1: New file (Full Download)
            mock_hist = MagicMock()
            mock_hist.history.return_value = pd.DataFrame({
                "Close": [100.0, 101.0],
                "Volume": [1000, 1200]
            }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
            
            # Mock dd._yf_ticker and dd._retry inside update_data
            with patch("scripts.update_data.dd") as mock_dd:
                mock_dd._yf_ticker.return_value = mock_hist
                mock_dd._retry.side_effect = lambda f, **k: f()
                
                status = update_data._update_macro_series(name, ticker, date(2023, 1, 3))
            
            assert status == "updated"
            assert (tmp_path / "sp500.csv").exists()
            
            # Case 2: Existing file, up to date
            # Create file with data up to today (2023-01-03)
            df = pd.DataFrame({
                "date": ["2023-01-03"],
                "sp500_close": [102.0]
            })
            df.to_csv(tmp_path / "sp500.csv", index=False)
            
            status = update_data._update_macro_series(name, ticker, date(2023, 1, 3))
            assert status == "uptodate"

    def test_fundamentals_staleness(self, tmp_path):
        """Verify fundamental staleness logic."""
        with patch("scripts.update_data.FUND_DIR", tmp_path):
            ticker = "AAPL"
            ticker_dir = tmp_path / ticker
            ticker_dir.mkdir()
            
            # Case 1: info.csv missing
            assert update_data._info_age_days(ticker) == 9999
            
            # Case 2: Incomplete data (missing files)
            assert update_data._fund_data_incomplete(ticker) is True
            
            # Case 3: Complete data (files exist with recent dates)
            for fname in ["financials.csv", "balance_sheet.csv", "cashflow.csv"]:
                # Create dummy csv with recent dates as columns (FMP format)
                df = pd.DataFrame({"2023-09-30": [1], "2022-09-30": [1]}, index=["Revenue"])
                df.to_csv(ticker_dir / fname)
            
            # Should be complete
            assert update_data._fund_data_incomplete(ticker) is False
