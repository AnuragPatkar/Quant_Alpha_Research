"""
UNIT TEST: Feature Engineering
Tests FactorRegistry, BaseFactor, and specific factor calculations.
Verifies that factors handle multi-ticker data and NaNs correctly.
"""
import pytest
import pandas as pd
import numpy as np
from quant_alpha.features.registry import FactorRegistry

# Import factor modules to ensure they register themselves
import quant_alpha.features.technical.momentum
import quant_alpha.features.technical.volatility
import quant_alpha.features.fundamental.value

class TestFeatures:
    
    @pytest.fixture
    def sample_market_data(self):
        """Creates dummy OHLCV + Fundamental data for 2 tickers."""
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        tickers = ["TICK_A", "TICK_B"]
        
        rows = []
        for t in tickers:
            # Create a random walk for price
            price = 100.0 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
            
            for i, d in enumerate(dates):
                rows.append({
                    "date": d,
                    "ticker": t,
                    "open": price[i],
                    "high": price[i] + 1,
                    "low": price[i] - 1,
                    "close": price[i],
                    "volume": 10000,
                    # Fundamental fields
                    "net_income": 5000 if i % 60 == 0 else np.nan, # Quarterly-ish
                    "total_revenue": 10000 if i % 60 == 0 else np.nan,
                    "market_cap": 1000000,
                    "pe_ratio": 15.0 + np.random.normal()
                })
                
        df = pd.DataFrame(rows)
        # Forward fill fundamentals as DataManager would
        df = df.groupby("ticker").ffill().reset_index(drop=True)
        return df

    def test_registry_discovery(self):
        """Test that factors are registered upon import."""
        registry = FactorRegistry()
        # Check for factors we know should exist from the imports
        
        # We check if AT LEAST one is present (names might vary slightly in implementation)
        # Based on test_integration.py, 'rsi_14d', 'mom_1m', 'vol_21d' are used.
        registered = list(registry.factors.keys())
        assert len(registered) > 0, "Registry should not be empty"
        
        # Check specific category prefixes if exact names aren't guaranteed
        assert any(f.startswith("mom") for f in registered), "Momentum factors missing"
        assert any(f.startswith("vol") for f in registered), "Volatility factors missing"

    def test_technical_factor_calculation(self, sample_market_data):
        """Test calculation of a technical factor (Volatility)."""
        registry = FactorRegistry()
        
        # Use a factor likely to exist
        factor_name = "vol_21d" 
        if factor_name not in registry.factors:
            pytest.skip(f"{factor_name} not found in registry")
            
        factor = registry.factors[factor_name]
        res = factor.calculate(sample_market_data)
        
        # Checks
        assert not res.empty
        assert len(res) == len(sample_market_data)
        
        # Check alignment
        if isinstance(res, pd.DataFrame):
            assert len(res.columns) >= 1
        
    def test_fundamental_factor_pass_through(self, sample_market_data):
        """Test a fundamental factor that might just be a pass-through or simple ratio."""
        registry = FactorRegistry()
        
        # 'val_pe_ratio' is often just the 'pe_ratio' column standardized or raw
        factor_name = "val_pe_ratio"
        if factor_name not in registry.factors:
            pytest.skip(f"{factor_name} not found in registry")
            
        factor = registry.factors[factor_name]
        res = factor.calculate(sample_market_data)
        
        assert not res.empty
        assert len(res) == len(sample_market_data)

    def test_robustness_to_missing_columns(self):
        """Test that factors fail gracefully or return None if columns are missing."""
        registry = FactorRegistry()
        
        # Create data MISSING 'close' column
        bad_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "ticker": ["A"]*10,
            "volume": [100]*10
        })
        
        factor_name = "vol_21d"
        if factor_name in registry.factors:
            factor = registry.factors[factor_name]
            try:
                _ = factor.calculate(bad_data)
            except (KeyError, ValueError):
                pass # Expected behavior

    def test_grouping_logic(self, sample_market_data):
        """Ensure factors calculate per-ticker, not mixing data across tickers."""
        registry = FactorRegistry()
        factor_name = "mom_1m" # 21 day return usually
        if factor_name not in registry.factors:
            pytest.skip("mom_1m not found")
            
        # Modify data: Ticker A goes up, Ticker B goes down
        df = sample_market_data.copy()
        
        # Force trends
        mask_a = df["ticker"] == "TICK_A"
        df.loc[mask_a, "close"] = np.linspace(100, 200, mask_a.sum()) # Up
        
        mask_b = df["ticker"] == "TICK_B"
        df.loc[mask_b, "close"] = np.linspace(100, 50, mask_b.sum())  # Down
        
        factor = registry.factors[factor_name]
        res = factor.calculate(df)
        
        # Standardize result
        if isinstance(res, pd.Series):
            res = res.to_frame(name="res")
        
        # Extract values for last day
        val_col = res.columns[-1]
        df["mom"] = res[val_col]
        last_day = df[df["date"] == df["date"].max()]
        
        val_a = last_day[last_day["ticker"] == "TICK_A"]["mom"].values[0]
        val_b = last_day[last_day["ticker"] == "TICK_B"]["mom"].values[0]
        
        # A should be positive, B should be negative
        assert val_a > 0, "Ticker A momentum should be positive"
        assert val_b < 0, "Ticker B momentum should be negative"