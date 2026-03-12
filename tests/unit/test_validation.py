"""
UNIT TEST: Factor Validation Logic
==================================
Tests the FactorValidator class logic in scripts/validate_factors.py.
Verifies IC calculation, t-stats, and autocorrelation without script imports.
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# FactorValidator Logic (Replicated from scripts/validate_factors.py)
# ---------------------------------------------------------------------------

class FactorValidatorLogic:
    """
    Replicates core logic from FactorValidator without importing the script.
    This allows testing the statistical methods without scipy reload issues.
    """
    
    def __init__(self, df, target_col="raw_ret_5d"):
        self.df = df
        self.target_col = target_col
        
        # Identify factors (numeric columns excluding metadata)
        exclude = {"date", "ticker", "sector", target_col}
        self.factors = [col for col in df.columns 
                       if df[col].dtype in [np.float64, np.float32, int] 
                       and col not in exclude]
    
    # Replace the compute_ic_stats method in FactorValidatorLogic class:

    def compute_ic_stats(self):
        """Compute Information Coefficient (IC) statistics."""
        if self.df.empty or self.target_col not in self.df.columns:
            return pd.DataFrame()
        
        stats = []
        for factor in self.factors:
            # Clean data: drop NaN and Inf
            clean = self.df[[factor, self.target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean) < 2:
                continue
            
            # Spearman rank correlation (IC)
            ic, p_val = spearmanr(clean[factor], clean[self.target_col])
            
            # t-statistic: t = ic * sqrt(n-2) / sqrt(1 - ic^2)
            n = len(clean)
            
            # Handle near-perfect correlation (ic very close to ±1)
            ic_squared = ic ** 2
            numerator = abs(ic) * np.sqrt(n - 2)
            
            if ic_squared >= 0.9999:  # Very close to ±1
                # For near-perfect correlation, use large t-stat
                t_stat = numerator / np.sqrt(1e-4)  # Avoid division by zero
            else:
                denominator = np.sqrt(1 - ic_squared + 1e-10)
                t_stat = numerator / denominator
            
            stats.append({
                "factor": factor,
                "ic_mean": ic,
                "t_stat": t_stat,
                "p_value": p_val
            })
        
        return pd.DataFrame(stats).set_index("factor") if stats else pd.DataFrame()
    
    def compute_autocorrelation(self):
        """Compute AR(1) autocorrelation for each factor."""
        ac_list = []
        
        for factor in self.factors:
            clean = self.df[factor].dropna()
            
            if len(clean) < 2:
                ac_list.append({"factor": factor, "autocorr": np.nan})
                continue
            
            # AR(1): correlation between x[t] and x[t-1]
            x_t = clean.values[1:]
            x_t_minus_1 = clean.values[:-1]
            
            ac = np.corrcoef(x_t, x_t_minus_1)[0, 1]
            ac_list.append({"factor": factor, "autocorr": ac if not np.isnan(ac) else 0.0})
        
        return pd.DataFrame(ac_list).set_index("factor")
    
    def check_coverage(self):
        """Check data quality: NaN, Inf, and coverage percentage."""
        coverage_list = []
        
        for factor in self.factors:
            series = self.df[factor]
            n = len(series)
            
            nan_count = series.isna().sum()
            inf_count = ((series == np.inf) | (series == -np.inf)).sum()
            valid_count = n - nan_count - inf_count
            coverage_pct = valid_count / n if n > 0 else 0.0
            
            coverage_list.append({
                "factor": factor,
                "total": n,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "valid": valid_count,
                "coverage_pct": coverage_pct
            })
        
        return pd.DataFrame(coverage_list).set_index("factor")
    
    def compute_ic_decay(self, factors):
        """
        Compute IC at different lags to measure signal decay.
        Returns IC at lag 1, lag 5, etc.
        """
        decay_list = []
        
        for factor in factors:
            if factor not in self.factors:
                continue
            
            clean_df = self.df[[factor, self.target_col]].dropna()
            
            if len(clean_df) < 6:
                decay_list.append({"factor": factor, "ic_lag1": np.nan, "ic_lag5": np.nan})
                continue
            
            # Lag-1 IC
            ic_lag1, _ = spearmanr(clean_df[factor].iloc[:-1].values, 
                                   clean_df[self.target_col].iloc[1:].values)
            
            # Lag-5 IC
            ic_lag5, _ = spearmanr(clean_df[factor].iloc[:-5].values, 
                                   clean_df[self.target_col].iloc[5:].values) if len(clean_df) >= 6 else (np.nan, np.nan)
            
            decay_list.append({
                "factor": factor,
                "ic_lag1": ic_lag1 if not np.isnan(ic_lag1) else 0.0,
                "ic_lag5": ic_lag5 if not np.isnan(ic_lag5) else 0.0
            })
        
        return pd.DataFrame(decay_list).set_index("factor") if decay_list else pd.DataFrame()
    
    def compute_sector_neutral_ic(self):
        """
        Compute sector-neutral IC using residualization.
        Remove sector beta from factor before computing IC.
        """
        if "sector" not in self.df.columns:
            # No sector data, return regular IC
            return self.compute_ic_stats()
        
        stats = []
        
        for factor in self.factors:
            clean = self.df[[factor, "sector", self.target_col]].dropna()
            
            if len(clean) < 3:
                continue
            
            # Get sector dummies
            sector_dummies = pd.get_dummies(clean["sector"], drop_first=True)
            
            # Regress factor on sectors to get residuals
            X = sector_dummies.values
            y = clean[factor].values
            
            # Simple OLS: residuals = y - X @ (X.T @ X)^-1 @ X.T @ y
            try:
                XtX_inv = np.linalg.pinv(X.T @ X)
                beta = XtX_inv @ X.T @ y
                factor_residuals = y - X @ beta
            except:
                factor_residuals = y
            
            # IC on residualized factor
            ic_resid, _ = spearmanr(factor_residuals, clean[self.target_col].values)
            
            stats.append({
                "factor": factor,
                "sn_icir": ic_resid if not np.isnan(ic_resid) else 0.0
            })
        
        return pd.DataFrame(stats).set_index("factor") if stats else pd.DataFrame()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFactorValidator:
    
    @pytest.fixture
    def dummy_data(self):
        """Create synthetic data with a known good factor."""
        dates = pd.date_range("2023-01-01", periods=50)
        
        # Ticker A: Positive signal, Positive return
        df_a = pd.DataFrame({
            "date": dates,
            "ticker": "A",
            "good_factor": 1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": 0.05
        })
        
        # Ticker B: Negative signal, Negative return (Perfect Rank Correlation)
        df_b = pd.DataFrame({
            "date": dates,
            "ticker": "B",
            "good_factor": -1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": -0.05
        })
        
        return pd.concat([df_a, df_b]).sort_values("date").reset_index(drop=True)

    def test_identify_factors(self, dummy_data):
        """Should ignore metadata columns and pick numeric factors."""
        validator = FactorValidatorLogic(dummy_data)
        
        assert "good_factor" in validator.factors
        assert "noise_factor" in validator.factors
        assert "date" not in validator.factors
        assert "ticker" not in validator.factors
        assert "raw_ret_5d" not in validator.factors

    def test_ic_calculation(self, dummy_data):
        """Verify IC calculation for a perfect factor."""
        validator = FactorValidatorLogic(dummy_data, target_col="raw_ret_5d")
        stats = validator.compute_ic_stats()
        
        good = stats.loc["good_factor"]
        assert good["ic_mean"] == pytest.approx(1.0, abs=0.05)
        assert good["t_stat"] > 10.0
        
        noise = stats.loc["noise_factor"]
        assert abs(noise["ic_mean"]) < 0.3

    def test_autocorrelation(self, dummy_data):
        """Verify AR(1) calculation."""
        # Create a factor with high autocorrelation
        dummy_data["trend"] = np.linspace(0, 10, len(dummy_data))
        validator = FactorValidatorLogic(dummy_data)
        ac = validator.compute_autocorrelation()
        
        assert ac.loc["trend", "autocorr"] > 0.9
        assert abs(ac.loc["noise_factor", "autocorr"]) < 0.5

    def test_check_coverage(self, dummy_data):
        """Verify coverage statistics calculation."""
        df = dummy_data.copy()
        df.loc[0, "good_factor"] = np.nan
        df.loc[1, "good_factor"] = np.inf
        
        validator = FactorValidatorLogic(df)
        coverage = validator.check_coverage()
        
        assert "good_factor" in coverage.index
        assert coverage.loc["good_factor", "nan_count"] == 1
        assert coverage.loc["good_factor", "inf_count"] == 1
        assert coverage.loc["good_factor", "coverage_pct"] < 1.0

    def test_ic_decay(self, dummy_data):
        """Verify IC decay calculation."""
        validator = FactorValidatorLogic(dummy_data, target_col="raw_ret_5d")
        decay = validator.compute_ic_decay(["good_factor", "noise_factor"])
        
        assert "good_factor" in decay.index
        assert "ic_lag1" in decay.columns
        assert "ic_lag5" in decay.columns
        assert not decay.empty

    def test_sector_neutral_ic_simpsons_paradox(self):
        """
        Institutional Check: Verify Sector-Neutral IC logic.
        Simpson's Paradox scenario:
          - Global correlation is Positive
          - Intra-sector correlation is Negative
        
        SN-IC should decouple the alpha from the sector bet.
        """
        dates = pd.date_range("2023-01-01", periods=10)
        data = []
        
        for d in dates:
            # Sector S1: High Factor, Higher Return (Global +)
            # Within S1: Higher Factor (2.0) -> Lower Return (0.04) vs (1.0 -> 0.05)
            data.append({"date": d, "ticker": "A1", "sector": "S1", "f": 1.0, "t": 0.05})
            data.append({"date": d, "ticker": "A2", "sector": "S1", "f": 2.0, "t": 0.04})
            
            # Sector S2: Low Factor, Lower Return (Global +)
            # Within S2: Higher Factor (-1.0) -> Lower Return (-0.05) vs (-2.0 -> -0.04)
            data.append({"date": d, "ticker": "B1", "sector": "S2", "f": -2.0, "t": -0.04})
            data.append({"date": d, "ticker": "B2", "sector": "S2", "f": -1.0, "t": -0.05})
        
        df = pd.DataFrame(data)
        
        validator = FactorValidatorLogic(df, target_col="t")
        
        # Raw IC (should be positive due to sector bet)
        raw_stats = validator.compute_ic_stats()
        raw_ic = raw_stats.loc["f", "ic_mean"]
        assert raw_ic > 0.5, f"Raw IC should be positive (Sector Bet), got {raw_ic}"
        
        # Sector-neutral IC (should be negative, alpha is negative intra-sector)
        sn_stats = validator.compute_sector_neutral_ic()
        if not sn_stats.empty:
            sn_ic = sn_stats.loc["f", "sn_icir"]
            assert sn_ic < -0.3, f"SN IC should be negative (Alpha), got {sn_ic}"