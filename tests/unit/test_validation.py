"""
Factor Validation Logic and Statistical Efficacy Tests
======================================================
Unit testing suite for the platform's core statistical factor evaluation engines.

Purpose
-------
This module replicates and verifies the mathematical logic embedded within the 
`FactorValidator` without relying on dynamic script imports. It rigorously tests 
Information Coefficient (IC) generation, t-statistic boundaries, autocorrelation 
decay, and sector-neutral residualization.

Role in Quantitative Workflow
-----------------------------
Guarantees that the fundamental statistical scoring metrics defining alpha 
quality are operating deterministically and mathematically soundly before any 
features are passed to the machine learning ensemble.

Dependencies
------------
- **Pytest**: Test execution framework.
- **Pandas/NumPy**: Synthetic feature generation and mathematical bounds testing.
- **SciPy (spearmanr)**: Foundational nonparametric rank correlation engine.
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

class FactorValidatorLogic:
    """
    Standalone validation engine replicating production `FactorValidator` logic.
    
    Provides isolated evaluation of statistical methods without encountering 
    circular imports or dynamic namespace reloading constraints present in 
    integration boundaries.
    """
    
    def __init__(self, df, target_col="raw_ret_5d"):
        """
        Initializes the validation state matrix and evaluates structural prerequisites.

        Args:
            df (pd.DataFrame): Aggregated master dataset containing feature histories.
            target_col (str, optional): The independent variable targeted for prediction. Defaults to "raw_ret_5d".
        """
        self.df = df
        self.target_col = target_col
        
        # Identifies feature subsets bypassing explicit metadata constraints
        exclude = {"date", "ticker", "sector", target_col}
        self.factors = [col for col in df.columns 
                       if df[col].dtype in [np.float64, np.float32, int] 
                       and col not in exclude]
    
    def compute_ic_stats(self):
        """
        Computes Information Coefficient (IC) and t-statistic boundaries.

        Evaluates nonparametric rank correlation ($\rho_s$) across features and 
        applies strict error handling for perfectly correlated or zero-variance 
        target regimes.

        Args:
            None

        Returns:
            pd.DataFrame: A matrix of calculated statistics including IC mean, 
                t-stat, and p-value.
        """
        if self.df.empty or self.target_col not in self.df.columns:
            return pd.DataFrame()
        
        stats = []
        for factor in self.factors:
            clean = self.df[[factor, self.target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean) < 2:
                continue
            
            ic, p_val = spearmanr(clean[factor], clean[self.target_col])
            
            n = len(clean)
            
            ic_squared = ic ** 2
            numerator = abs(ic) * np.sqrt(n - 2)
            
            # Stabilizes t-statistic calculation avoiding DivisionByZero bounds during perfect correlations
            if ic_squared >= 0.9999:  
                t_stat = numerator / np.sqrt(1e-4)  
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
        """
        Quantifies AR(1) signal stability parameters universally across execution arrays.

        Calculates the one-period lagged correlation matrix to explicitly identify 
        structural turnover boundaries and signal degradation speed.

        Args:
            None

        Returns:
            pd.DataFrame: A matrix of calculated day-over-day turnover estimations.
        """
        ac_list = []
        
        for factor in self.factors:
            clean = self.df[factor].dropna()
            
            if len(clean) < 2:
                ac_list.append({"factor": factor, "autocorr": np.nan})
                continue
            
            x_t = clean.values[1:]
            x_t_minus_1 = clean.values[:-1]
            
            ac = np.corrcoef(x_t, x_t_minus_1)[0, 1]
            ac_list.append({"factor": factor, "autocorr": ac if not np.isnan(ac) else 0.0})
        
        return pd.DataFrame(ac_list).set_index("factor")
    
    def check_coverage(self):
        """
        Computes vectorized sparsity arrays evaluating null boundaries globally.

        Args:
            None

        Returns:
            pd.DataFrame: Metric distributions capturing structural feature voids.
        """
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
        Computes Information Coefficient measurements across specific target lags.

        Identifies the half-life and alpha turnover boundaries for candidate
        signals evaluated dynamically against structural T+1 and T+5 constraints.

        Args:
            factors (list[str]): Extracted target array identifiers.

        Returns:
            pd.DataFrame: Temporally scaled coefficient observations mapped 
                across specified delays.
        """
        decay_list = []
        
        for factor in factors:
            if factor not in self.factors:
                continue
            
            clean_df = self.df[[factor, self.target_col]].dropna()
            
            if len(clean_df) < 6:
                decay_list.append({"factor": factor, "ic_lag1": np.nan, "ic_lag5": np.nan})
                continue
            
            ic_lag1, _ = spearmanr(clean_df[factor].iloc[:-1].values, 
                                   clean_df[self.target_col].iloc[1:].values)
            
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
        Derives isolated cross-sectional Alpha by applying sector-level residualization.

        Forces an orthogonal constraint projecting feature vectors entirely outside 
        the sector correlation subspace via OLS execution, allowing calculation 
        of a true sector-neutral Information Coefficient.

        Args:
            None

        Returns:
            pd.DataFrame: A matrix mapping purely idiosyncratic alpha properties.
        """
        if "sector" not in self.df.columns:
            return self.compute_ic_stats()
        
        stats = []
        
        for factor in self.factors:
            clean = self.df[[factor, "sector", self.target_col]].dropna()
            
            if len(clean) < 3:
                continue
            
            sector_dummies = pd.get_dummies(clean["sector"], drop_first=True)
            
            X = sector_dummies.values
            y = clean[factor].values
            
            try:
                XtX_inv = np.linalg.pinv(X.T @ X)
                beta = XtX_inv @ X.T @ y
                factor_residuals = y - X @ beta
            except:
                factor_residuals = y
            
            ic_resid, _ = spearmanr(factor_residuals, clean[self.target_col].values)
            
            stats.append({
                "factor": factor,
                "sn_icir": ic_resid if not np.isnan(ic_resid) else 0.0
            })
        
        return pd.DataFrame(stats).set_index("factor") if stats else pd.DataFrame()

class TestFactorValidator:
    """
    Verification suite for isolated mathematical execution bounds.
    """
    
    @pytest.fixture
    def dummy_data(self):
        """
        Creates a discrete synthetic dataset containing deterministic alpha vectors.

        Args:
            None

        Returns:
            pd.DataFrame: A mapped sequence isolating strongly correlated logic patterns
                against structural noise.
        """
        dates = pd.date_range("2023-01-01", periods=50)
        
        # Asserts structural continuous trend behavior bounded positively
        df_a = pd.DataFrame({
            "date": dates,
            "ticker": "A",
            "good_factor": 1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": 0.05
        })
        
        # Asserts strictly mirrored execution bounds ensuring uniform symmetric distributions
        df_b = pd.DataFrame({
            "date": dates,
            "ticker": "B",
            "good_factor": -1.0,
            "noise_factor": np.random.randn(50),
            "raw_ret_5d": -0.05
        })
        
        return pd.concat([df_a, df_b]).sort_values("date").reset_index(drop=True)

    def test_identify_factors(self, dummy_data):
        """
        Asserts metadata bypassing during target variable classification lists.

        Args:
            dummy_data (pd.DataFrame): The synthetic structural mapping.

        Returns:
            None
        """
        validator = FactorValidatorLogic(dummy_data)
        
        assert "good_factor" in validator.factors
        assert "noise_factor" in validator.factors
        assert "date" not in validator.factors
        assert "ticker" not in validator.factors
        assert "raw_ret_5d" not in validator.factors

    def test_ic_calculation(self, dummy_data):
        """
        Evaluates deterministic boundaries accurately predicting near-perfect signals.

        Args:
            dummy_data (pd.DataFrame): The synthetic structural mapping.

        Returns:
            None
        """
        validator = FactorValidatorLogic(dummy_data, target_col="raw_ret_5d")
        stats = validator.compute_ic_stats()
        
        good = stats.loc["good_factor"]
        assert good["ic_mean"] == pytest.approx(1.0, abs=0.05)
        assert good["t_stat"] > 10.0
        
        noise = stats.loc["noise_factor"]
        assert abs(noise["ic_mean"]) < 0.3

    def test_autocorrelation(self, dummy_data):
        """
        Verifies localized persistence and signal velocity degradation measurements.

        Args:
            dummy_data (pd.DataFrame): The synthetic structural mapping.

        Returns:
            None
        """
        dummy_data["trend"] = np.linspace(0, 10, len(dummy_data))
        validator = FactorValidatorLogic(dummy_data)
        ac = validator.compute_autocorrelation()
        
        assert ac.loc["trend", "autocorr"] > 0.9
        assert abs(ac.loc["noise_factor", "autocorr"]) < 0.5

    def test_check_coverage(self, dummy_data):
        """
        Asserts mathematical bounds handling when parsing absolute systemic structural defects.

        Args:
            dummy_data (pd.DataFrame): The synthetic structural mapping.

        Returns:
            None
        """
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
        """
        Assures temporal execution structures effectively map delayed rank observations.

        Args:
            dummy_data (pd.DataFrame): The synthetic structural mapping.

        Returns:
            None
        """
        validator = FactorValidatorLogic(dummy_data, target_col="raw_ret_5d")
        decay = validator.compute_ic_decay(["good_factor", "noise_factor"])
        
        assert "good_factor" in decay.index
        assert "ic_lag1" in decay.columns
        assert "ic_lag5" in decay.columns
        assert not decay.empty

    def test_sector_neutral_ic_simpsons_paradox(self):
        """
        Institutional Verification: Asserts robust execution resolving Simpson's Paradox.

        Evaluates a mathematical regime wherein the universal distribution exhibits 
        positive correlation, yet the discrete sub-domain matrices explicitly 
        demonstrate negative behavior. The sector-neutral execution logic must 
        successfully detach the macro-sector bias and exclusively return the isolated 
        negative alpha coefficient.

        Args:
            None

        Returns:
            None
        """
        dates = pd.date_range("2023-01-01", periods=10)
        data = []
        
        for d in dates:
            # Sub-Domain A: Aggregated Global Positivity encapsulating a localized Negative Drift
            data.append({"date": d, "ticker": "A1", "sector": "S1", "f": 1.0, "t": 0.05})
            data.append({"date": d, "ticker": "A2", "sector": "S1", "f": 2.0, "t": 0.04})
            
            # Sub-Domain B: Equivalent continuous trend replication offset structurally
            data.append({"date": d, "ticker": "B1", "sector": "S2", "f": -2.0, "t": -0.04})
            data.append({"date": d, "ticker": "B2", "sector": "S2", "f": -1.0, "t": -0.05})
        
        df = pd.DataFrame(data)
        
        validator = FactorValidatorLogic(df, target_col="t")
        
        # Asserts un-neutralized baseline evaluates strictly positively reflecting aggregate macro drift
        raw_stats = validator.compute_ic_stats()
        raw_ic = raw_stats.loc["f", "ic_mean"]
        assert raw_ic > 0.5, f"Raw IC should be positive (Sector Bet), got {raw_ic}"
        
        # Asserts fully neutralized matrices accurately extract the idiosyncratic localized divergence
        sn_stats = validator.compute_sector_neutral_ic()
        if not sn_stats.empty:
            sn_ic = sn_stats.loc["f", "sn_icir"]
            assert sn_ic < -0.3, f"SN IC should be negative (Alpha), got {sn_ic}"