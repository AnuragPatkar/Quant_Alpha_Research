"""
Statistical Significance & Hypothesis Testing Engine
====================================================
Rigorous validation suite for alpha signals and strategy returns.

Purpose
-------
The `SignificanceTester` provides a battery of statistical tests to determine if observed
alpha performance is genuine or likely due to random chance (luck). It addresses three
critical questions:
1.  Is the predictive power (IC) significantly different from zero? (t-test)
2.  Is the risk-adjusted return (Sharpe) robust to data perturbation? (Bootstrap)
3.  Is the signal regime-stable over time? (Stationarity/ADF)

Usage
-----
.. code-block:: python

    tester = SignificanceTester(ic_series=daily_ic, returns_series=strategy_rets)

    # 1. Test Predictive Power Significance
    ic_stats = tester.t_test_ic()

    # 2. Estimate Sharpe Ratio Confidence Intervals
    sharpe_ci = tester.bootstrap_sharpe(n_samples=5000)

    # 3. Check for Regime Stability (Stationarity)
    stationarity = tester.check_stationarity()

Importance
----------
-   **False Discovery Rate Control**: Mitigates Type I errors (false positives) in factor selection.
-   **Performance Robustness**: Bootstrapping reveals the sensitivity of the Sharpe Ratio to specific
    market regimes or outliers, avoiding the normality assumptions of analytic standard errors (Lo, 2002).
-   **Modeling Assumptions**: Stationarity checks ensure that historical relationships are
    likely to persist, a fundamental requirement for linear predictive models.

Tools & Frameworks
------------------
-   **SciPy (stats)**: Parametric hypothesis testing (Student's t-test).
-   **NumPy**: Non-parametric resampling (Bootstrapping) and vector algebra.
-   **Statsmodels**: Time-series diagnostics (Augmented Dickey-Fuller test).
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class SignificanceTester:
    """
    Statistical engine for validating signal efficacy and return robustness.
    """
    
    def __init__(self, ic_series: pd.Series = None, returns_series: pd.Series = None):
        """
        Initialize with signal or return data.

        Args:
            ic_series (Optional[pd.Series]): Time-series of Information Coefficients.
            returns_series (Optional[pd.Series]): Time-series of strategy returns.
        """
        self.ic_series = ic_series
        self.returns_series = returns_series
        
    def t_test_ic(self):
        """
        Performs a one-sample Student's t-test on the Information Coefficient (IC) series.
        
        Hypothesis:
        .. math::
            H_0: \\mu_{IC} = 0 \\quad \\text{(No predictive power)}
            
            H_1: \\mu_{IC} \\neq 0

        Returns:
            Dict: Test statistics, p-value, and significance flag ($p < 0.05$).
        """
        if self.ic_series is None:
            raise ValueError("IC series not provided.")
            
        clean_ic = self.ic_series.dropna()  # Ensure valid N for degrees of freedom
        t_stat, p_val = stats.ttest_1samp(clean_ic, 0)
        
        return {
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'mean_ic': clean_ic.mean(),
            'n_obs': len(clean_ic)
        }

    def bootstrap_sharpe(self, n_samples=1000, confidence_level=0.95):
        """
        Non-parametric Bootstrap resampling to estimate Sharpe Ratio distribution.
        
        Methodology:
        Generates $B$ resampled return histories (with replacement) to approximate the sampling
        distribution of the Sharpe Ratio. This avoids normality assumptions.
        
        .. math::
            SR_{ann} = \\frac{\\mu_{boot}}{\\sigma_{boot}} \\times \\sqrt{252}

        Args:
            n_samples (int): Number of bootstrap iterations ($B$).
            confidence_level (float): Width of the confidence interval (e.g., 0.95).

        Returns:
            Dict: Statistics including Mean Sharpe, CI bounds, and Probability(SR > 0).
        """
        if self.returns_series is None:
            raise ValueError("Returns series not provided.")
            
        # Optimization: Convert to numpy array once for faster indexing
        rets = self.returns_series.dropna().values
        sharpe_dist = []
        
        for _ in range(n_samples):
            # Monte Carlo Simulation: Resample with replacement
            sample = np.random.choice(rets, size=len(rets), replace=True)
            
            # Degenerate Case: Avoid division by zero if sample is flat
            std_dev = np.std(sample)
            if std_dev == 0:
                continue
            
            # Annualized Sharpe Calculation
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            sharpe_dist.append(sharpe)
            
        sharpe_dist = np.array(sharpe_dist)
        
        # Percentile Method for Confidence Intervals
        lower = np.percentile(sharpe_dist, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(sharpe_dist, (1 + confidence_level) / 2 * 100)
        
        return {
            'mean_sharpe': np.mean(sharpe_dist),
            'lower_bound': lower,
            'upper_bound': upper,
            'std_error': np.std(sharpe_dist),
            'prob_sharpe_positive': np.mean(sharpe_dist > 0)
        }

    def check_stationarity(self):
        """
        Performs the Augmented Dickey-Fuller (ADF) test for unit roots.

        Stationarity ($H_0$ rejected) implies the mean and variance of the signal
        do not change over time, which is a prerequisite for most predictive models.
        
        Returns:
            Optional[Dict]: ADF statistic and p-value, or None if statsmodels is missing.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("statsmodels not installed. Skipping ADF test.")
            return None
            
        if self.ic_series is None:
            return None
            
        # ADF Null Hypothesis: The series has a unit root (is non-stationary)
        result = adfuller(self.ic_series.dropna())
        return {
            'adf_stat': result[0],
            'p_value': result[1],
            'stationary': result[1] < 0.05
        }
