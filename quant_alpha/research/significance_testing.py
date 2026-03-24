"""
Statistical Significance & Hypothesis Testing Engine.
=====================================================

Provides non-parametric and parametric boundaries to validate alpha signal efficacy.

Purpose
-------
This module strictly evaluates whether generated strategy returns and Information 
Coefficients (IC) diverge significantly from random noise, anchoring statistical 
confidence prior to model deployment.

Role in Quantitative Workflow
-----------------------------
Acts as the rigorous mathematical filter separating spurious over-fitted artifacts 
from genuine structural alpha. Enforces stationarity bounds to confirm that predictive 
power is theoretically persistent out-of-sample.

Mathematical Dependencies
-------------------------
- **SciPy (stats)**: Calculates continuous sample T-tests bounding hypothesis distributions.
- **Statsmodels**: Binds Augmented Dickey-Fuller (ADF) constraints testing for unit roots.
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class SignificanceTester:
    """Statistical engine for validating signal efficacy and return robustness."""

    def __init__(
        self,
        ic_series: pd.Series = None,
        returns_series: pd.Series = None,
        risk_free_rate: float = 0.04,
    ):
        """
        Initializes the structural inference evaluation map.
        
        Args:
            ic_series (pd.Series, optional): Time-series array defining daily Information Coefficients.
            returns_series (pd.Series, optional): Time-series array bounding daily geometric returns.
            risk_free_rate (float): Standard annual systemic risk-free baseline. Defaults to 0.04.
        """
        self.ic_series      = ic_series
        self.returns_series = returns_series
        self.risk_free_rate = risk_free_rate

    def t_test_ic(self) -> dict:
        """
        Evaluates a standard one-sample t-test bound against the IC distribution.

        Null Hypothesis ($H_0$): $\mu_{IC} = 0$ (The signal exhibits zero structural predictive power).
        Alternative Hypothesis ($H_1$): $\mu_{IC} \neq 0$.

        Returns:
            dict: A mapping of the extracted t-statistic, p-value bounds, significance flag, and sample count.
            
        Raises:
            ValueError: If the necessary continuous IC series was omitted during initialization.
        """
        if self.ic_series is None:
            raise ValueError("IC series not provided.")

        clean_ic    = self.ic_series.dropna()
        t_stat, p_val = stats.ttest_1samp(clean_ic, 0)

        return {
            't_stat':     float(t_stat),
            'p_value':    float(p_val),
            'significant': bool(p_val < 0.05),
            'mean_ic':    float(clean_ic.mean()),
            'n_obs':      len(clean_ic),
        }

    def bootstrap_sharpe(
        self,
        n_samples: int = 1000,
        confidence_level: float = 0.95,
        risk_free_rate: float = None,
    ) -> dict:
        """
        Estimates the empirical Sharpe Ratio distribution via non-parametric bootstrapping.

        Iteratively simulates $N$ resampling horizons mapping localized standard deviation 
        and annualized excess boundaries to strictly bound the true population Sharpe parameters.
        
        Annualized Geometric Formula: $S = \frac{\mu(r - rf_{daily})}{\sigma(r - rf_{daily})} \times \sqrt{252}$

        Args:
            n_samples (int): Total bootstrap stochastic iterations. Defaults to 1000.
            confidence_level (float): The targeted parametric width of the CI. Defaults to 0.95.
            risk_free_rate (float, optional): The annual benchmark risk-free rate threshold. 
                Dynamically overrides the instance configuration if supplied. Defaults to None.

        Returns:
            dict: The simulated distributional characteristics including Mean Sharpe, CI bounds, and probability bounds.
            
        Raises:
            ValueError: If the continuous geometric returns series was omitted.
        """
        if self.returns_series is None:
            raise ValueError("Returns series not provided.")

        rf       = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rf_daily = rf / 252
        rets     = self.returns_series.dropna().values

        sharpe_dist = []

        for _ in range(n_samples):
            sample = np.random.choice(rets, size=len(rets), replace=True)

            # Isolates pure excess returns strictly bounding standard mathematical Sharpe formulations.
            excess  = sample - rf_daily
            std_dev = np.std(excess)

            if std_dev == 0:
                continue

            sharpe = np.mean(excess) / std_dev * np.sqrt(252)
            sharpe_dist.append(sharpe)

        sharpe_arr = np.array(sharpe_dist)

        if len(sharpe_arr) == 0:
            return {
                'mean_sharpe':          np.nan,
                'lower_bound':          np.nan,
                'upper_bound':          np.nan,
                'std_error':            np.nan,
                'prob_sharpe_positive': np.nan,
            }

        alpha  = (1 - confidence_level) / 2
        lower  = float(np.percentile(sharpe_arr, alpha * 100))
        upper  = float(np.percentile(sharpe_arr, (1 - alpha) * 100))

        return {
            'mean_sharpe':          float(np.mean(sharpe_arr)),
            'lower_bound':          lower,
            'upper_bound':          upper,
            'std_error':            float(np.std(sharpe_arr)),
            'prob_sharpe_positive': float(np.mean(sharpe_arr > 0)),
        }

    def check_stationarity(self) -> dict | None:
        """
        Conducts the Augmented Dickey-Fuller (ADF) bound testing checking for unit roots.

        Null Hypothesis ($H_0$): The executing matrix series contains a unit root (non-stationary).
        Alternative Hypothesis ($H_1$): The series enforces strict stationarity.

        Returns:
            Optional[dict]: Explicit testing statistics mapping scalar outputs, or None if 
            underlying math engines fail dependency resolutions.
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.warning("statsmodels not installed. Skipping ADF test.")
            return None

        if self.ic_series is None:
            return None

        result = adfuller(self.ic_series.dropna())
        return {
            'adf_stat':   float(result[0]),
            'p_value':    float(result[1]),
            'stationary': bool(result[1] < 0.05),
        }