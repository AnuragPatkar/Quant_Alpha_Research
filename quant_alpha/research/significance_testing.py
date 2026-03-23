"""
quant_alpha/research/significance_testing.py
==============================================
Statistical Significance & Hypothesis Testing Engine.

FIXES:
  BUG-084 (MEDIUM): bootstrap_sharpe() computed Sharpe as
           mean(sample) / std(sample) × √252, omitting the risk-free rate.
           The confirmed Sharpe formula is:
           S = (mean(r) - rf/252) / std(r) × √252
           Without subtracting rf/252, the bootstrap Sharpe overstates
           the metric by ~rf/√252 × √252 = rf ≈ 0.04 at rf=4%.
           Fixed: excess = sample - rf_daily before Sharpe computation.
           A risk_free_rate parameter (default 0.04) has been added to
           __init__ and bootstrap_sharpe() to allow customisation.
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
        Args:
            ic_series       : Time-series of daily Information Coefficients.
            returns_series  : Time-series of daily strategy returns.
            risk_free_rate  : Annual risk-free rate (default 4%).
        """
        self.ic_series      = ic_series
        self.returns_series = returns_series
        self.risk_free_rate = risk_free_rate

    def t_test_ic(self) -> dict:
        """
        One-sample t-test on the IC series.

        H₀: μ_IC = 0  (no predictive power)
        H₁: μ_IC ≠ 0

        Returns:
            Dict: t-stat, p-value, significance flag, mean IC, N.
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
        Non-parametric bootstrap to estimate the Sharpe Ratio distribution.

        Annualised Sharpe:
            S = (mean(r - rf/252) / std(r - rf/252)) × √252

        FIX BUG-084: subtract daily risk-free rate before computing mean/std.

        Args:
            n_samples        : Bootstrap iterations.
            confidence_level : Width of CI (default 95%).
            risk_free_rate   : Annual rf rate. Overrides instance value if supplied.

        Returns:
            Dict: Mean Sharpe, CI bounds, std error, P(SR > 0).
        """
        if self.returns_series is None:
            raise ValueError("Returns series not provided.")

        rf       = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        rf_daily = rf / 252
        rets     = self.returns_series.dropna().values

        sharpe_dist = []

        for _ in range(n_samples):
            sample = np.random.choice(rets, size=len(rets), replace=True)

            # FIX BUG-084: compute excess returns before Sharpe
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
        Augmented Dickey-Fuller test for unit roots.

        H₀: series has a unit root (non-stationary)
        H₁: series is stationary (p < 0.05 → reject H₀)

        Returns:
            Dict with adf_stat, p_value, stationary flag, or None if
            statsmodels is not installed.
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