import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class SignificanceTester:
    """
    Performs statistical tests on Alpha signals.
    """
    
    def __init__(self, ic_series: pd.Series = None, returns_series: pd.Series = None):
        self.ic_series = ic_series
        self.returns_series = returns_series
        
    def t_test_ic(self):
        """
        One-sample t-test to check if mean IC is significantly different from 0.
        """
        if self.ic_series is None:
            raise ValueError("IC series not provided.")
            
        clean_ic = self.ic_series.dropna()
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
        Bootstrap resampling to estimate Sharpe Ratio confidence interval.
        """
        if self.returns_series is None:
            raise ValueError("Returns series not provided.")
            
        rets = self.returns_series.dropna().values
        sharpe_dist = []
        
        for _ in range(n_samples):
            # Resample with replacement
            sample = np.random.choice(rets, size=len(rets), replace=True)
            if np.std(sample) == 0: continue
            
            # Annualized Sharpe
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            sharpe_dist.append(sharpe)
            
        sharpe_dist = np.array(sharpe_dist)
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
        Augmented Dickey-Fuller test for stationarity of IC.
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
            'adf_stat': result[0],
            'p_value': result[1],
            'stationary': result[1] < 0.05
        }
