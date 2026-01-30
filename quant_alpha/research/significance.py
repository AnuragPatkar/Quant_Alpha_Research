"""
Statistical Significance Testing
================================
Tools for testing statistical significance of alpha signals.

Features:
- T-tests for IC significance
- Bootstrap confidence intervals
- Deflated Sharpe ratio (multiple testing adjustment)
- False discovery rate control

Author: [Your Name]
Last Updated: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def calculate_t_statistic(
    ic_series: pd.Series,
    null_hypothesis: float = 0.0
) -> Dict[str, float]:
    """
    Calculate t-statistic for IC series.
    
    Tests H0: mean(IC) = null_hypothesis
    
    Args:
        ic_series: Series of IC values (one per period)
        null_hypothesis: Value under null hypothesis (usually 0)
        
    Returns:
        Dictionary with t-stat, p-value, and significance assessment
        
    Example:
        >>> result = calculate_t_statistic(ic_series)
        >>> if result['p_value'] < 0.05:
        ...     print("Statistically significant!")
    """
    ic_clean = ic_series.dropna()
    n = len(ic_clean)
    
    if n < 3:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant_5pct': False,
            'significant_1pct': False,
            'n_observations': n
        }
    
    mean_ic = ic_clean.mean()
    std_ic = ic_clean.std()
    se = std_ic / np.sqrt(n)
    
    if se < 1e-10:
        t_stat = 0.0
        p_value = 1.0
    else:
        t_stat = (mean_ic - null_hypothesis) / se
        # Two-tailed test
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_5pct': p_value < 0.05,
        'significant_1pct': p_value < 0.01,
        'mean_ic': float(mean_ic),
        'std_ic': float(std_ic),
        'se': float(se),
        'n_observations': n,
        'degrees_of_freedom': n - 1
    }


def test_statistical_significance(
    ic_series: pd.Series,
    method: str = 't_test',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Comprehensive statistical significance testing.
    
    Args:
        ic_series: Series of IC values
        method: 't_test' or 'bootstrap'
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with test results
    """
    ic_clean = ic_series.dropna()
    n = len(ic_clean)
    
    if n < 5:
        logger.warning(f"Too few observations ({n}) for significance testing")
        return {'significant': False, 'reason': 'insufficient_data'}
    
    results = {
        'n_observations': n,
        'mean_ic': float(ic_clean.mean()),
        'median_ic': float(ic_clean.median()),
        'std_ic': float(ic_clean.std()),
        'pct_positive': float((ic_clean > 0).mean()),
    }
    
    # T-test
    t_results = calculate_t_statistic(ic_clean)
    results.update({
        't_statistic': t_results['t_statistic'],
        'p_value_t': t_results['p_value'],
    })
    
    # Bootstrap if requested
    if method == 'bootstrap' or n_bootstrap > 0:
        boot_results = bootstrap_confidence_interval(
            ic_clean.values,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level
        )
        results.update({
            'ci_lower': boot_results['ci_lower'],
            'ci_upper': boot_results['ci_upper'],
            'bootstrap_p_value': boot_results['p_value'],
        })
    
    # Information Ratio
    ir = ic_clean.mean() / (ic_clean.std() + 1e-10)
    results['information_ratio'] = float(ir)
    
    # Final significance assessment
    results['significant_t_test'] = t_results['p_value'] < 0.05
    
    if 'bootstrap_p_value' in results:
        results['significant_bootstrap'] = results['bootstrap_p_value'] < 0.05
        results['significant'] = results['significant_t_test'] and results['significant_bootstrap']
    else:
        results['significant'] = results['significant_t_test']
    
    return results


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval.
    
    Bootstrap is more robust than t-test when:
    - Data is not normally distributed
    - Sample size is small
    - Distribution is skewed
    
    Args:
        data: Array of values
        statistic: 'mean' or 'median'
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_state: Random seed
        
    Returns:
        Dictionary with CI bounds and p-value
    """
    np.random.seed(random_state)
    
    n = len(data)
    if n < 3:
        return {
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': 1.0
        }
    
    # Bootstrap resampling
    boot_stats = []
    
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=n, replace=True)
        
        if statistic == 'mean':
            boot_stats.append(np.mean(boot_sample))
        else:
            boot_stats.append(np.median(boot_sample))
    
    boot_stats = np.array(boot_stats)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_stats, alpha/2 * 100)
    ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)
    
    # P-value (proportion of bootstrap samples <= 0)
    if statistic == 'mean':
        observed = np.mean(data)
    else:
        observed = np.median(data)
    
    # Two-tailed p-value
    if observed >= 0:
        p_value = 2 * np.mean(boot_stats <= 0)
    else:
        p_value = 2 * np.mean(boot_stats >= 0)
    
    p_value = min(p_value, 1.0)
    
    return {
        'observed': float(observed),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_width': float(ci_upper - ci_lower),
        'p_value': float(p_value),
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }


def calculate_deflated_sharpe(
    sharpe_ratio: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0,
    kurtosis: float = 3,
    expected_max_sr: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate Deflated Sharpe Ratio (DSR).
    
    Adjusts Sharpe ratio for multiple testing bias.
    When you try many strategies and pick the best one,
    the observed Sharpe is biased upward.
    
    Reference: Bailey and Lopez de Prado (2014)
    
    Args:
        sharpe_ratio: Observed Sharpe ratio
        n_trials: Number of strategies/parameters tried
        n_observations: Number of return observations
        skewness: Return skewness (0 for normal)
        kurtosis: Return kurtosis (3 for normal)
        expected_max_sr: Expected max SR under null (calculated if None)
        
    Returns:
        Dictionary with DSR and p-value
    """
    if n_trials <= 0 or n_observations <= 0:
        return {
            'deflated_sharpe': sharpe_ratio,
            'p_value': 1.0,
            'haircut': 0.0
        }
    
    # Expected maximum Sharpe under null hypothesis
    if expected_max_sr is None:
        # Approximate expected max of n_trials standard normals
        euler_mascheroni = 0.5772156649
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) - \
                         (euler_mascheroni + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
    
    # Standard error of Sharpe ratio
    sr_std = np.sqrt((1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                     ((kurtosis - 3) / 4) * sharpe_ratio**2) / n_observations)
    
    # Deflated Sharpe Ratio
    if sr_std > 1e-10:
        dsr = (sharpe_ratio - expected_max_sr) / sr_std
        p_value = 1 - stats.norm.cdf(dsr)
    else:
        dsr = 0
        p_value = 1.0
    
    # Haircut (how much SR is reduced)
    haircut = max(0, (expected_max_sr - 0) / (sharpe_ratio + 1e-10))
    
    return {
        'observed_sharpe': float(sharpe_ratio),
        'expected_max_sharpe': float(expected_max_sr),
        'deflated_sharpe': float(max(0, sharpe_ratio - expected_max_sr)),
        'dsr_z_score': float(dsr),
        'p_value': float(p_value),
        'haircut_pct': float(haircut * 100),
        'significant': p_value < 0.05,
        'n_trials': n_trials,
        'n_observations': n_observations
    }


def multiple_testing_correction(
    p_values: List[float],
    method: str = 'fdr_bh',
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Correct p-values for multiple testing.
    
    Methods:
    - 'bonferroni': Conservative, controls family-wise error rate
    - 'fdr_bh': Benjamini-Hochberg, controls false discovery rate
    
    Args:
        p_values: List of p-values
        method: Correction method
        alpha: Significance level
        
    Returns:
        Dictionary with corrected p-values and significant flags
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    if method == 'bonferroni':
        # Bonferroni correction
        corrected = np.minimum(p_values * n, 1.0)
        significant = corrected < alpha
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # Calculate adjusted p-values
        cummax = np.maximum.accumulate((n / (np.arange(n) + 1)) * sorted_p[::-1])[::-1]
        corrected_sorted = np.minimum(cummax, 1.0)
        
        # Restore original order
        corrected = np.empty(n)
        corrected[sorted_idx] = corrected_sorted
        
        significant = corrected < alpha
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'original_p_values': p_values.tolist(),
        'corrected_p_values': corrected.tolist(),
        'significant': significant.tolist(),
        'n_significant': int(significant.sum()),
        'method': method,
        'alpha': alpha
    }