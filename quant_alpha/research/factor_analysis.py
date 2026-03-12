"""
Factor Analysis & Performance Evaluation Engine
===============================================
Comprehensive suite for evaluating the predictive power and stability of alpha factors.

Purpose
-------
The `FactorAnalyzer` serves as the primary validation gate for new signals. It computes
industry-standard metrics including Information Coefficient (IC), Quantile Monotonicity,
and Factor Autocorrelation (Turnover Proxy). It transforms raw signal data into
statistically rigorous performance attribution reports.

Usage
-----
.. code-block:: python

    # Initialize with signal and forward returns
    analyzer = FactorAnalyzer(data=df, factor_col='value_score', target_col='ret_5d')

    # 1. Compute Predictive Power (IC)
    ic_series = analyzer.calculate_ic(method='spearman')
    summary = analyzer.get_ic_summary()

    # 2. Check Monotonicity (Quantiles)
    q_ret, spread = analyzer.calculate_quantile_returns(quantiles=5)

Importance
----------
- **Predictive Power**: Quantifies the correlation between signal and future returns ($IC$).
- **Monotonicity**: Verifies that higher factor scores consistently lead to higher returns
  (Top Quintile > Bottom Quintile).
- **Turnover Proxy**: High autocorrelation implies stable signals, reducing transaction costs.

Tools & Frameworks
------------------
- **SciPy (stats)**: Pearson/Spearman correlation and t-tests for significance.
- **Pandas**: GroupBy split-apply-combine operations for cross-sectional analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
from .utils import prepare_factor_data

logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """
    Engine for cross-sectional factor performance evaluation.
    Computes IC time-series, monotonic quantile spreads, and signal stability metrics.
    """

    def __init__(self, data: pd.DataFrame, factor_col: str, target_col: str = 'raw_ret_5d'):
        """
        Initializes the analyzer with cleaned, aligned data.

        Args:
            data (pd.DataFrame): Input containing factor values and forward returns.
            factor_col (str): Column name of the alpha signal.
            target_col (str): Column name of the target return (e.g., 'raw_ret_5d').
        """
        self.factor_col = factor_col
        self.target_col = target_col
        self.data = prepare_factor_data(data, factor_col, target_col)
        self._ic_series = None

    def calculate_ic(self, method='spearman'):
        """
        Calculates the Information Coefficient (IC) Time Series.

        Computes the cross-sectional correlation between factor values and forward returns
        for each time period $t$.

        Args:
            method (str): 'spearman' (Rank IC, robust to outliers) or 'pearson' (Linear IC).

        Returns:
            pd.Series: Time-series of daily IC values.
        """
        logger.info(f"Calculating {method.capitalize()} IC for {self.factor_col}...")

        def _ic_func(group):
            # Statistical Significance Guard: Require min N observations per cross-section
            if len(group) < 10:
                return np.nan
            if method == 'spearman':
                return stats.spearmanr(group[self.factor_col], group[self.target_col])[0]
            else:
                return stats.pearsonr(group[self.factor_col], group[self.target_col])[0]

        # Split-Apply-Combine: Calculate IC per date (cross-section)
        self._ic_series = self.data.groupby(level='date').apply(_ic_func)
        return self._ic_series

    def get_ic_summary(self):
        """
        Generates summary statistics for the Alpha signal.

        Metrics:
        - **Mean IC**: Average predictive power.
        - **IC IR**: Information Ratio (Mean / Std), proxy for risk-adjusted performance.
        - **t-stat**: Significance test against null hypothesis ($IC=0$).
        """
        if self._ic_series is None:
            self.calculate_ic()

        ic = self._ic_series
        return {
            'Mean IC':       ic.mean(),
            'IC Std':        ic.std(),
            'IR (IC/Std)':   ic.mean() / ic.std() if ic.std() != 0 else 0,
            'Hit Ratio (>0)': (ic > 0).mean(),
            't-stat':        stats.ttest_1samp(ic.dropna(), 0)[0]
        }

    def calculate_quantile_returns(self, quantiles=5, period='D'):
        """
        Analyzes the Monotonicity of the signal via Quantile Bucketing.

        Sorts assets into $N$ buckets based on factor value and computes the mean return
        of each bucket. Ideal factors show strictly increasing returns from Q1 to QN.

        Returns:
            Tuple[pd.Series, float]: (Mean returns per quantile, Long-Short Spread).
        """
        def _quantile_bucket(x):
            try:
                # Robust Binning: duplicates='drop' merges bins if signal distribution is highly concentrated
                return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                return np.nan

        # Cross-Sectional Ranking: Assign quantiles per day
        self.data['quantile'] = self.data.groupby(level='date')[self.factor_col].transform(
            _quantile_bucket
        )

        q_ret = self.data.groupby('quantile')[self.target_col].mean()

        # Robustness: Explicitly retrieve the min/max available quantiles from the index.
        # This prevents IndexErrors if 'duplicates=drop' reduces the number of bins
        # (e.g., in highly concentrated signals where Q1 == Q2).
        q_min = q_ret.index.min()
        q_max = q_ret.index.max()
        spread = q_ret.loc[q_max] - q_ret.loc[q_min]

        return q_ret, spread

    def calculate_autocorrelation(self, lag=1):
        """
        Calculates Factor Autocorrelation as a proxy for Turnover.

        .. math:: \rho = \text{corr}(F_t, F_{t-\text{lag}})

        Interpretation:
        - High $\rho$ (> 0.9): Slow-moving signal, Low Turnover (e.g., Value).
        - Low $\rho$ (< 0.5): Fast-decaying signal, High Turnover (e.g., Intraday Momentum).
        """
        factor_matrix = self.data[self.factor_col].unstack()
        rho = factor_matrix.corrwith(factor_matrix.shift(lag), axis=1).mean()
        return rho

    def plot_ic_ts(self, window=20, save_path=None):
        """
        Visualizes the Information Coefficient time series with a smoothing trend line.

        Args:
            window (int): Rolling mean window for trend visualization.
            save_path (Optional[str]): File path to save the plot image.
        """
        if self._ic_series is None:
            self.calculate_ic()

        ic = self._ic_series
        ma = ic.rolling(window).mean()

        plt.figure(figsize=(12, 6))
        plt.bar(ic.index, ic, color='gray', alpha=0.3, label='Daily IC')
        plt.plot(ma.index, ma, color='blue', linewidth=2, label=f'{window}-Day MA')
        plt.axhline(ic.mean(), color='red', linestyle='--', label=f'Mean: {ic.mean():.3f}')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title(f"Information Coefficient (IC): {self.factor_col}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_quantile_returns(self, quantiles=5, save_path=None):
        """
        Visualizes the monotonicity of quantile returns (Long-Short spread).

        Args:
            quantiles (int): Number of buckets used for segmentation.
            save_path (Optional[str]): File path to save the plot image.
        """
        q_ret, spread = self.calculate_quantile_returns(quantiles)

        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in q_ret.values]
        q_ret.plot(kind='bar', color=colors, alpha=0.7)
        plt.title(f"Mean Return by Quantile (Spread: {spread:.4f})")
        plt.xlabel("Quantile (1=Low, 5=High)")
        plt.ylabel("Mean Forward Return")
        plt.grid(axis='y', alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()