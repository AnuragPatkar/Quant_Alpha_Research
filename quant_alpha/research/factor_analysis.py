import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging
from .utils import prepare_factor_data

logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """
    Comprehensive Factor Analysis Tool.
    Calculates IC, Rank IC, Quantile Returns, and Turnover.
    """

    def __init__(self, data: pd.DataFrame, factor_col: str, target_col: str = 'raw_ret_5d'):
        self.factor_col = factor_col
        self.target_col = target_col
        self.data = prepare_factor_data(data, factor_col, target_col)
        self._ic_series = None

    def calculate_ic(self, method='spearman'):
        """
        Calculates Information Coefficient (IC) over time.
        method: 'spearman' (Rank IC) or 'pearson' (Linear IC)
        """
        logger.info(f"Calculating {method.capitalize()} IC for {self.factor_col}...")

        def _ic_func(group):
            if len(group) < 10:
                return np.nan
            if method == 'spearman':
                return stats.spearmanr(group[self.factor_col], group[self.target_col])[0]
            else:
                return stats.pearsonr(group[self.factor_col], group[self.target_col])[0]

        self._ic_series = self.data.groupby(level='date').apply(_ic_func)
        return self._ic_series

    def get_ic_summary(self):
        """Returns summary statistics of the IC series."""
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
        Calculates mean returns for each factor quantile.

        BUG-1 FIX: spread used iloc[-1] - iloc[0] which breaks when quantiles
        are missing due to duplicates='drop'. Now uses explicit label lookup.
        """
        def _quantile_bucket(x):
            try:
                return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                return np.nan

        self.data['quantile'] = self.data.groupby(level='date')[self.factor_col].transform(
            _quantile_bucket
        )

        q_ret = self.data.groupby('quantile')[self.target_col].mean()

        # BUG-1 FIX: use explicit label lookup instead of iloc
        # iloc[-1] was Q5 only if all 5 quantiles present — not guaranteed
        q_min = q_ret.index.min()
        q_max = q_ret.index.max()
        spread = q_ret.loc[q_max] - q_ret.loc[q_min]

        return q_ret, spread

    def calculate_autocorrelation(self, lag=1):
        """
        Calculates factor autocorrelation (turnover proxy).
        High autocorrelation = Low Turnover.
        """
        factor_matrix = self.data[self.factor_col].unstack()
        rho = factor_matrix.corrwith(factor_matrix.shift(lag), axis=1).mean()
        return rho

    def plot_ic_ts(self, window=20, save_path=None):
        """Plots IC time series with moving average."""
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
        """Bar chart of quantile returns."""
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