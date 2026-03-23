"""
quant_alpha/research/factor_analysis.py
=========================================
Factor Analysis & Performance Evaluation Engine.

FIXES:
  BUG-083 (HIGH): `from .utils import prepare_factor_data` resolved to
           quant_alpha.research.utils which does not exist as a separate
           module, causing ImportError at startup. The helper is trivially
           small and has been inlined directly.

  BUG-085 (MEDIUM): calculate_autocorrelation() called
           factor_matrix.corrwith(factor_matrix.shift(lag), axis=1).mean()
           axis=1 computes row-wise correlation ACROSS TICKERS (spatial),
           not ACROSS TIME (temporal). This made the autocorrelation metric
           meaningless as a turnover proxy. Fixed to axis=0 (default), which
           computes the time-series correlation per ticker then averages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FIX BUG-083: inline prepare_factor_data (was in missing .utils module)
# ---------------------------------------------------------------------------

def _prepare_factor_data(
    data: pd.DataFrame,
    factor_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Validate and index factor data for cross-sectional analysis.

    Ensures the DataFrame has a MultiIndex (date, ticker) and that both
    the factor column and target column are present and numeric.
    """
    df = data.copy()

    # Coerce to MultiIndex if flat columns supplied
    if not isinstance(df.index, pd.MultiIndex):
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.set_index(['date', 'ticker'])
        else:
            raise ValueError(
                "prepare_factor_data: DataFrame must have a MultiIndex "
                "(date, ticker) or 'date'/'ticker' columns."
            )

    for col in [factor_col, target_col]:
        if col not in df.columns:
            raise ValueError(
                f"prepare_factor_data: required column '{col}' not found."
            )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where either column is missing — can't compute IC otherwise
    df = df.dropna(subset=[factor_col, target_col])
    return df


# ---------------------------------------------------------------------------
# FactorAnalyzer
# ---------------------------------------------------------------------------

class FactorAnalyzer:
    """
    Engine for cross-sectional factor performance evaluation.
    Computes IC time-series, monotonic quantile spreads, and signal stability.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        factor_col: str,
        target_col: str = 'raw_ret_5d',
    ):
        self.factor_col = factor_col
        self.target_col = target_col
        # FIX BUG-083: use inlined helper instead of missing .utils import
        self.data = _prepare_factor_data(data, factor_col, target_col)
        self._ic_series = None

    def calculate_ic(self, method: str = 'spearman') -> pd.Series:
        """
        Calculate the Information Coefficient (IC) time series.

        Computes the cross-sectional correlation between factor and forward
        returns for each date.

        Args:
            method : 'spearman' (Rank IC, robust to outliers) or 'pearson'.

        Returns:
            pd.Series: Daily IC values indexed by date.
        """
        logger.info(f"Calculating {method.capitalize()} IC for {self.factor_col}...")

        def _ic_func(group):
            if len(group) < 10:
                return np.nan
            if method == 'spearman':
                return stats.spearmanr(
                    group[self.factor_col], group[self.target_col]
                )[0]
            return stats.pearsonr(
                group[self.factor_col], group[self.target_col]
            )[0]

        self._ic_series = (
            self.data.groupby(level='date').apply(_ic_func)
        )
        return self._ic_series

    def get_ic_summary(self) -> dict:
        """
        Generate summary statistics for the alpha signal.

        Metrics:
        - Mean IC, IC Std, ICIR = IC_mean / IC_std
        - Hit Ratio: fraction of dates with positive IC
        - t-stat = ICIR × sqrt(N_dates)  ← uses scipy ttest for correctness
        """
        if self._ic_series is None:
            self.calculate_ic()

        ic = self._ic_series.dropna()
        n  = len(ic)

        return {
            'Mean IC':        float(ic.mean()),
            'IC Std':         float(ic.std()),
            'ICIR (IC/Std)':  float(ic.mean() / ic.std()) if ic.std() != 0 else 0.0,
            'Hit Ratio (>0)': float((ic > 0).mean()),
            # t-stat = ICIR × sqrt(N) — stats.ttest_1samp returns this correctly
            't-stat':         float(stats.ttest_1samp(ic, 0)[0]) if n >= 3 else np.nan,
            'N':              n,
        }

    def calculate_quantile_returns(
        self,
        quantiles: int = 5,
    ):
        """
        Analyze signal monotonicity via quantile bucketing.

        Returns:
            Tuple[pd.Series, float]: (Mean returns per quantile, L-S spread).
        """
        def _quantile_bucket(x):
            try:
                return pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                return np.nan

        self.data['quantile'] = (
            self.data
            .groupby(level='date')[self.factor_col]
            .transform(_quantile_bucket)
        )

        q_ret  = self.data.groupby('quantile')[self.target_col].mean()
        q_min  = q_ret.index.min()
        q_max  = q_ret.index.max()
        spread = float(q_ret.loc[q_max] - q_ret.loc[q_min])

        return q_ret, spread

    def calculate_autocorrelation(self, lag: int = 1) -> float:
        """
        Calculate factor autocorrelation as a proxy for turnover.

        High ρ (> 0.9): slow-moving signal, low turnover (e.g. Value).
        Low  ρ (< 0.5): fast-decaying signal, high turnover (e.g. Momentum).

        FIX BUG-085: corrwith must use axis=0 (default) to compute
        time-series correlation per ticker, then average across tickers.
        The original code used axis=1 which computed cross-ticker
        (spatial) correlation per row — wrong for a turnover proxy.
        """
        factor_matrix = self.data[self.factor_col].unstack()   # dates × tickers

        # FIX BUG-085: axis=0 (default) → time-series correlation per ticker
        rho = factor_matrix.corrwith(factor_matrix.shift(lag))  # axis=0 default
        return float(rho.mean())

    def plot_ic_ts(self, window: int = 20, save_path: str = None) -> None:
        """Visualize the IC time series with a rolling trend line."""
        if self._ic_series is None:
            self.calculate_ic()

        ic = self._ic_series
        ma = ic.rolling(window).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(ic.index, ic, color='gray', alpha=0.3, label='Daily IC')
        ax.plot(ma.index, ma, color='blue', linewidth=2,
                label=f'{window}-Day MA')
        ax.axhline(float(ic.mean()), color='red', linestyle='--',
                   label=f'Mean: {ic.mean():.3f}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f"Information Coefficient: {self.factor_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_quantile_returns(
        self, quantiles: int = 5, save_path: str = None
    ) -> None:
        """Visualize the monotonicity of quantile returns."""
        q_ret, spread = self.calculate_quantile_returns(quantiles)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in q_ret.values]
        q_ret.plot(kind='bar', color=colors, alpha=0.7, ax=ax)
        ax.set_title(
            f"Mean Return by Quantile | Spread: {spread:.4f}"
        )
        ax.set_xlabel(f"Quantile (1=Low, {quantiles}=High)")
        ax.set_ylabel("Mean Forward Return")
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()