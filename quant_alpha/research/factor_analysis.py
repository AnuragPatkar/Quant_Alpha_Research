"""
Factor Analysis & Performance Evaluation Engine.
==============================================

Provides statistical verification of cross-sectional alpha factors.

Purpose
-------
This module computes structural Information Coefficient (IC) time-series, 
monotonic quantile spreads, and signal autocorrelation to evaluate the predictive 
power and stability of quantitative features.

Role in Quantitative Workflow
-----------------------------
Acts as the primary diagnostic gateway mapping standalone mathematical formulas 
to forward asset returns. Ensures that features deployed to predictive ensembles 
demonstrate strict out-of-sample stationarity and monotonic ranking capabilities.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Vectorized group-by aggregations and time-series alignments.
- **SciPy (stats)**: Calculates cross-sectional Spearman/Pearson Rank Correlations 
  and standard hypothesis testing (T-tests).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)



def _prepare_factor_data(
    data: pd.DataFrame,
    factor_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Validates and temporally aligns factor data for cross-sectional analysis.

    Enforces a strict MultiIndex (date, ticker) topology, dropping missing structural 
    values to strictly prevent cascading NaN evaluation errors during correlation steps.
    
    Args:
        data (pd.DataFrame): The raw observational panel matrix.
        factor_col (str): The column target containing the synthesized alpha signal.
        target_col (str): The column target containing the forward geometric returns.
        
    Returns:
        pd.DataFrame: A dimensionally aligned dataset indexed by (date, ticker).
        
    Raises:
        ValueError: If the structural index is flat without identifying columns, or 
            if strict target components are absent from the matrix.
    """
    df = data.copy()

    # Resolves internal schema parameters to MultiIndex if flat columns supplied
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

    # Prunes mathematical intersection boundaries preventing NaN contamination
    df = df.dropna(subset=[factor_col, target_col])
    return df


class FactorAnalyzer:
    """
    Engine for cross-sectional factor performance evaluation.
    Computes IC time-series matrices, monotonic quantile spreads, and 
    autocorrelation signal stability boundaries.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        factor_col: str,
        target_col: str = 'raw_ret_5d',
    ):
        """
        Initializes the cross-sectional factor analyzer.
        
        Args:
            data (pd.DataFrame): Input market panel containing features and forward targets.
            factor_col (str): Matrix key defining the quantitative feature vector.
            target_col (str): Matrix key defining the forward prediction horizon returns.
        """
        self.factor_col = factor_col
        self.target_col = target_col
        self.data = _prepare_factor_data(data, factor_col, target_col)
        self._ic_series = None

    def calculate_ic(self, method: str = 'spearman') -> pd.Series:
        """
        Calculates the Information Coefficient (IC) structural time-series.

        Computes the discrete cross-sectional correlation mapping the factor distribution 
        against corresponding forward asset returns independently for each temporal bucket.

        Args:
            method (str): The statistical correlation methodology ('spearman' for rank 
                invariance, 'pearson' for strictly linear evaluations). Defaults to 'spearman'.

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
        Generates robust summary statistics tracking the alpha signal's predictive efficacy.
        
        Computes empirical thresholds determining standard significance (Mean IC, IC Std, 
        Information Ratio, Hit Ratio, and discrete T-statistics).
        
        Returns:
            dict: Standardized dictionary mapping continuous performance scalars.
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
            # Extracts continuous discrete bounded significance testing parameters directly via SciPy
            't-stat':         float(stats.ttest_1samp(ic, 0)[0]) if n >= 3 else np.nan,
            'N':              n,
        }

    def calculate_quantile_returns(
        self,
        quantiles: int = 5,
    ):
        """
        Analyzes strict signal monotonicity via cross-sectional quantile bucketing.
        
        Args:
            quantiles (int): The number of discrete cross-sectional bins. Defaults to 5.

        Returns:
            Tuple[pd.Series, float]: The aggregated mean geometric returns bounded per 
                quantile, and the resultant empirical Long-Short statistical spread.
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
        Calculates mathematical factor autocorrelation serving as a structural turnover proxy.

        High autocorrelation ($\rho > 0.9$) denotes a slow-moving signal with intrinsically 
        low portfolio turnover (e.g., Value bounds). Low autocorrelation ($\rho < 0.5$) 
        denotes a fast-decaying signal mapping to high turnover mandates (e.g., Fast Momentum).
        
        Args:
            lag (int): The discrete temporal lag boundary for shift extraction. Defaults to 1.
            
        Returns:
            float: The aggregate time-series correlation averaged uniformly across all tickers.
        """
        factor_matrix = self.data[self.factor_col].unstack()   # dates × tickers

        # Computes the discrete autocorrelation metric measuring signal stickiness via 
        # strictly evaluating temporal limits (axis=0) cross-sectionally.
        rho = factor_matrix.corrwith(factor_matrix.shift(lag))  # axis=0 default
        return float(rho.mean())

    def plot_ic_ts(self, window: int = 20, save_path: str = None) -> None:
        """
        Visualizes the Information Coefficient time series against a rolling trend smoothing line.
        
        Args:
            window (int): The expanding moving average limit for variance extraction. Defaults to 20.
            save_path (str, optional): The filepath to explicitly target rendering to disk. 
                Defaults to None, triggering interactive evaluation.
                
        Returns:
            None: Exposes explicit plots via backend execution APIs.
        """
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
        """
        Visualizes the specific structural monotonic linearity of aggregated quantile returns.
        
        Args:
            quantiles (int): Explicit total count of structural rendering bins. Defaults to 5.
            save_path (str, optional): The filepath to explicitly target rendering to disk. 
                Defaults to None, triggering interactive evaluation.
                
        Returns:
            None: Exposes explicit plots via backend execution APIs.
        """
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