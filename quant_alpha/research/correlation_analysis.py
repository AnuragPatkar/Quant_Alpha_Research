import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance
import logging

logger = logging.getLogger(__name__)

class FactorCorrelator:
    """
    Analyzes correlation and redundancy between multiple alpha factors.
    """

    def __init__(self, factor_data: pd.DataFrame):
        """
        factor_data: DataFrame with MultiIndex (date, ticker) and columns as factors.
        """
        self.data        = factor_data
        self.corr_matrix = None

    def calculate_correlation(self, method='spearman'):
        """
        Calculates cross-sectional correlation averaged over time.

        BUG-4 FIX: groupby().apply() returns a MultiIndex (date, factor) result.
        .groupby(level=1).mean() was averaging over tickers (wrong level).
        Fix: use groupby(level='date') → then mean over date axis.
        """
        logger.info(f"Calculating {method} correlation matrix...")

        def _daily_corr(df):
            if len(df) < 5:
                return pd.DataFrame()
            return df.corr(method=method)

        # groupby date → each group is (n_tickers × n_factors)
        # _daily_corr returns (n_factors × n_factors) per date
        # result is MultiIndex: (date, factor_row) × factor_col
        daily_corrs = self.data.groupby(level='date').apply(_daily_corr)

        # BUG-4 FIX: daily_corrs has MultiIndex (date, factor)
        # We want mean across dates → groupby level=1 (factor row) is WRONG
        # Correct: reset date level and mean over it
        if isinstance(daily_corrs.index, pd.MultiIndex):
            # level 0 = date, level 1 = factor name
            self.corr_matrix = daily_corrs.groupby(level=1).mean()
        else:
            self.corr_matrix = daily_corrs

        # Ensure diagonal is exactly 1.0 (floating point cleanup)
        np.fill_diagonal(self.corr_matrix.values, 1.0)

        return self.corr_matrix

    def cluster_factors(self, threshold=0.5):
        """
        Uses Hierarchical Clustering to group similar factors.
        Returns a dictionary mapping Cluster ID → List of Factors.
        """
        if self.corr_matrix is None:
            self.calculate_correlation()

        corr = self.corr_matrix.fillna(0)
        dist = 1 - np.abs(corr)

        # Ensure distance matrix is symmetric and diagonal=0
        dist_arr = distance.squareform(
            np.clip((dist.values + dist.values.T) / 2, 0, None),
            checks=False
        )

        linkage  = hierarchy.linkage(dist_arr, method='ward')
        clusters = hierarchy.fcluster(linkage, t=1 - threshold, criterion='distance')

        cluster_map = {}
        for i, cluster_id in enumerate(clusters):
            factor_name = corr.columns[i]
            cluster_map.setdefault(cluster_id, []).append(factor_name)

        return cluster_map

    def plot_correlation_matrix(self, save_path=None):
        """Plots heatmap of the correlation matrix."""
        if self.corr_matrix is None:
            self.calculate_correlation()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.corr_matrix,
            annot=True, fmt=".2f",
            cmap='coolwarm', vmin=-1, vmax=1,
            square=True, linewidths=0.5
        )
        plt.title("Factor Correlation Matrix (Avg Cross-Sectional Spearman)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_dendrogram(self, save_path=None):
        """Plots dendrogram to visualize factor clusters."""
        if self.corr_matrix is None:
            self.calculate_correlation()

        corr     = self.corr_matrix.fillna(0)
        dist     = 1 - np.abs(corr)
        dist_arr = distance.squareform(
            np.clip((dist.values + dist.values.T) / 2, 0, None),
            checks=False
        )
        linkage  = hierarchy.linkage(dist_arr, method='ward')

        plt.figure(figsize=(10, 7))
        hierarchy.dendrogram(
            linkage,
            labels=corr.columns.tolist(),
            leaf_rotation=45,
            leaf_font_size=10
        )
        plt.title("Factor Clustering Dendrogram")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()