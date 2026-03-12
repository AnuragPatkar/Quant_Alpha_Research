"""
Factor Correlation & Redundancy Analysis Engine
===============================================
Diagnostic suite for evaluating the orthogonality of alpha signals.

Purpose
-------
The `FactorCorrelator` quantifies the linear and non-linear dependencies between multiple
alpha factors. It employs cross-sectional Spearman Rank Correlation averaged over time
to construct a robust correlation matrix, followed by Hierarchical Clustering (Ward's Method)
to identify redundant signal clusters.

Usage
-----
.. code-block:: python

    # Initialize with a MultiIndex DataFrame (date, ticker) -> factors
    correlator = FactorCorrelator(factor_data)

    # 1. Compute Cross-Sectional Correlation Matrix
    corr_matrix = correlator.calculate_correlation(method='spearman')

    # 2. Identify Redundant Clusters (Distance < 0.5)
    clusters = correlator.cluster_factors(threshold=0.5)

    # 3. Visualize
    correlator.plot_dendrogram()

Importance
----------
- **Multicollinearity Detection**: High correlation between factors implies they are proxies
  for the same underlying phenomenon, leading to concentrated risk rather than diversification.
- **Dimensionality Reduction**: Identifying clusters allows for pruning redundant factors
  before they enter the ML pipeline, improving model stability.

Tools & Frameworks
------------------
- **Pandas**: Efficient MultiIndex grouping and split-apply-combine operations.
- **SciPy (cluster/spatial)**: Hierarchical clustering (Linkage/Dendrogram) and distance metrics.
- **Seaborn/Matplotlib**: Heatmap and dendrogram visualization.
"""

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
    Engine for analyzing the covariance structure and redundancy of factor sets.
    """

    def __init__(self, factor_data: pd.DataFrame):
        """
        Initialize the correlator.

        Args:
            factor_data (pd.DataFrame): Input data with MultiIndex (date, ticker) containing factor columns.
        """
        self.data = factor_data
        self.corr_matrix = None

    def calculate_correlation(self, method='spearman'):
        """
        Calculates the Time-Averaged Cross-Sectional Correlation Matrix.

        Methodology:
        1.  Compute the correlation matrix $C_t$ for all factors at each time step $t$ (cross-section).
        2.  Average these matrices over time to obtain the final estimate $\bar{C}$.
            .. math:: \bar{C}_{ij} = \frac{1}{T} \sum_{t=1}^T \text{Corr}(F_{i,t}, F_{j,t})

        Args:
            method (str): Correlation metric ('spearman' for Rank IC, 'pearson' for Linear).

        Returns:
            pd.DataFrame: Symmetric $(N \times N)$ correlation matrix.
        """
        logger.info(f"Calculating {method} correlation matrix...")

        def _daily_corr(df):
            # Minimum sample size check to ensure statistical significance of the snapshot
            if len(df) < 5:
                return pd.DataFrame()
            return df.corr(method=method)

        # Split-Apply-Combine: Calculate snapshot correlation per date
        daily_corrs = self.data.groupby(level='date').apply(_daily_corr)

        # Aggregation: Average over the time dimension (Level 0: date)
        # The resulting DataFrame preserves the factor names in indices and columns.
        if isinstance(daily_corrs.index, pd.MultiIndex):
            self.corr_matrix = daily_corrs.groupby(level=1).mean()
        else:
            self.corr_matrix = daily_corrs

        # Numerical Stability: Enforce strict diagonal unity to correct floating-point epsilon errors.
        np.fill_diagonal(self.corr_matrix.values, 1.0)

        return self.corr_matrix

    def cluster_factors(self, threshold=0.5):
        """
        Performs Hierarchical Clustering to identify redundant factor groups.

        Uses **Ward's Method** on a distance matrix derived from absolute correlation:
        .. math:: d_{ij} = 1 - |\\rho_{ij}|

        Args:
            threshold (float): Distance cut-off. Factors with correlation $> (1 - \text{threshold})$ are grouped.

        Returns:
            Dict[int, List[str]]: Mapping of Cluster ID to list of constituent factors.
        """
        if self.corr_matrix is None:
            self.calculate_correlation()

        # Construct Distance Matrix: Map correlation [-1, 1] to distance [0, 1]
        corr = self.corr_matrix.fillna(0)
        dist = 1 - np.abs(corr)

        # Topology Requirement: Ensure matrix is strictly symmetric and diagonal is zero
        dist_arr = distance.squareform(
            np.clip((dist.values + dist.values.T) / 2, 0, None),
            checks=False
        )

        # Hierarchical Clustering (Ward's variance minimization)
        linkage = hierarchy.linkage(dist_arr, method='ward')
        clusters = hierarchy.fcluster(linkage, t=1 - threshold, criterion='distance')

        cluster_map = {}
        for i, cluster_id in enumerate(clusters):
            factor_name = corr.columns[i]
            cluster_map.setdefault(cluster_id, []).append(factor_name)

        return cluster_map

    def plot_correlation_matrix(self, save_path=None):
        """
        Visualizes the Factor Correlation Matrix as a heatmap.

        Args:
            save_path (Optional[str]): File path to save the plot image.
        """
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
        """
        Visualizes the Hierarchical Clustering Dendrogram.
        Useful for determining the optimal number of latent clusters.

        Args:
            save_path (Optional[str]): File path to save the plot image.
        """
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