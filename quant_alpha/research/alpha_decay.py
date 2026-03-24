"""
Alpha Decay Analysis Engine
===========================
Quantifies the persistence of predictive power (Information Coefficient) over varying time horizons.

Purpose
-------
The `AlphaDecayAnalyzer` evaluates how quickly a factor's signal degrades. This is critical for
determining the optimal rebalancing frequency and estimating portfolio turnover. A factor with
rapid decay requires high-frequency trading (high transaction costs), while slow decay implies
capacity for larger AUM and lower turnover.

Usage
-----
.. code-block:: python

    # Initialize with price and factor data
    analyzer = AlphaDecayAnalyzer(data=factor_df, factor_col='momentum_rsi')

    # Compute IC decay profile up to 10 days
    decay_profile = analyzer.calculate_decay(max_horizon=10)

    # Estimate signal half-life (e.g., t where IC_t = 0.5 * IC_1)
    half_life = analyzer.get_half_life()

Importance
----------
- **Turnover Estimation**: Factors with short half-lives (< 5 days) drive high portfolio turnover,
  necessitating strict transaction cost controls ($TC < \\alpha$).
- **Alpha Persistence**: Measures the orthogonality of the signal across time lags:
  .. math::
      IC(h) = \\text{corr}(F_t, R_{t+h})
- **Execution Timing**: Helps determining if trade execution can be delayed without significant
  alpha loss (implementation shortfall mitigation).

Tools & Frameworks
------------------
- **Pandas**: Time-series manipulation and lag generation (`shift`).
- **SciPy (stats)**: Spearman Rank Correlation for non-linear dependency checks.
- **Matplotlib**: Visualization of the decay curve.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AlphaDecayAnalyzer:
    """
    Engine structurally extracting the temporal degradation of isolated alpha vectors.
    Computes the term structure of the Information Coefficient (IC) and explicitly 
    bounds the empirical signal half-life metric.
    """

    def __init__(self, data: pd.DataFrame, factor_col: str):
        """
        Initializes the mathematical evaluator bounds scaling continuous market and factor data.

        Args:
            data (pd.DataFrame): Input dataset containing 'date', 'ticker', 'close', and the factor column.
            factor_col (str): The name of the column containing the alpha signal.
        """
        # Architecturally forces rigorous state isolation bounding immutable copies 
        # strictly averting computational overlaps originating from destructive lag generation maps.
        self.data = data.copy()
        self.factor_col = factor_col
        self.decay_results = {}

    def calculate_decay(self, max_horizon=10):
        """
        Calculates the Information Coefficient (IC) profile across multiple time horizons.

        Generates forward returns $R_{t, t+h}$ for $h \in [1, \text{max\_horizon}]$
        and computes the Spearman Rank Correlation with the factor value at time $t$.

        Args:
            max_horizon (int): The temporal parameter dictating maximal forward evaluation drift.

        Returns:
            Dict[int, float]: Mapping of horizon (days) to Mean Rank IC.
        """
        logger.info(f"Calculating Alpha Decay up to {max_horizon} days...")

        # Asserts rigid idempotency mappings operating mathematically distinct continuous states, 
        # aggressively discarding accumulated forward lag matrix variables systematically.
        work_df = self.data.sort_values(['ticker', 'date']).copy()

        self.decay_results = {}

        for h in range(1, max_horizon + 1):
            col_name = f"fwd_ret_{h}d"

            # Extracts absolute scalar structural return bounds: R_{t+h} / R_t - 1
            work_df[col_name] = (
                work_df.groupby('ticker')['close'].shift(-h)
                / work_df['close'] - 1
            )

            # Exposes discrete mathematical intersections guaranteeing identical target mappings
            valid = work_df.dropna(subset=[self.factor_col, col_name])

            if valid.empty:
                self.decay_results[h] = 0.0
                continue

            # Evaluates structural average Cross-Sectional Rank IC natively mapped per date
            daily_ic = valid.groupby('date').apply(
                lambda x: stats.spearmanr(
                    x[self.factor_col], x[col_name]
                )[0] if len(x) >= 5 else np.nan
            )

            self.decay_results[h] = daily_ic.mean()

            # Immediately evicts extraneous memory dependencies structurally locking usage strictly down to O(N).
            work_df.drop(columns=[col_name], inplace=True)

        return self.decay_results

    def get_half_life(self):
        """
        Estimates the Alpha Half-Life.

        Defined as the time horizon $h$ where the magnitude of the Information Coefficient
        decays to 50% of its initial strength ($|IC_h| \leq 0.5 \times |IC_1|$).

        Returns:
            Optional[int]: The exact evaluated day $h$ wherein the specific geometric threshold boundary is strictly breached. 
            Returns None if the continuous vector natively persists explicitly.
        """
        if not self.decay_results:
            self.calculate_decay()

        ic_day1 = abs(self.decay_results.get(1, 0))
        if ic_day1 == 0:
            return None

        half_ic = ic_day1 / 2.0
        for h, ic in self.decay_results.items():
            if abs(ic) <= half_ic:
                return h
        return None  # IC never decays to half

    def plot_decay(self, save_path=None):
        """
        Visualizes the Alpha Decay Term Structure.

        Args:
            save_path (Optional[str]): The discrete string filepath to explicitly encode output to disk.
                If None, forces direct graphical execution bounds interactively.
                
        Returns:
            None: Emits graphical state arrays dynamically intercepting structural execution pipelines.
        """
        if not self.decay_results:
            self.calculate_decay()

        horizons = list(self.decay_results.keys())
        ics      = list(self.decay_results.values())
        half_life = self.get_half_life()

        plt.figure(figsize=(10, 6))
        plt.plot(horizons, ics, marker='o', linestyle='-',
                 color='purple', linewidth=2, markersize=6)
        plt.fill_between(horizons, ics, alpha=0.1, color='purple')
        plt.axhline(0, color='black', linewidth=0.5)

        # Annotation: Half-life threshold
        ic_day1 = ics[0] if ics else 0
        half_ic = ic_day1 / 2.0
        plt.axhline(half_ic, color='gray', linestyle='--',
                    alpha=0.7, label=f'Half IC ({half_ic:.4f})')

        if half_life:
            plt.axvline(half_life, color='red', linestyle=':',
                        alpha=0.7, label=f'Half-Life: Day {half_life}')

        plt.title(f"Alpha Decay: {self.factor_col}")
        plt.xlabel("Horizon (Days)")
        plt.ylabel("Rank IC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(horizons)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()