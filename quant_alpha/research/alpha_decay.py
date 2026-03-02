import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class AlphaDecayAnalyzer:
    """
    Analyzes how the predictive power (IC) of a factor decays over longer horizons.
    """

    def __init__(self, data: pd.DataFrame, factor_col: str):
        """
        data: DataFrame with 'date', 'ticker', 'factor_col', and 'close' price.
        """
        self.data         = data.copy()   # BUG-3 FIX: copy at init to prevent mutation
        self.factor_col   = factor_col
        self.decay_results = {}

    def calculate_decay(self, max_horizon=10):
        """
        Calculates Rank IC for horizons 1 to max_horizon days.

        BUG-3 FIX: Method was modifying self.data in-place by adding fwd_ret_Nd columns.
        Calling twice caused columns to accumulate (fwd_ret_1d, fwd_ret_2d already exist
        on second call → shift(-h) applied to wrong data).
        Fix: work on a local copy, never mutate self.data.
        """
        logger.info(f"Calculating Alpha Decay up to {max_horizon} days...")

        # BUG-3 FIX: work on local copy — never mutate self.data
        work_df = self.data.sort_values(['ticker', 'date']).copy()

        self.decay_results = {}

        for h in range(1, max_horizon + 1):
            col_name = f'fwd_ret_{h}d'

            # Calculate forward return on local copy only
            work_df[col_name] = (
                work_df.groupby('ticker')['close'].shift(-h)
                / work_df['close'] - 1
            )

            valid = work_df.dropna(subset=[self.factor_col, col_name])

            if valid.empty:
                self.decay_results[h] = 0.0
                continue

            daily_ic = valid.groupby('date').apply(
                lambda x: stats.spearmanr(
                    x[self.factor_col], x[col_name]
                )[0] if len(x) >= 5 else np.nan
            )

            self.decay_results[h] = daily_ic.mean()

            # Drop the column from work_df after use — keeps memory clean
            work_df.drop(columns=[col_name], inplace=True)

        return self.decay_results

    def get_half_life(self):
        """
        Returns the horizon at which IC drops to half of Day-1 IC.
        Useful for determining optimal holding period.
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
        """Plots the IC decay curve with half-life annotation."""
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

        # Half-life line
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