"""
Market Regime Detection Engine
==============================
Identifies latent market states using heuristic and probabilistic models.

Purpose
-------
The `RegimeDetector` classifies market environments into distinct regimes (e.g.,
Bull/Bear, High/Low Volatility). Strategies often perform differently depending
on the regime; this module provides the signals to switch logic dynamically
(Regime Switching models).

It offers two approaches:
1.  **Heuristic**: Deterministic intersection of Trend (Moving Average) and Volatility filters.
2.  **Probabilistic**: Gaussian Hidden Markov Model (HMM) to infer latent states from observed returns.

Usage
-----
.. code-block:: python

    detector = RegimeDetector(benchmark_prices=spy_close)

    # 1. Heuristic: Trend + Volatility Quadrants
    regimes = detector.detect_trend_vol_regime(trend_window=200, vol_window=20)

    # 2. Probabilistic: Hidden Markov Model (Unsupervised)
    hmm_states = detector.detect_hmm_regime(n_components=2)

Importance
----------
-   **Risk Management**: "Risk-off" signals during High Volatility/Bear states prevent
    large drawdowns ($MaxDD$).
-   **Alpha Preservation**: Momentum strategies often fail in mean-reverting (choppy) regimes;
    identifying these states prevents whipsaw losses.
-   **Complexity**: The vectorised heuristic runs in $O(N)$, while the HMM relies on
    Expectation-Maximization (Baum-Welch) with complexity $O(N \cdot K^2)$.

Tools & Frameworks
------------------
-   **hmmlearn**: Implementation of Hidden Markov Models with Gaussian Emissions.
-   **Pandas/NumPy**: Time-series alignment and vectorized conditional logic (`np.select`).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Engine for classifying market conditions into discrete states.
    """

    def __init__(self, benchmark_prices: pd.Series):
        """
        Initializes the state identification constraint models against standard geometric vectors.

        Args:
            benchmark_prices (pd.Series): Time-series of close prices (e.g., SPY).
        """
        self.prices = benchmark_prices.sort_index()
        self.regimes = None

    def detect_trend_vol_regime(self, trend_window=200, vol_window=20):
        """
        Evaluates geometric conditions bounding market environments using a deterministic 
        4-Quadrant computational heuristic framework.

        Logic:
        1.  **Trend**: Price > Simple Moving Average ($SMA_{200}$).
        2.  **Volatility**: Realized Volatility > Median Historical Volatility.

        States:
        -   **Bull_LowVol**: Ideal conditions for leverage/beta.
        -   **Bull_HighVol**: Risky Rally (possible top formation).
        -   **Bear_LowVol**: Slow Bleed / Stagnation.
        -   **Bear_HighVol**: Crash / Panic (Risk-Off).

        Returns:
            pd.Series: Linear topological array explicitly assigning strict classification bounds.
        """
        # Identifies baseline trajectory dynamics projecting strict simple moving average maps
        ma = self.prices.rolling(trend_window).mean()
        trend_signal = (self.prices > ma).astype(int)  # 1=Bull, 0=Bear

        # Evaluates structural historical variance metrics comparing to normalized median bounds
        returns = self.prices.pct_change()
        vol = returns.rolling(vol_window).std() * np.sqrt(252)
        vol_threshold = vol.rolling(252).median()
        vol_signal = (vol > vol_threshold).astype(int)  # 1=HighVol, 0=LowVol

        self.regimes = pd.DataFrame({
            'trend':     trend_signal,
            'vol_state': vol_signal,
            'vol_val':   vol,
        }, index=self.prices.index)

        # Instantiates high-performance parallel logic bypassing O(N) row-wise assignment iterations, 
        # projecting direct dimensional bounds optimally scaled for memory buffers.
        conditions = [
            (self.regimes['trend'] == 1) & (self.regimes['vol_state'] == 0),
            (self.regimes['trend'] == 1) & (self.regimes['vol_state'] == 1),
            (self.regimes['trend'] == 0) & (self.regimes['vol_state'] == 0),
            (self.regimes['trend'] == 0) & (self.regimes['vol_state'] == 1),
        ]
        choices = ['Bull_LowVol', 'Bull_HighVol', 'Bear_LowVol', 'Bear_HighVol']
        self.regimes['regime'] = np.select(conditions, choices, default='Unknown')

        return self.regimes['regime']

    def detect_hmm_regime(self, n_components=2):
        """
        Architects systemic probability inferences spanning latent regime distributions via 
        a mathematically structural Gaussian Hidden Markov Model (HMM).

        Model Assumption:
        Enforces assumptions mapping discrete return probabilities as a functional mixture of 
        continuous Gaussian derivations intrinsically dependent on latent state Markov variables ($S_t$).

        .. math::
            P(R_t | S_t=k) \\sim \\mathcal{N}(\\mu_k, \\sigma_k^2)

        Args:
            n_components (int): Target state matrix dimension bounding output probabilities. Defaults to 2.

        Returns:
            Optional[pd.Series]: Discrete integer representation mapping the deterministic hidden state (0 to K-1), 
            or None if external dependencies resolve negatively.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed. Run: pip install hmmlearn")
            return None

        returns = self.prices.pct_change().dropna()

        # Resolves coordinate matrices explicitly stripping null initialization parameters inherent 
        # to trailing boundary difference equations.
        dates = returns.index
        values = returns.values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(values)
        hidden_states = model.predict(values)

        # Commits state persistence logic explicitly unifying scalar alignments dynamically mapped to execution grids.
        if self.regimes is None:
            self.regimes = pd.DataFrame(index=dates)
        self.regimes = self.regimes.reindex(self.regimes.index.union(dates))
        self.regimes.loc[dates, 'hmm_state'] = hidden_states

        return pd.Series(hidden_states, index=dates, name='hmm_state')

    def plot_regimes(self, save_path=None):
        """
        Constructs graphical coordinates mapping standardized time-series trajectories overlaying discrete latent boundary assignments.

        Args:
            save_path (Optional[str]): Filepath to force disk persistence maps.
            
        Returns:
            None: Dynamically projects plots locally explicitly manipulating Matplotlib configurations.
        """
        if self.regimes is None or 'regime' not in self.regimes.columns:
            logger.warning("Run detect_trend_vol_regime first.")
            return

        color_map = {
            'Bull_LowVol':  'green',
            'Bull_HighVol': 'yellow',
            'Bear_LowVol':  'orange',
            'Bear_HighVol': 'red',
            'Unknown':      'gray',
        }

        plt.figure(figsize=(14, 6))
        plt.plot(self.prices.index, self.prices,
                 color='black', alpha=0.3, linewidth=0.8, label='Price')

        for regime, color in color_map.items():
            mask = self.regimes['regime'] == regime
            if mask.any():
                plt.scatter(
                    self.prices[mask].index,
                    self.prices[mask],
                    color=color, s=2, label=regime, alpha=0.6
                )

        plt.title("Market Regimes (Trend × Volatility)")
        plt.legend(markerscale=5, loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()