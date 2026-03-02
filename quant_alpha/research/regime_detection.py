import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes based on Trend and Volatility.
    """

    def __init__(self, benchmark_prices: pd.Series):
        """
        benchmark_prices: Series of S&P 500 (or equivalent) close prices.
        """
        self.prices  = benchmark_prices.sort_index()
        self.regimes = None

    def detect_trend_vol_regime(self, trend_window=200, vol_window=20):
        """
        Classifies regimes into 4 quadrants:
        1. Bull_LowVol  — Ideal
        2. Bull_HighVol — Risky Rally
        3. Bear_LowVol  — Slow Bleed
        4. Bear_HighVol — Crash

        BUG-2 FIX: Replaced slow .apply(axis=1) with vectorized np.select()
        ~10x faster on large data.
        """
        ma           = self.prices.rolling(trend_window).mean()
        trend_signal = (self.prices > ma).astype(int)   # 1=Bull, 0=Bear

        returns      = self.prices.pct_change()
        vol          = returns.rolling(vol_window).std() * np.sqrt(252)
        vol_threshold = vol.rolling(252).median()
        vol_signal   = (vol > vol_threshold).astype(int) # 1=HighVol, 0=LowVol

        self.regimes = pd.DataFrame({
            'trend':     trend_signal,
            'vol_state': vol_signal,
            'vol_val':   vol,
        }, index=self.prices.index)

        # BUG-2 FIX: vectorized np.select instead of .apply(axis=1)
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
        Uses Gaussian HMM to detect latent regimes.
        Requires: pip install hmmlearn

        BUG-5 FIX: Use returns.index after dropna() instead of prices.index[1:]
        prices.index[1:] assumed exactly 1 NaN from pct_change — not guaranteed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed. Run: pip install hmmlearn")
            return None

        returns = self.prices.pct_change().dropna()

        # BUG-5 FIX: use returns.index directly — not prices.index[1:]
        # pct_change().dropna() may drop more than 1 row if NaNs exist in prices
        dates  = returns.index
        values = returns.values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        model.fit(values)
        hidden_states = model.predict(values)

        # Align with correct dates
        if self.regimes is None:
            self.regimes = pd.DataFrame(index=dates)
        self.regimes = self.regimes.reindex(self.regimes.index.union(dates))
        self.regimes.loc[dates, 'hmm_state'] = hidden_states

        return pd.Series(hidden_states, index=dates, name='hmm_state')

    def plot_regimes(self, save_path=None):
        """Plots price colored by regime."""
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