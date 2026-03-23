"""
Real-Time Strategy Performance & Risk Telemetry
===============================================
Stateful engine for continuous evaluation of portfolio attribution, signal efficacy,
and risk exposure.

Purpose
-------
The `PerformanceTracker` aggregates daily trading results to compute rolling time-series
metrics. It acts as the "Flight Recorder" for the strategy, providing instantaneous
feedback on:
1.  **Signal Quality**: Information Coefficient (IC) decay and statistical significance.
2.  **Risk-Adjusted Returns**: Sharpe Ratio, Sortino Ratio (implied), and Max Drawdown.
3.  **Factor Attribution**: Decomposition of returns into Market Beta and Idiosyncratic Alpha.
4.  **Operational Efficiency**: Monitoring of turnover rates and transaction cost drag.

Usage
-----
Typically instantiated within a backtest loop or live trading supervisor.

.. code-block:: python

    tracker = PerformanceTracker(window_days=60)
    tracker.update(
        date='2024-01-15',
        predictions={'AAPL': 0.05, ...},
        actual_returns={'AAPL': 0.04, ...},
        portfolio_return=0.015,
        benchmark_return=0.010
    )
    status = tracker.get_status()

Importance
----------
- **Alpha Preservation**: Detects "regime shifts" where the correlation between signals
  and returns (IC) breaks down, triggering circuit breakers.
- **Risk Control**: Enforces hard constraints on Drawdown and Volatility.
- **Attribution**: Distinguishes between skill (Alpha) and luck (Beta/Market drift).

Tools & Frameworks
------------------
- **Pandas/NumPy**: Vectorized time-series analysis and covariance estimation.
- **SciPy**: Spearman Rank Correlation for non-linear dependence measuring.
- **Matplotlib**: Visualization of equity curves and exposure profiles.

FIXES
-----
  BUG-077 (HIGH): _calculate_current_drawdown() used `== 0` float equality to
           guard against division by zero in the drawdown denominator.
           `cummax.iloc[-1]` is a floating-point cumulative product — exact
           equality to 0 is almost never true for a valid equity curve, so
           the guard silently fails, and NaN/Inf can propagate when the
           portfolio value is extremely small (near-zero after ruin).
           Fixed: replaced `== 0` with `< 1e-12` which correctly catches
           any degenerate near-zero denominator regardless of floating-point
           rounding.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Deque
import logging
from datetime import datetime, timedelta
from collections import deque
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Maintains a rolling buffer of trading history to calculate online risk metrics.

    Implements a sliding window approach (O(1) amortized updates) to track
    non-stationary performance statistics.
    """

    def __init__(
        self,
        window_days: int = 60,
        ic_warning_threshold: float = 0.02,
        ic_critical_threshold: float = 0.01,
        dd_warning_threshold: float = 0.10,
        dd_critical_threshold: float = 0.15,
        max_history: int = 252,
        risk_free_rate: float = 0.04,
    ):
        """
        Args:
            window_days (int)           : Rolling window N for rolling statistics (e.g. 60 days).
            ic_warning_threshold (float): Signal quality floor triggering warnings.
            ic_critical_threshold (float): Signal quality floor triggering halts.
            dd_warning_threshold (float): Drawdown % triggering risk reduction.
            dd_critical_threshold (float): Drawdown % triggering liquidation.
            max_history (int)           : Buffer size for full history storage.
            risk_free_rate (float)      : Annualized Rf for Sharpe calculation.
        """
        self.window_days  = window_days
        self.ic_warning   = ic_warning_threshold
        self.ic_critical  = ic_critical_threshold
        self.dd_warning   = dd_warning_threshold
        self.dd_critical  = dd_critical_threshold
        self.max_history  = max_history
        self.risk_free_rate = risk_free_rate

        # State Management: Deque provides O(1) appends and auto-eviction for rolling windows.
        self.history: Deque = deque(maxlen=max_history)

        logger.info(
            f"PerformanceTracker initialized "
            f"({window_days}d window, max history {max_history})"
        )

    def update(
        self,
        date: str,
        predictions: Dict[str, float],
        actual_returns: Dict[str, float],
        portfolio_return: float,
        benchmark_return: float = 0.0,
        turnover: float = 0.0,
        transaction_costs: float = 0.0,
        long_exposure: float = 0.0,
        short_exposure: float = 0.0,
        sector_exposure: Optional[Dict[str, float]] = None,
    ):
        """
        Ingests daily trading results and computes point-in-time signal quality.

        Args:
            date (str)                 : Trading date (ISO-8601).
            predictions (Dict)         : Model alpha scores.
            actual_returns (Dict)      : Realized asset returns.
            portfolio_return (float)   : Net strategy return Rp.
            benchmark_return (float)   : Market index return Rb.
            turnover (float)           : Portfolio turnover ratio.
            transaction_costs (float)  : Slippage + Commissions.
            long_exposure (float)      : Gross Long Exposure.
            short_exposure (float)     : Gross Short Exposure.
            sector_exposure (Optional) : Exposure breakdown by sector.
        """
        # IC via Spearman Rank Correlation
        tickers = set(predictions.keys()) & set(actual_returns.keys())

        if len(tickers) < 2:
            daily_ic = 0.0
        else:
            pred_array   = np.array([predictions[t]     for t in tickers])
            actual_array = np.array([actual_returns[t]  for t in tickers])

            daily_ic, _ = spearmanr(pred_array, actual_array)

            if np.isnan(daily_ic):
                daily_ic = 0.0

        record = {
            'date':               pd.to_datetime(date),
            'ic':                 daily_ic,
            'portfolio_return':   portfolio_return,
            'benchmark_return':   benchmark_return,
            'active_return':      portfolio_return - benchmark_return,
            'turnover':           turnover,
            'transaction_costs':  transaction_costs,
            'long_exposure':      long_exposure,
            'short_exposure':     short_exposure,
            'num_positions':      len(tickers),
        }

        if sector_exposure:
            record['sector_exposure'] = sector_exposure

        self.history.append(record)
        logger.debug(
            f"Updated: {date}, IC={daily_ic:.4f}, Return={portfolio_return:.4f}"
        )

    def get_status(self) -> Dict:
        """
        Computes aggregate risk and performance metrics over the rolling window.

        Returns
        -------
        Dict — Comprehensive status report including Alpha, Beta, Sharpe, and Drawdown.
        """
        if not self.history:
            return {'status': 'NO_DATA'}

        df     = pd.DataFrame(list(self.history))
        recent = df.tail(self.window_days)

        valid_stats       = recent.dropna(subset=['portfolio_return', 'benchmark_return'])
        valid_stats_total = df.dropna(subset=['portfolio_return', 'benchmark_return'])

        ic_recent = recent['ic'].mean()
        ic_std    = recent['ic'].std()

        returns_recent   = valid_stats['portfolio_return']
        benchmark_recent = valid_stats['benchmark_return']

        active_ret_daily_rolling = recent['active_return'].mean()
        active_ret_rolling_annual = active_ret_daily_rolling * 252

        active_ret_daily_total  = df['active_return'].mean()
        active_ret_total_annual = active_ret_daily_total * 252

        sharpe = self._calculate_sharpe(returns_recent)

        if len(returns_recent) > 1 and benchmark_recent.std() > 1e-8:
            covariance = np.cov(returns_recent, benchmark_recent)[0, 1]
            variance   = benchmark_recent.var()
            beta       = covariance / variance
            alpha      = (
                (returns_recent.mean() - self.risk_free_rate / 252)
                - beta * (benchmark_recent.mean() - self.risk_free_rate / 252)
            ) * 252
        else:
            beta  = 0.0
            alpha = 0.0

        if len(valid_stats_total) > 1 and valid_stats_total['benchmark_return'].std() > 1e-8:
            cov_t   = np.cov(
                valid_stats_total['portfolio_return'],
                valid_stats_total['benchmark_return'],
            )[0, 1]
            var_t      = valid_stats_total['benchmark_return'].var()
            beta_total = cov_t / var_t
            alpha_total = (
                (valid_stats_total['portfolio_return'].mean() - self.risk_free_rate / 252)
                - beta_total * (valid_stats_total['benchmark_return'].mean() - self.risk_free_rate / 252)
            ) * 252
        else:
            beta_total  = 0.0
            alpha_total = 0.0

        cum_returns = (1 + df['portfolio_return']).cumprod()
        current_dd  = self._calculate_current_drawdown(cum_returns)
        max_dd      = self._calculate_max_drawdown(cum_returns)

        if ic_recent < self.ic_critical or current_dd > self.dd_critical:
            status = 'CRITICAL'
        elif ic_recent < self.ic_warning or current_dd > self.dd_warning:
            status = 'WARNING'
        else:
            status = 'HEALTHY'

        return {
            'status':                       status,
            'date':                         df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'ic_rolling':                   ic_recent,
            'ic_std':                       ic_std,
            'sharpe_rolling':               sharpe,
            'beta':                         beta,
            'alpha_annual':                 alpha,
            'alpha_total':                  alpha_total,
            'beta_total':                   beta_total,
            'active_return_rolling_annual': (
                active_ret_rolling_annual
                if not np.isnan(active_ret_rolling_annual) else 0.0
            ),
            'active_return_total_annual':   (
                active_ret_total_annual
                if not np.isnan(active_ret_total_annual) else 0.0
            ),
            'current_drawdown':             current_dd,
            'max_drawdown':                 max_dd,
            'avg_turnover':                 (
                recent['turnover'].mean()
                if 'turnover' in recent.columns else 0.0
            ),
            'total_costs':                  (
                recent['transaction_costs'].sum()
                if 'transaction_costs' in recent.columns else 0.0
            ),
            'num_days':                     len(recent),
            'avg_positions':                recent['num_positions'].mean(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        r"""
        Calculates Annualized Sharpe Ratio.

        S = (mean_excess * 252) / (std_excess * sqrt(252))
          = mean_excess * sqrt(252) / std_excess
        """
        if len(returns) < 2:
            return 0.0

        daily_rf      = self.risk_free_rate / 252
        excess        = returns - daily_rf
        mean_excess   = excess.mean() * 252
        std_excess    = excess.std() * np.sqrt(252)

        if std_excess > 1e-9:
            return float(mean_excess / std_excess)
        return 0.0

    def _calculate_current_drawdown(self, cum_returns: pd.Series) -> float:
        """
        Calculates percentage decline from the High-Water Mark (HWM).

        FIX BUG-077: Original code used `== 0` to guard the denominator.
        Float equality is almost never true for a floating-point cumulative
        product — the guard was effectively dead code. A near-zero HWM
        (e.g. after a near-ruin scenario) produces Inf or NaN.
        Fixed: use `< 1e-12` which catches any degenerate near-zero value.
        """
        cummax = cum_returns.cummax()
        hwm    = cummax.iloc[-1]

        # FIX BUG-077: was `if cummax.iloc[-1] == 0` — float equality fails here
        if hwm < 1e-12:
            return 0.0

        current_dd = (cum_returns.iloc[-1] - hwm) / hwm
        return float(abs(current_dd))

    def _calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """Calculates Maximum Drawdown (MDD) over the entire history."""
        cummax   = cum_returns.cummax()
        # Guard against zero/near-zero HWM to avoid division by zero
        safe_max = cummax.replace(0.0, np.nan)
        drawdown = (cum_returns - cummax) / safe_max
        return float(abs(drawdown.min()))

    def get_history_df(self) -> pd.DataFrame:
        """Exports the full transaction and performance log as a DataFrame."""
        return pd.DataFrame(list(self.history))

    def plot_performance(self, save_path: str = None):
        """
        Generates a 3-panel performance report visualization.

        Panels:
        1. Rolling IC (N-day moving average) vs Warning Thresholds.
        2. Cumulative Returns (Portfolio vs Benchmark).
        3. Turnover & Cumulative Cost Impact.
        """
        import matplotlib.pyplot as plt

        if not save_path:
            logger.warning("No save_path provided for plot_performance. Skipping.")
            return

        df = pd.DataFrame(list(self.history))
        if df.empty:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 15))

        # Panel 1 — Rolling IC
        rolling_ic = df['ic'].rolling(window=20).mean()
        axes[0].plot(df['date'], rolling_ic, label='20-day MA')
        axes[0].axhline(y=self.ic_warning,  color='orange', linestyle='--', label='Warning')
        axes[0].axhline(y=self.ic_critical, color='red',    linestyle='--', label='Critical')
        axes[0].set_title('Rolling IC (Spearman)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Panel 2 — Cumulative returns vs Benchmark
        cum_returns = (1 + df['portfolio_return']).cumprod() - 1
        cum_bench   = (1 + df['benchmark_return']).cumprod() - 1
        axes[1].plot(df['date'], cum_returns * 100, label='Portfolio')
        axes[1].plot(df['date'], cum_bench   * 100, label='Benchmark', alpha=0.7)
        axes[1].set_title('Cumulative Returns (%)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Panel 3 — Turnover & Costs
        if 'turnover' in df.columns:
            ax2 = axes[2].twinx()
            axes[2].bar(
                df['date'], df['turnover'] * 100,
                alpha=0.3, color='gray', label='Turnover %',
            )
            ax2.plot(
                df['date'], df['transaction_costs'].cumsum(),
                color='red', label='Cum Costs',
            )
            axes[2].set_title('Turnover & Transaction Costs', fontweight='bold')
            axes[2].set_ylabel('Turnover (%)')
            ax2.set_ylabel('Cumulative Costs')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance chart saved → {save_path}")