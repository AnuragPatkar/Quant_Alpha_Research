"""
quant_alpha/backtest/attribution.py
=====================================
Performance attribution utilities used by run_backtest.py.

Provides:
  - SimpleAttribution   : PnL decomposition from a trades DataFrame.
  - FactorAttribution   : Rolling IC and raw daily IC from factor/return panels.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Simple PnL Attribution
# ---------------------------------------------------------------------------

class SimpleAttribution:
    """
    Decomposes realised PnL from a trades log into directional and win/loss
    statistics.  Expects the trades DataFrame produced by BacktestEngine.run().
    """

    def analyze_pnl_drivers(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute basic PnL attribution statistics from a trades log.

        Parameters
        ----------
        trades : DataFrame with at minimum columns:
                 ['ticker', 'direction', 'pnl']
                 direction ∈ {'long', 'short'} or {1, -1}
                 pnl = realised profit / loss per trade in dollar terms.

        Returns
        -------
        Dict with keys:
          hit_ratio, win_loss_ratio, long_pnl_contribution,
          short_pnl_contribution, total_pnl, n_trades.
        """
        if trades is None or trades.empty:
            return {
                "hit_ratio":               0.0,
                "win_loss_ratio":          0.0,
                "long_pnl_contribution":   0.0,
                "short_pnl_contribution":  0.0,
                "total_pnl":               0.0,
                "n_trades":                0,
            }

        df = trades.copy()

        # Normalise 'pnl' column name — engine may use 'trade_pnl' or 'pnl'
        pnl_col = next(
            (c for c in ("pnl", "trade_pnl", "profit", "realised_pnl") if c in df.columns),
            None,
        )
        if pnl_col is None:
            return {"error": "No PnL column found in trades DataFrame."}

        pnl = df[pnl_col].dropna().values.astype(float)

        n_trades    = len(pnl)
        n_winners   = int((pnl > 0).sum())
        n_losers    = int((pnl < 0).sum())
        hit_ratio   = n_winners / n_trades if n_trades > 0 else 0.0

        avg_win  = float(pnl[pnl > 0].mean()) if n_winners > 0 else 0.0
        avg_loss = float(abs(pnl[pnl < 0].mean())) if n_losers > 0 else 1e-12
        win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-12 else 0.0

        # Directional split
        dir_col = next(
            (c for c in ("direction", "side", "trade_direction") if c in df.columns),
            None,
        )
        if dir_col is not None:
            direction = df[dir_col].str.lower() if df[dir_col].dtype == object else df[dir_col]
            long_mask  = direction.isin(["long",  "buy",  1])
            short_mask = direction.isin(["short", "sell", -1])
            long_pnl  = float(df.loc[long_mask,  pnl_col].sum())
            short_pnl = float(df.loc[short_mask, pnl_col].sum())
        else:
            long_pnl  = float(pnl[pnl >= 0].sum())
            short_pnl = float(pnl[pnl <  0].sum())

        return {
            "hit_ratio":              round(hit_ratio, 4),
            "win_loss_ratio":         round(win_loss_ratio, 4),
            "long_pnl_contribution":  round(long_pnl,  2),
            "short_pnl_contribution": round(short_pnl, 2),
            "total_pnl":              round(float(pnl.sum()), 2),
            "n_trades":               n_trades,
        }


# ---------------------------------------------------------------------------
# Factor IC Attribution
# ---------------------------------------------------------------------------

class FactorAttribution:
    """
    Computes rolling and raw daily Information Coefficients between a factor
    panel and a forward-return panel.  Used in the IC analysis section of
    run_backtest.py.
    """

    def calculate_rolling_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: int = 30,
        method: str = "spearman",
    ) -> pd.Series:
        """
        Compute a rolling-mean IC series.

        Parameters
        ----------
        factor_values   : DataFrame indexed by (date, ticker) with one factor column.
        forward_returns : DataFrame indexed by (date, ticker) with one return column.
        window          : Rolling window in trading days.
        method          : 'spearman' (default) or 'pearson'.

        Returns
        -------
        pd.Series : Rolling mean IC, indexed by date.
        """
        raw_ic = self.calculate_raw_ic(factor_values, forward_returns, method=method)
        return raw_ic.rolling(window=window, min_periods=max(1, window // 2)).mean()

    def calculate_raw_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.Series:
        """
        Compute the daily cross-sectional IC series.

        For each date, correlate the factor values across tickers with the
        forward returns across tickers.

        Parameters
        ----------
        factor_values   : DataFrame indexed by (date, ticker), one factor column.
        forward_returns : DataFrame indexed by (date, ticker), one return column.
        method          : 'spearman' (default) or 'pearson'.

        Returns
        -------
        pd.Series : Daily IC, indexed by date.  NaN for dates with < 5 tickers.
        """
        from scipy.stats import spearmanr, pearsonr

        # Align on common index
        f_col = factor_values.columns[0]
        r_col = forward_returns.columns[0]

        merged = factor_values[[f_col]].join(
            forward_returns[[r_col]], how="inner"
        ).dropna()

        if merged.empty:
            return pd.Series(dtype=float, name="ic")

        date_level = merged.index.get_level_values("date")
        unique_dates = date_level.unique().sort_values()
        ics = {}

        for date in unique_dates:
            grp = merged.loc[date]
            if len(grp) < 5:
                ics[date] = np.nan
                continue
            f = grp[f_col].values.astype(float)
            r = grp[r_col].values.astype(float)
            mask = np.isfinite(f) & np.isfinite(r)
            if mask.sum() < 5:
                ics[date] = np.nan
                continue
            if method == "spearman":
                ic, _ = spearmanr(f[mask], r[mask])
            else:
                ic, _ = pearsonr(f[mask], r[mask])
            ics[date] = float(ic) if np.isfinite(ic) else np.nan

        return pd.Series(ics, name="ic")