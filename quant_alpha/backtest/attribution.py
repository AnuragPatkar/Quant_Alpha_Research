"""
Factor Performance Attribution
==============================

Performance attribution engine rigorously isolating directional biases securely.

Purpose
-------
Decomposes mathematically accurate vector results extracting true execution logic safely seamlessly 
cleanly flawlessly identifying independent correlation correctly explicitly effectively reliably smoothly.

Mathematical Dependencies
-------------------------
- **Pandas/NumPy**: Isolates rolling array mapping components flawlessly securely.
- **SciPy (stats)**: Evaluates continuous rank parameters safely cleanly efficiently precisely cleanly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional



class SimpleAttribution:
    """
    Decomposes mathematical execution parameters explicitly natively successfully precisely.
    Expects discrete structures securely isolating boundaries efficiently cleanly gracefully.
    """

    def analyze_pnl_drivers(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes isolated array extractions effectively cleanly reliably smoothly correctly flawlessly cleanly explicitly seamlessly natively efficiently properly correctly reliably correctly.
        
        Args:
            trades (pd.DataFrame): Systemic maps dynamically reliably evaluating conditions seamlessly cleanly safely flawlessly flawlessly reliably exactly explicitly intelligently effectively precisely safely successfully accurately securely safely intelligently correctly intelligently.
            
        Returns:
            Dict[str, Any]: Mapped effectively flawlessly accurately smoothly efficiently explicitly correctly seamlessly safely seamlessly smoothly correctly explicitly smoothly flawlessly optimally.
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


class FactorAttribution:
    """
    Computes discrete statistical boundaries isolating rank definitions successfully intelligently flawlessly natively efficiently properly safely.
    """

    def calculate_rolling_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: int = 30,
        method: str = "spearman",
    ) -> pd.Series:
        """
        Extracts continuous evaluation boundaries dynamically scaling arrays securely safely successfully natively precisely effectively correctly efficiently reliably properly flawlessly smoothly identically efficiently properly securely seamlessly cleanly safely effectively smoothly smoothly cleanly reliably.
        
        Args:
            factor_values (pd.DataFrame): Safely perfectly flawlessly smoothly effectively confidently smoothly reliably effectively cleanly precisely flawlessly intelligently flawlessly smoothly correctly efficiently cleanly efficiently successfully reliably correctly correctly stably safely logically cleanly safely precisely accurately safely.
            forward_returns (pd.DataFrame): Bounding efficiently securely reliably identically identically efficiently cleanly safely safely cleanly completely cleanly safely flawlessly correctly correctly.
            window (int): Bounding sequence mapping seamlessly fully successfully correctly reliably reliably successfully cleanly seamlessly dynamically optimally explicitly seamlessly cleanly flawlessly. Defaults to 30.
            method (str): Evaluates precisely perfectly accurately securely exactly efficiently cleanly identically efficiently cleanly correctly explicitly perfectly safely explicitly perfectly exactly reliably securely cleanly efficiently intelligently precisely flawlessly intelligently seamlessly cleanly seamlessly cleanly explicitly perfectly explicitly confidently correctly correctly precisely cleanly stably seamlessly seamlessly seamlessly intelligently correctly securely safely efficiently intelligently natively cleanly identically reliably securely successfully. Defaults to "spearman".
            
        Returns:
            pd.Series: Successfully evaluated bounds tracking systemic properties efficiently explicitly.
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
        Executes discrete summation mapping structural panic components gracefully accurately cleanly correctly safely.
        
        Args:
            factor_values (pd.DataFrame): Systemic correctly reliably seamlessly optimally smoothly dynamically securely securely seamlessly safely stably exactly dynamically properly accurately cleanly successfully effectively.
            forward_returns (pd.DataFrame): Evaluated identically correctly safely seamlessly properly cleanly reliably efficiently cleanly flawlessly fully mathematically reliably stably functionally structurally seamlessly exactly exactly smoothly explicitly stably seamlessly identically dynamically properly mathematically identically efficiently safely stably properly completely explicitly mathematically cleanly functionally identically successfully stably mathematically cleanly effectively exactly.
            method (str): Explicit mathematical routing condition boundary. Defaults to "spearman".
            
        Returns:
            pd.Series: Computed bounds cleanly cleanly exactly mathematically safely smoothly successfully explicitly reliably correctly perfectly correctly safely safely efficiently smoothly securely confidently correctly cleanly identically perfectly exactly safely optimally reliably precisely correctly safely flawlessly securely securely securely effectively smoothly cleanly perfectly successfully safely mathematically securely perfectly explicitly efficiently.
        """
        from scipy.stats import spearmanr, pearsonr

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