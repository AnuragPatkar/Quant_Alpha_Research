"""
Factor Attribution Module
Decomposes portfolio returns into Systematic Factors and Idiosyncratic Alpha.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class FactorAttribution:
    """
    Analyzes sources of portfolio performance.
    Decomposition: Total Return = Sum(Exposure * Factor_Return) + Selection (Alpha)
    """
    
    def __init__(self):
        logger.info("FactorAttribution Module Initialized")
    
    def analyze(
        self,
        portfolio_returns: pd.Series,
        factor_exposures: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Decomposes returns into factor-based and residual components.
        
        Args:
            portfolio_returns: Series with DatetimeIndex
            factor_exposures: DataFrame (Date index, Factor columns) - Portfolio weighted average exposure
            factor_returns: DataFrame (Date index, Factor columns) - Factor returns
        """
        if portfolio_returns.empty:
            return {"error": "Portfolio returns series is empty"}

        results = {
            'total_return': portfolio_returns.sum(),
            'annualized_return': portfolio_returns.mean() * 252,
            'annualized_vol': portfolio_returns.std() * np.sqrt(252)
        }
        
        # Factor Decomposition
        if factor_exposures is not None and factor_returns is not None:
            factor_results = self._calculate_factor_contribution(
                portfolio_returns, factor_exposures, factor_returns
            )
            results.update(factor_results)
        else:
            results['factor_contribution'] = {}
            results['residual_alpha'] = results['total_return']
            
        return results
    
    def _calculate_factor_contribution(
        self,
        portfolio_returns: pd.Series,
        exposures: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Dict:
        """
        Detailed calculation of factor contributions over time.
        Assumes exposures are lagged (t-1) aligned with returns (t).
        """
        # 1. Align Data
        # Ensure all inputs share the same index (dates)
        common_dates = portfolio_returns.index.intersection(exposures.index).intersection(factor_returns.index)
        
        if len(common_dates) == 0:
            logger.error("No overlapping data found for attribution analysis.")
            return {'factor_contribution': {}, 'residual_alpha': 0}

        p_ret = portfolio_returns.loc[common_dates]
        exp = exposures.loc[common_dates]
        f_ret = factor_returns.loc[common_dates]

        # 2. Calculate Contribution: Exposure * Factor Return
        # Element-wise multiplication for matching columns
        contributions = pd.DataFrame(index=common_dates)
        
        for col in exp.columns:
            if col in f_ret.columns:
                contributions[col] = exp[col] * f_ret[col]

        # 3. Aggregation
        total_factor_contrib = contributions.sum(axis=1)
        residual_alpha = p_ret - total_factor_contrib

        return {
            'factor_contribution': contributions.sum().to_dict(),
            'cumulative_attribution': contributions.cumsum().to_dict(),
            'total_factor_return': total_factor_contrib.sum(),
            'residual_alpha': residual_alpha.sum(),
            'alpha_volatility': residual_alpha.std() * np.sqrt(252)
        }

    def calculate_rolling_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Calculates Rolling Information Coefficient (Spearman Rank Correlation).
        IMPORTANT: forward_returns must be shifted (t+1) aligned to factor_values (t).
        """
        ic_series = []
        
        # Ensure inputs are sorted MultiIndex
        if not isinstance(factor_values.index, pd.MultiIndex):
             logger.warning("Factor values should have MultiIndex (date, ticker).")
             return pd.Series()

        dates = factor_values.index.get_level_values(0).unique()
        
        for date in dates:
            try:
                f_slice = factor_values.xs(date)
                r_slice = forward_returns.xs(date)
                
                # Align tickers
                common = f_slice.index.intersection(r_slice.index)
                if len(common) < 5: continue
                
                # Calculate IC for this date
                # Assuming single column or taking the first column if DataFrame
                f_data = f_slice.loc[common]
                if isinstance(f_data, pd.DataFrame):
                    f_data = f_data.iloc[:, 0]
                
                r_data = r_slice.loc[common]
                if isinstance(r_data, pd.DataFrame):
                    r_data = r_data.iloc[:, 0]
                
                ic, _ = stats.spearmanr(f_data, r_data)
                ic_series.append({'date': date, 'ic': ic})
            except Exception:
                continue
            
        if not ic_series:
            return pd.Series()
            
        ic_df = pd.DataFrame(ic_series).set_index('date').sort_index()
        return ic_df['ic'].rolling(window=window).mean()


class SimpleAttribution:
    """Quick trade-level attribution without complex factor models."""
    
    def analyze_pnl_drivers(self, trades_df: pd.DataFrame) -> Dict:
        """Breaks down PnL by Long/Short."""
        if trades_df.empty:
            return {'long_pnl_contribution': 0.0, 'short_pnl_contribution': 0.0, 'hit_ratio': 0.0}
            
        if 'pnl' not in trades_df.columns:
            return {"error": "Trades DataFrame must contain 'pnl' column"}
        
        # Filter for closed trades (non-zero PnL) to avoid skewing stats with entry orders
        closed_trades = trades_df[trades_df['pnl'] != 0]
        
        # Logic: PnL is realized on the CLOSING leg.
        # Closing a Long position involves SELLING.
        # Closing a Short position involves BUYING.
        # Assuming Long-Only strategy for simplicity in this report.
        
        # Fix: Sum PnL from all trades. In Long-Only, 'buy' has 0 PnL, 'sell' has realized PnL.
        # If Shorting exists, 'buy' would have PnL.
        long_pnl = trades_df[trades_df['pnl'] > 0]['pnl'].sum() + trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        short_pnl = 0.0 # Placeholder for Long-Only
        
        winners = closed_trades[closed_trades['pnl'] > 0]
        losers = closed_trades[closed_trades['pnl'] < 0]
        
        hit_ratio = len(winners) / len(closed_trades) if len(closed_trades) > 0 else 0
        
        avg_win = winners['pnl'].mean() if not winners.empty else 0
        avg_loss = losers['pnl'].mean() if not losers.empty else 0
        
        # Safe Win/Loss Ratio
        if avg_loss != 0 and avg_win != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
        else:
            win_loss_ratio = 0
        
        return {
            'long_pnl_contribution': long_pnl,
            'short_pnl_contribution': short_pnl,
            'hit_ratio': hit_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio
        }
