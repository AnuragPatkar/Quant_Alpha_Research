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
        Enforces 1-day lag on exposures to ensure Point-in-Time alignment.
        """
        # 1. Align Data
        # Shift exposures by 1 day (T-1 exposure explains T return)
        lagged_exposures = exposures.shift(1)
        
        # Ensure all inputs share the same index (dates)
        common_dates = portfolio_returns.index.intersection(lagged_exposures.index).intersection(factor_returns.index)
        
        if len(common_dates) == 0:
            logger.error("No overlapping data found for attribution analysis.")
            return {'factor_contribution': {}, 'residual_alpha': 0}

        p_ret = portfolio_returns.loc[common_dates]
        exp = lagged_exposures.loc[common_dates]
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

        # NEW: Risk-Adjusted Alpha Metrics
        ann_alpha = residual_alpha.mean() * 252
        alpha_vol = residual_alpha.std() * np.sqrt(252)
        information_ratio = ann_alpha / alpha_vol if alpha_vol != 0 else 0.0
        
        # NEW: Factor Efficiency Analysis
        efficiency = self.calculate_factor_efficiency(contributions)

        return {
            'factor_contribution': contributions.sum().to_dict(),
            'cumulative_attribution': contributions.cumsum().to_dict(),
            'total_factor_return': total_factor_contrib.sum(),
            'residual_alpha': residual_alpha.sum(),
            'alpha_volatility': alpha_vol,
            'information_ratio': information_ratio,
            'factor_efficiency': efficiency
        }

    def calculate_factor_efficiency(self, contributions: pd.DataFrame) -> Dict:
        """
        Calculates the 'Hit Rate' of each factor. 
        How many days did the factor actually contribute positively?
        """
        efficiency = {}
        for col in contributions.columns:
            if contributions[col].empty: continue
            
            pos_days = (contributions[col] > 0).sum()
            total_days = len(contributions)
            
            # T-test for mean different from 0
            # Safety: If variance is 0 (constant contribution), t-stat is 0
            if np.std(contributions[col]) == 0:
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_1samp(contributions[col], 0)
            
            efficiency[col] = {
                'hit_rate': pos_days / total_days if total_days > 0 else 0.0,
                'avg_contribution': contributions[col].mean(),
                't_stat': t_stat if not np.isnan(t_stat) else 0.0,
                'p_value': p_val if not np.isnan(p_val) else 1.0
            }
        return efficiency

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
                
                # Safety: Handle constant arrays to avoid RuntimeWarnings
                if np.std(f_data) == 0 or np.std(r_data) == 0:
                    ic = 0.0
                else:
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
            return {
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'hit_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_loss_ratio': 0.0,
                'long_pnl_contribution': 0.0,
                'short_pnl_contribution': 0.0
            }
            
        if 'pnl' not in trades_df.columns:
            return {"error": "Trades DataFrame must contain 'pnl' column"}
        
        # Filter for closed trades
        # Case-insensitive status check
        if 'status' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'].astype(str).str.lower() == 'closed']
        else:
            closed_trades = trades_df[trades_df['pnl'] != 0]
        
        # Gross Profit vs Net PnL Logic
        total_pnl = closed_trades['pnl'].sum()
        
        winners = closed_trades[closed_trades['pnl'] > 0]
        losers = closed_trades[closed_trades['pnl'] < 0]
        
        gross_profit = winners['pnl'].sum()
        gross_loss = losers['pnl'].sum()
        
        winning_trades = len(winners)
        losing_trades = len(losers)
        total_trades = len(closed_trades)
        
        hit_ratio = winning_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = winners['pnl'].mean() if not winners.empty else 0.0
        avg_loss = losers['pnl'].mean() if not losers.empty else 0.0
        
        # Safe Win/Loss Ratio
        if avg_loss != 0 and avg_win != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
        else:
            win_loss_ratio = 0.0
            
        # Legacy Support for Long/Short breakdown
        # Assuming Long-Only if side is missing, or parsing side if present
        long_pnl = total_pnl
        short_pnl = 0.0
        if 'side' in closed_trades.columns:
             long_mask = closed_trades['side'].astype(str).str.lower() == 'long'
             long_pnl = closed_trades[long_mask]['pnl'].sum()
             short_pnl = closed_trades[~long_mask]['pnl'].sum()
        
        # Standardized Keys
        return {
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'hit_ratio': hit_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'long_pnl_contribution': long_pnl,
            'short_pnl_contribution': short_pnl
        }
