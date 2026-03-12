"""
Factor Attribution Module
=========================
Decomposes portfolio returns into Systematic Factors and Idiosyncratic Alpha.

Purpose
-------
This module implements performance attribution analysis, allowing researchers to separate
returns driven by systematic risk premia (Beta) from returns driven by skill/selection (Alpha).
It supports both factor-based attribution (Brinson-Fachler style or Regression) and 
simple trade-level PnL decomposition for execution analysis.

Usage
-----
.. code-block:: python

    from quant_alpha.backtest.attribution import FactorAttribution
    
    attr = FactorAttribution()
    results = attr.analyze(
        portfolio_returns=returns_series,
        factor_exposures=exposures_df,  # Index: Date, Columns: Factors
        factor_returns=factor_ret_df    # Index: Date, Columns: Factors
    )

Importance
----------
- **Alpha Verification**: Confirms performance stems from the intended signal rather than incidental factor bets.
- **Risk Decomposition**: Quantifies the portion of variance explained by systematic factors ($R^2$).
- **Efficiency**: Optimized for vectorization with $O(T \times F)$ complexity.

Tools & Frameworks
------------------
- **Pandas**: High-performance time-series alignment and rolling window operations.
- **NumPy**: Vectorized linear algebra for contribution calculation.
- **SciPy**: Statistical significance testing (Student's t-test, Spearman rank correlation).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class FactorAttribution:
    """
    Institutional-grade performance attribution engine.
    Decomposition Model: $R_{p,t} = \\sum_{k=1}^{K} (w_{p,t-1}^T \\beta_{k,t-1}) \\times R_{f,k,t} + \\epsilon_t$
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
        Orchestrates the attribution pipeline: Total Return Analysis $\rightarrow$ Factor Decomposition.
        
        Args:
            portfolio_returns: pd.Series (Time-series of portfolio returns).
            factor_exposures: pd.DataFrame (Date $\times$ Factor). Portfolio weighted average exposure.
            factor_returns: pd.DataFrame (Date $\times$ Factor). Returns of the factors themselves.
            
        Returns:
            Dict containing performance metrics, residuals, and factor contributions.
        """
        if portfolio_returns.empty:
            return {"error": "Portfolio returns series is empty"}

        results = {
            'total_return': portfolio_returns.sum(),
            'annualized_return': portfolio_returns.mean() * 252,
            'annualized_vol': portfolio_returns.std() * np.sqrt(252)
        }
        
        # Conditional Execution: Perform decomposition only if factor data is provided
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
        Computes attribution of returns to specific factors.
        
        CRITICAL: Enforces 1-day lag on exposures to prevent look-ahead bias.
        Equation: $Contrib_{f,t} = Exposure_{f, t-1} \\times Return_{f, t}$
        """
        # 1. Data Alignment & Lagging
        # Shift exposures by 1 day: T-1 holdings determine T performance.
        lagged_exposures = exposures.shift(1)
        
        # Intersection of indices (Point-in-Time alignment)
        common_dates = portfolio_returns.index.intersection(lagged_exposures.index).intersection(factor_returns.index)
        
        if len(common_dates) == 0:
            logger.error("No overlapping data found for attribution analysis.")
            return {'factor_contribution': {}, 'residual_alpha': 0}

        p_ret = portfolio_returns.loc[common_dates]
        exp = lagged_exposures.loc[common_dates]
        f_ret = factor_returns.loc[common_dates]

        # 2. Factor Contribution Calculation
        # Element-wise multiplication (Hadamard product) for matching columns
        contributions = pd.DataFrame(index=common_dates)
        
        for col in exp.columns:
            if col in f_ret.columns:
                contributions[col] = exp[col] * f_ret[col]

        # 3. Residual & Aggregation
        total_factor_contrib = contributions.sum(axis=1)
        residual_alpha = p_ret - total_factor_contrib

        # Risk-Adjusted Alpha Metrics (Information Ratio)
        ann_alpha = residual_alpha.mean() * 252
        alpha_vol = residual_alpha.std() * np.sqrt(252)
        information_ratio = ann_alpha / alpha_vol if alpha_vol != 0 else 0.0
        
        # Factor Efficiency (Statistical Significance)
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
        Evaluates the statistical significance of each factor's contribution.
        Metrics: Hit Rate ($P(Ret > 0)$), t-statistic on mean contribution.
        """
        efficiency = {}
        for col in contributions.columns:
            if contributions[col].empty: continue
            
            pos_days = (contributions[col] > 0).sum()
            total_days = len(contributions)
            
            # One-sample t-test ($H_0: \mu = 0$)
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
        Computes the Rolling Information Coefficient (IC) using Spearman Rank Correlation.
        
        Constraint: `forward_returns` must be pre-aligned (T+1 returns aligned to T factor values).
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
                
                # Extract vector for correlation
                # Handles cases where input is DataFrame (take first col) or Series
                f_data = f_slice.loc[common]
                if isinstance(f_data, pd.DataFrame):
                    f_data = f_data.iloc[:, 0]
                
                r_data = r_slice.loc[common]
                if isinstance(r_data, pd.DataFrame):
                    r_data = r_data.iloc[:, 0]
                
                # Numerical Stability: Handle zero variance to prevent divide-by-zero/NaNs
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
    """Execution-level attribution analyzing PnL drivers directly from trade logs."""
    
    def analyze_pnl_drivers(self, trades_df: pd.DataFrame) -> Dict:
        """Aggregates trade attributes to derive Gross/Net PnL, Win/Loss ratios, and side-specific contributions."""
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
        
        # Filter for closed trades (Realized PnL)
        # Supports explicit 'status' column or implicit PnL non-zero check
        if 'status' in trades_df.columns:
            closed_trades = trades_df[trades_df['status'].astype(str).str.lower() == 'closed']
        else:
            closed_trades = trades_df[trades_df['pnl'] != 0]
        
        # PnL Aggregation
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
        
        # Safe Win/Loss Ratio (Guard against division by zero)
        if avg_loss != 0 and avg_win != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
        else:
            win_loss_ratio = 0.0
            
        # Directional Decomposition (Long vs Short)
        # Handles legacy schemas where 'side' might be missing (defaults to Long-Only)
        long_pnl = total_pnl
        short_pnl = 0.0
        if 'side' in closed_trades.columns:
             long_mask = closed_trades['side'].astype(str).str.lower() == 'long'
             long_pnl = closed_trades[long_mask]['pnl'].sum()
             short_pnl = closed_trades[~long_mask]['pnl'].sum()
        
        # Return standardized metrics dictionary
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
