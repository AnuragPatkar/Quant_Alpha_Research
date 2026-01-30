"""
Portfolio Utilities
===================
Portfolio analysis and utility functions.

Classes:
    - PortfolioAnalyzer: Portfolio analysis tools
    
Author: Senior Quant Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class PortfolioAnalyzer:
    """
    Portfolio analysis tools.
    
    Provides:
        - Weight calculations
        - Portfolio return aggregation
        - Turnover analysis
        - Holdings analysis
    
    Example (static methods):
        >>> weights = PortfolioAnalyzer.calculate_weights(positions)
        >>> turnover = PortfolioAnalyzer.rebalancing_turnover(old, new)
    
    Example (instance):
        >>> analyzer = PortfolioAnalyzer(positions_df, returns_df)
        >>> stats = analyzer.calculate_statistics()
    """
    
    def __init__(
        self,
        positions: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None
    ):
        """
        Initialize portfolio analyzer.
        
        Args:
            positions: DataFrame with columns [date, ticker, weight/stocks]
            returns: DataFrame with [date, ticker, return] or wide format
        """
        self.positions = positions.copy() if positions is not None else None
        self.returns = returns.copy() if returns is not None else None
        
        # Ensure date column is datetime
        if self.positions is not None and 'date' in self.positions.columns:
            self.positions['date'] = pd.to_datetime(self.positions['date'])
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate portfolio statistics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        if self.positions is None:
            return {}
        
        stats = {}
        
        # Position statistics
        stats.update(self._position_stats())
        
        # Turnover (if multiple periods)
        if self._has_multiple_periods():
            stats.update(self._turnover_stats())
        
        return stats
    
    def _has_multiple_periods(self) -> bool:
        """Check if we have multiple rebalance periods."""
        if self.positions is None or 'date' not in self.positions.columns:
            return False
        return self.positions['date'].nunique() > 1
    
    def _position_stats(self) -> Dict[str, float]:
        """Calculate position statistics."""
        stats = {}
        
        if 'date' in self.positions.columns:
            if 'stocks' in self.positions.columns:
                # Positions stored as list in 'stocks' column
                positions_per_period = self.positions['stocks'].apply(len)
            else:
                positions_per_period = self.positions.groupby('date')['ticker'].count()
            
            stats['avg_positions'] = positions_per_period.mean()
            stats['min_positions'] = positions_per_period.min()
            stats['max_positions'] = positions_per_period.max()
        else:
            stats['avg_positions'] = len(self.positions)
        
        # Unique tickers
        if 'ticker' in self.positions.columns:
            stats['unique_tickers'] = self.positions['ticker'].nunique()
        elif 'stocks' in self.positions.columns:
            all_stocks = []
            for stocks in self.positions['stocks']:
                all_stocks.extend(stocks if isinstance(stocks, list) else [stocks])
            stats['unique_tickers'] = len(set(all_stocks))
        
        return stats
    
    def _turnover_stats(self) -> Dict[str, float]:
        """Calculate turnover statistics."""
        dates = sorted(self.positions['date'].unique())
        
        if len(dates) < 2:
            return {}
        
        turnovers = []
        
        for i in range(len(dates) - 1):
            curr_date = dates[i]
            next_date = dates[i + 1]
            
            # Get positions for each date
            curr_row = self.positions[self.positions['date'] == curr_date]
            next_row = self.positions[self.positions['date'] == next_date]
            
            # Extract stock lists
            if 'stocks' in self.positions.columns:
                curr_stocks = set(curr_row['stocks'].iloc[0]) if len(curr_row) > 0 else set()
                next_stocks = set(next_row['stocks'].iloc[0]) if len(next_row) > 0 else set()
            else:
                curr_stocks = set(curr_row['ticker']) if len(curr_row) > 0 else set()
                next_stocks = set(next_row['ticker']) if len(next_row) > 0 else set()
            
            # Calculate turnover
            total = len(curr_stocks.union(next_stocks))
            if total > 0:
                changed = len(curr_stocks.symmetric_difference(next_stocks))
                turnover = changed / total
                turnovers.append(turnover)
        
        if turnovers:
            return {
                'avg_turnover': np.mean(turnovers),
                'total_rebalances': len(turnovers)
            }
        
        return {}
    
    def get_holdings_summary(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Get holdings summary for a specific date.
        
        Args:
            date: Date to analyze (None = latest)
            
        Returns:
            DataFrame with holdings information
        """
        if self.positions is None:
            return pd.DataFrame()
        
        if 'date' not in self.positions.columns:
            return self.positions.copy()
        
        if date is None:
            date = self.positions['date'].max()
        
        holdings = self.positions[self.positions['date'] == date].copy()
        return holdings
    
    def print_summary(self):
        """Print formatted portfolio summary."""
        stats = self.calculate_statistics()
        
        print("\n" + "="*60)
        print("  PORTFOLIO ANALYSIS")
        print("="*60)
        
        print("\n  POSITION STATISTICS")
        print("  " + "-"*50)
        print(f"  Avg Positions:     {stats.get('avg_positions', 0):>10.1f}")
        print(f"  Unique Tickers:    {stats.get('unique_tickers', 0):>10.0f}")
        
        if 'min_positions' in stats:
            print(f"  Min Positions:     {stats['min_positions']:>10.0f}")
            print(f"  Max Positions:     {stats['max_positions']:>10.0f}")
        
        if 'avg_turnover' in stats:
            print("\n  TURNOVER")
            print("  " + "-"*50)
            print(f"  Avg Turnover:      {stats['avg_turnover']:>10.1%}")
            print(f"  Total Rebalances:  {stats['total_rebalances']:>10.0f}")
        
        print("\n" + "="*60 + "\n")
    
    # =========================================================================
    # STATIC METHODS (Original API preserved)
    # =========================================================================
    
    @staticmethod
    def calculate_weights(
        positions: List[str],
        equal_weight: bool = True,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights.
        
        Args:
            positions: List of ticker symbols
            equal_weight: Use equal weighting
            custom_weights: Custom weights dict (if not equal weight)
            
        Returns:
            Dictionary mapping ticker to weight
        """
        if not positions:
            return {}
        
        if equal_weight:
            weight = 1.0 / len(positions)
            return {ticker: weight for ticker in positions}
        elif custom_weights is not None:
            return custom_weights
        else:
            return {}
    
    @staticmethod
    def portfolio_return(
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Calculate portfolio returns from individual stock returns.
        
        Args:
            returns: DataFrame with tickers as columns, dates as index
            weights: Dictionary mapping ticker to weight
            
        Returns:
            Series of portfolio returns
        """
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        
        for date in returns.index:
            day_return = 0.0
            total_weight = 0.0
            
            for ticker, weight in weights.items():
                if ticker in returns.columns:
                    ret = returns.loc[date, ticker]
                    if not pd.isna(ret):
                        day_return += weight * ret
                        total_weight += weight
            
            # Normalize if not all weights present
            if total_weight > 0:
                portfolio_returns[date] = day_return / total_weight
            else:
                portfolio_returns[date] = np.nan
        
        return portfolio_returns
    
    @staticmethod
    def rebalancing_turnover(
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover from rebalancing.
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
            
        Returns:
            Turnover (0 to 1, where 1 = 100% turnover)
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        
        turnover = 0.0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            turnover += abs(new_w - old_w)
        
        return turnover / 2  # Divide by 2 as each trade affects two sides
    
    @staticmethod
    def calculate_concentration(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate portfolio concentration metrics.
        
        Args:
            weights: Dictionary mapping ticker to weight
            
        Returns:
            Dictionary with concentration metrics
        """
        if not weights:
            return {'hhi': 0, 'effective_n': 0, 'top_5_weight': 0}
        
        weight_values = list(weights.values())
        weight_array = np.array(weight_values)
        
        # Herfindahl-Hirschman Index
        hhi = (weight_array ** 2).sum()
        
        # Effective number of stocks
        effective_n = 1 / hhi if hhi > 0 else 0
        
        # Top 5 concentration
        sorted_weights = sorted(weight_values, reverse=True)
        top_5_weight = sum(sorted_weights[:5])
        
        return {
            'hhi': hhi,
            'effective_n': effective_n,
            'top_5_weight': top_5_weight
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_portfolio(
    positions: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Quick portfolio analysis (convenience function).
    
    Args:
        positions: Positions DataFrame
        returns: Optional returns DataFrame
        
    Returns:
        Dictionary with portfolio statistics
    """
    analyzer = PortfolioAnalyzer(positions, returns)
    return analyzer.calculate_statistics()


def print_portfolio_summary(
    positions: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None
):
    """
    Print portfolio summary (convenience function).
    
    Args:
        positions: Positions DataFrame
        returns: Optional returns DataFrame
    """
    analyzer = PortfolioAnalyzer(positions, returns)
    analyzer.print_summary()