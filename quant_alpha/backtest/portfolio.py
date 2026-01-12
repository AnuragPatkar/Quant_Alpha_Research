"""
Portfolio Utilities
===================
Simple portfolio analysis utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class PortfolioAnalyzer:
    """Simple portfolio analysis tools."""
    
    @staticmethod
    def calculate_weights(positions: List[str], equal_weight: bool = True) -> Dict[str, float]:
        """Calculate portfolio weights."""
        if equal_weight:
            weight = 1.0 / len(positions)
            return {ticker: weight for ticker in positions}
        else:
            # Could add other weighting schemes here
            return {}
    
    @staticmethod
    def portfolio_return(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns from individual stock returns."""
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        
        for date in returns.index:
            day_return = 0
            for ticker, weight in weights.items():
                if ticker in returns.columns:
                    day_return += weight * returns.loc[date, ticker]
            portfolio_returns[date] = day_return
        
        return portfolio_returns
    
    @staticmethod
    def rebalancing_turnover(old_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
        """Calculate portfolio turnover from rebalancing."""
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        
        turnover = 0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            turnover += abs(new_w - old_w)
        
        return turnover / 2  # Divide by 2 as each trade affects two sides