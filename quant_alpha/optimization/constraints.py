"""
Portfolio Constraints - Production Grade
Helper module to generate CVXPY constraints for portfolio optimization.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PortfolioConstraints:
    """
    Factory for creating CVXPY constraints for portfolio optimization.
    
    Usage:
        w = cp.Variable(n)
        cons = PortfolioConstraints.long_only(w)
        cons += PortfolioConstraints.position_limit(w, max_weight=0.10)
    """
    
    @staticmethod
    def long_only(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: Weights must be non-negative (Long Only).
        """
        return [w >= 0]
    
    @staticmethod
    def fully_invested(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: Sum of weights must equal 1.0.
        """
        return [cp.sum(w) == 1.0]
    
    @staticmethod
    def dollar_neutral(w: cp.Variable) -> List[cp.Constraint]:
        """
        Constraint: Sum of weights must equal 0.0 (Market Neutral).
        """
        return [cp.sum(w) == 0.0]
    
    @staticmethod
    def leverage_limit(w: cp.Variable, limit: float = 1.0) -> List[cp.Constraint]:
        """
        Constraint: Gross leverage (sum of absolute weights) <= limit.
        """
        return [cp.norm(w, 1) <= limit]
    
    @staticmethod
    def position_limit(
        w: cp.Variable, 
        max_weight: float = 1.0, 
        min_weight: float = 0.0
    ) -> List[cp.Constraint]:
        """
        Constraint: Individual position limits.
        min_weight <= w_i <= max_weight
        """
        constraints = []
        if max_weight < 1.0:
            constraints.append(w <= max_weight)
        
        # FIX: Always apply min_weight (allows negative values for Long/Short)
        constraints.append(w >= min_weight)
        return constraints
    
    @staticmethod
    def sector_exposure_limit(
        w: cp.Variable,
        tickers: List[str],
        sector_map: Dict[str, str],
        max_sector_weight: float = 0.30
    ) -> List[cp.Constraint]:
        """
        Constraint: Limit Net Sector Exposure magnitude.
        
        Args:
            w: CVXPY variable for weights
            tickers: List of tickers corresponding to w indices
            sector_map: Dictionary mapping ticker -> sector
            max_sector_weight: Maximum allowed weight per sector
        """
        constraints = []
        
        # Group indices by sector
        sector_indices = {}
        for i, ticker in enumerate(tickers):
            sector = sector_map.get(ticker, 'Unknown')
            if sector not in sector_indices:
                sector_indices[sector] = []
            sector_indices[sector].append(i)
            
        # Add constraint for each sector
        for sector, indices in sector_indices.items():
            # FIX: Use abs() to constrain magnitude for Long/Short safety
            # |Sum(w_sector)| <= Limit
            constraints.append(cp.abs(cp.sum(w[indices])) <= max_sector_weight)
            
        return constraints

    @staticmethod
    def turnover_limit(
        w: cp.Variable,
        current_weights: np.ndarray,
        turnover_cap: float
    ) -> List[cp.Constraint]:
        """
        Constraint: Limit turnover from current portfolio.
        sum(|w - w_current|) <= turnover_cap
        """
        return [cp.norm(w - current_weights, 1) <= turnover_cap]
