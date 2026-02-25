"""
Kelly Criterion Position Sizing - Production Grade
Optimal bet size for maximum geometric growth, incorporating covariance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import cvxpy as cp
import logging

logger = logging.getLogger(__name__)

class KellyCriterion:
    """
    Multi-Asset Kelly Criterion Position Sizing.
    
    Mathematically equivalent to maximizing the geometric growth rate:
    Maximize: r - 0.5 * σ^2
    
    This can be solved as a Quadratic Program (Mean-Variance) with:
    Risk Aversion (λ) = 1 / Fraction
    
    Note: Fractional Kelly (e.g., 0.5) is strongly recommended.
    """
    
    def __init__(
        self, 
        fraction: float = 0.5,
        max_leverage: float = 1.0,
        use_solver: bool = True
    ):
        """
        Args:
            fraction: Fraction of Kelly to use (0.25-0.5 recommended)
            max_leverage: Maximum total portfolio leverage
            use_solver: Use CVXPY solver (True) or numpy heuristic (False)
        """
        if fraction <= 0:
            raise ValueError("Kelly fraction must be > 0. Recommended range: 0.1 to 1.0")
        self.fraction = fraction
        self.max_leverage = max_leverage
        self.use_solver = use_solver
        
        logger.info(f"Kelly Criterion initialized: {fraction:.1%} of Full Kelly")

    def _prepare_data(
        self, 
        expected_returns: Dict[str, float], 
        covariance_matrix: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        
        # FIX: Ensure tickers exist in both Index and Columns of Covariance Matrix
        available_tickers = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers = list(expected_returns.keys())
        valid_tickers = [t for t in tickers if t in available_tickers]
        n = len(valid_tickers)
        
        if n == 0:
            raise ValueError("No valid tickers found with both returns and covariance data.")
            
        mu = np.array([expected_returns[t] for t in valid_tickers])
        Sigma = covariance_matrix.loc[valid_tickers, valid_tickers].values
        
        return valid_tickers, mu, Sigma, n

    def calculate_portfolio(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04
    ) -> Dict[str, float]:
        """
        Calculates the covariance-aware optimal Kelly allocation.
        
        Args:
            expected_returns: Dict of expected annualized returns
            covariance_matrix: Annualized covariance matrix (preferably shrunk)
            risk_free_rate: Annualized risk-free rate
            
        Returns:
            Dict of position weights
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        # 1. Calculate Excess Returns
        excess_returns = mu - risk_free_rate

        try:
            if self.use_solver:
                # --- SOLVER APPROACH (Production Grade) ---
                # Kelly Objective: Maximize (w.T * excess_ret) - (1 / (2*f)) * (w.T * Sigma * w)
                # This correctly handles constraints while finding optimal sizing.
                
                w = cp.Variable(n)
                
                # Risk aversion parameter for Fractional Kelly
                # Full Kelly corresponds to lambda = 1. Fractional f corresponds to lambda = 1/f
                risk_aversion = 1.0 / self.fraction
                
                objective = cp.Maximize(
                    w @ excess_returns - 0.5 * risk_aversion * cp.quad_form(w, Sigma)
                )
                
                constraints = [
                    w >= 0,  # Long only
                    cp.sum(w) <= self.max_leverage  # Leverage limit
                ]
                
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.OSQP)
                
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    raise ValueError(f"Solver failed: {prob.status}")
                
                optimal_weights = w.value
                
            else:
                # --- HEURISTIC APPROACH (Legacy/Fallback) ---
                inv_sigma = np.linalg.pinv(Sigma)
                full_kelly = inv_sigma @ excess_returns
                optimal_weights = full_kelly * self.fraction
                
                # Heuristic clipping (suboptimal)
                optimal_weights = np.maximum(optimal_weights, 0.0)
                
                # Scale down if leverage exceeded
                total_lev = np.sum(optimal_weights)
                if total_lev > self.max_leverage:
                    optimal_weights = optimal_weights * (self.max_leverage / total_lev)

            # 2. Construct Result Dictionary
            weight_dict = {
                tickers[i]: round(float(w), 8)
                for i, w in enumerate(optimal_weights)
                if w > 1e-6
            }
            
            # NOTE: No renormalization here. Kelly dictates the absolute size.
            # If sum(weights) is 0.6, it means 40% cash.
            
            logger.info(f"Multi-Asset Kelly optimized for {len(weight_dict)} positions.")
            return weight_dict

        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}")
            return {t: 1.0/n for t in tickers}