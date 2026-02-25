"""
Mean-Variance Optimization (Markowitz) - Production Grade
Mathematically sound formulation using CVXPY with dynamic constraints.
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class MeanVarianceOptimizer:
    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the optimizer. Market variables like risk_free_rate 
        are passed directly to the optimization methods to ensure dynamic fetching.
        """
        self.risk_aversion = risk_aversion

    def _prepare_data(
        self, 
        expected_returns: Dict[str, float], 
        covariance_matrix: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        """
        Centralized data preparation to prevent DRY violations.
        Assumes covariance_matrix is pre-calculated (and shrunk if necessary) upstream.
        """
        # FIX: Intersect tickers to prevent KeyError if covariance matrix is missing assets
        available_tickers = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers = [t for t in expected_returns.keys() if t in available_tickers]
        n = len(tickers)
        
        if n == 0:
            raise ValueError("No valid tickers found in intersection of expected_returns and covariance_matrix.")
            
        mu = np.array([expected_returns[t] for t in tickers])
        Sigma = covariance_matrix.loc[tickers, tickers].values
        
        return tickers, mu, Sigma, n

    def _safe_normalize(self, optimized_w: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """
        Safely extracts and normalizes weights. 
        Raises an error if the solver returned severely infeasible weights.
        """
        n = len(tickers)
        weight_dict = {
            tickers[i]: round(float(optimized_w[i]), 8) 
            for i in range(n) if optimized_w[i] > 1e-6
        }
        
        total_sum = sum(weight_dict.values())
        
        # Only re-normalize if we are extremely close to 1.0
        if abs(total_sum - 1.0) < 1e-4:
            if total_sum > 0:
                return {k: v / total_sum for k, v in weight_dict.items()}
            return weight_dict
        else:
            raise ValueError(f"Solver returned infeasible weights summing to {total_sum:.4f}")

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Standard Mean-Variance Optimization.
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        w = cp.Variable(n)
        
        portfolio_return = mu @ w
        portfolio_variance = cp.quad_form(w, Sigma)
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_variance)
        
        constraints_list = [cp.sum(w) == 1.0]
        
        if not constraints or not constraints.get('allow_short', False):
            constraints_list.append(w >= 0)
            
        # Dynamic max weight to prevent infeasibility in small universes
        requested_max = constraints.get('max_weight', 0.05) if constraints else 0.05
        dynamic_max_w = max(requested_max, (1.0 / n) + 0.01)
        constraints_list.append(w <= dynamic_max_w)

        prob = cp.Problem(objective, constraints_list)
        
        try:
            # OSQP is the standard for Quadratic Programs (Linear Constraints)
            prob.solve(solver=cp.OSQP)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.SolverError(f"Solver status: {prob.status}")

            weight_dict = self._safe_normalize(w.value, tickers)
            logger.info(f"Optimization successful for {len(weight_dict)} assets.")
            return weight_dict

        except (cp.SolverError, ValueError) as e:
            logger.error(f"Optimization failed: {e}. Falling back to Equal Weight.")
            return {t: 1.0/n for t in tickers}

    def solve_max_sharpe(
        self, 
        expected_returns: Dict[str, float], 
        covariance_matrix: pd.DataFrame, 
        risk_free_rate: float,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Direct Sharpe Maximization using Charnes-Cooper transformation.
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        y = cp.Variable(n)
        kappa = cp.Variable()

        objective = cp.Maximize((mu - risk_free_rate) @ y)
        
        requested_max = constraints.get('max_weight', 0.05) if constraints else 0.05
        dynamic_max_w = max(requested_max, (1.0 / n) + 0.01)

        constraints_list = [
            cp.quad_form(y, Sigma) <= 1, # Quadratic Constraint
            cp.sum(y) == kappa,
            y >= 0,
            y <= dynamic_max_w * kappa,
            kappa >= 0
        ]

        prob = cp.Problem(objective, constraints_list)
        
        try:
            # FIX: OSQP cannot handle Quadratic Constraints. ECOS is required for QCQP/SOCP.
            prob.solve(solver=cp.ECOS)
            
            if prob.status not in ["optimal", "optimal_inaccurate"] or y.value is None or kappa.value <= 1e-6:
                raise cp.SolverError(f"Solver status: {prob.status} or invalid kappa.")
            
            weights = y.value / kappa.value
            weight_dict = self._safe_normalize(weights, tickers)
            return weight_dict
            
        except (cp.SolverError, ValueError) as e:
            logger.error(f"Sharpe optimization failed: {e}. Falling back to Equal Weight.")
            return {t: 1.0/n for t in tickers}
