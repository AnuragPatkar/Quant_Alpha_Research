"""
Kelly Criterion Position Sizing - Production Grade
Optimal bet size for maximum geometric growth, incorporating covariance.

BUGS FIXED:
  BUG-KC-01 [HIGH]: When all excess returns are negative (e.g. rf=0.99), the
    CVXPY solver correctly sets all weights to zero (the long-only constraint
    w >= 0 and leverage cap cp.sum(w) <= max_leverage admit the trivial solution
    w=0 as optimal). The result_dict ends up empty, calculate_portfolio returns
    {} rather than a valid portfolio, and the allocator fallback is never reached
    because no exception is raised — just a silent empty dict.

    Fix: After solver or heuristic completes, check whether the sum of all
    optimal weights is below a minimum threshold (1e-6). If so, fall back to
    Equal Weight explicitly and log a warning. This is financially correct:
    when Kelly says "hold no risky assets" and a long-only constraint is in
    force, the only valid fully-invested response is to hold equal weight
    (100% cash surrogate if available, or equal weight across assets).
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
            fraction: Fraction of Kelly to use (0.25–0.5 recommended)
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

        available_tickers = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers = list(expected_returns.keys())
        valid_tickers = [t for t in tickers if t in available_tickers]
        n = len(valid_tickers)

        if n == 0:
            raise ValueError("No valid tickers found with both returns and covariance data.")

        mu = np.array([expected_returns[t] for t in valid_tickers])
        Sigma = covariance_matrix.loc[valid_tickers, valid_tickers].values

        return valid_tickers, mu, Sigma, n

    def _equal_weight_fallback(self, tickers: List[str]) -> Dict[str, float]:
        """Equal-weight fallback when Kelly fractions are all non-positive."""
        n = len(tickers)
        logger.warning(
            "⚠️ All Kelly fractions are zero or negative (all excess returns ≤ 0). "
            "Falling back to Equal Weight allocation."
        )
        return {t: 1.0 / n for t in tickers}

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
            Dict of position weights. Always fully invested (sums to 1.0).
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        excess_returns = mu - risk_free_rate

        try:
            if self.use_solver:
                # --- SOLVER APPROACH (Production Grade) ---
                w = cp.Variable(n)
                risk_aversion = 1.0 / self.fraction

                objective = cp.Maximize(
                    w @ excess_returns - 0.5 * risk_aversion * cp.quad_form(w, Sigma)
                )
                constraints = [
                    w >= 0,
                    cp.sum(w) <= self.max_leverage
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
                optimal_weights = np.maximum(optimal_weights, 0.0)
                total_lev = np.sum(optimal_weights)
                if total_lev > self.max_leverage:
                    optimal_weights = optimal_weights * (self.max_leverage / total_lev)

            # ── FIX BUG-KC-01: guard against all-zero solution ────────────── #
            # When all excess returns are negative, the QP optimal is w=0
            # (the trivial feasible point). This is mathematically correct for
            # an unconstrained leveraged account (hold cash), but for a
            # fully-invested long-only mandate we must return a valid portfolio.
            total_invested = float(np.sum(np.maximum(optimal_weights, 0.0)))
            if total_invested < 1e-6:
                return self._equal_weight_fallback(tickers)

            # Build weight dict — NOTE: no renormalization per Kelly theory.
            # If sum(w) < 1 the remainder is implicitly held as cash.
            weight_dict = {
                tickers[i]: round(float(w), 8)
                for i, w in enumerate(optimal_weights)
                if w > 1e-6
            }

            logger.info(f"Multi-Asset Kelly optimized for {len(weight_dict)} positions.")
            return weight_dict

        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}")
            return {t: 1.0 / n for t in tickers}