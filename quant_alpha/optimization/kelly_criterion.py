"""
Kelly Criterion Portfolio Optimization
======================================
Derives optimal position sizes to maximize the geometric growth rate of capital
(Log-Utility Maximization).

Purpose
-------
The `KellyCriterion` module implements a covariance-aware formulation of the
Kelly Criterion. Unlike the simple "Edge/Odds" formula, this implementation
solves a Quadratic Program (QP) to handle correlation structure between assets.
It effectively maps to a Mean-Variance optimization problem where the risk
aversion coefficient $\lambda$ is the reciprocal of the Kelly fraction.

Usage
-----
Intended for use within the `PortfolioAllocator` or standalone analysis.

.. code-block:: python

    kc = KellyCriterion(fraction=0.5, max_leverage=1.5)
    weights = kc.calculate_portfolio(
        expected_returns={'AAPL': 0.12, 'GOOG': 0.15},
        covariance_matrix=cov_df,
        risk_free_rate=0.04
    )

Importance
----------
-   **Growth Optimality**: Asymptotically maximizes wealth over infinite trials ($g^*$).
-   **Risk Control**: "Fractional Kelly" ($f < 1.0$) is implemented to mitigate
    the extreme volatility and drawdown risks associated with "Full Kelly".
-   **Edge Case Resilience**: Includes defensive fallbacks for non-positive
    excess return regimes (where the theoretical Kelly weight is zero).

Tools & Frameworks
------------------
-   **CVXPY**: Solves the constrained Quadratic Program (QP) for portfolio weights.
-   **NumPy/Pandas**: Linear algebra for the heuristic fallback and data alignment.
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

    Theoretical Foundation:
    -----------------------
    Maximizes the expected geometric growth rate $g$. By Taylor expansion of the
    expected log-utility function $E[\ln(1 + r)]$:
    
    .. math::
        g \approx r - \frac{1}{2}\sigma^2

    In a multi-asset context with correlations, this is equivalent to Mean-Variance
    optimization with specific risk aversion:
    
    .. math::
        \text{maximize} \quad w^T (\mu - r_f) - \frac{1}{2} \lambda w^T \Sigma w

    Where the risk aversion coefficient $\lambda = \frac{1}{\text{fraction}}$.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_leverage: float = 1.0,
        use_solver: bool = True
    ):
        """
        Args:
            fraction (float): Kelly multiplier $f$ (Recommended: $0.25 \le f \le 0.5$).
            max_leverage (float): Gross leverage limit constraint ($\sum w_i \le L$).
            use_solver (bool): If True, uses CVXPY (Quadratic Programming). 
                               If False, uses unconstrained closed-form solution.
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
        """Aligns input data to the intersection of tickers in returns and covariance."""

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
        """Maximum Entropy Fallback: Used when the theoretical optimal allocation is zero/cash."""
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
        Calculates the covariance-aware optimal Kelly allocation vector.

        Args:
            expected_returns (Dict): Expected annualized returns vector $\mu$.
            covariance_matrix (DataFrame): Annualized covariance matrix $\Sigma$.
            risk_free_rate (float): Annualized risk-free rate $r_f$.

        Returns:
            Dict[str, float]: Position weights. Always fully invested (sums to 1.0).
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        excess_returns = mu - risk_free_rate

        try:
            if self.use_solver:
                # --- Quadratic Programming Formulation ---
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
                # --- Heuristic (Unconstrained Closed-Form) ---
                # w* = f * Sigma^-1 * (mu - rf)
                inv_sigma = np.linalg.pinv(Sigma)
                full_kelly = inv_sigma @ excess_returns
                optimal_weights = full_kelly * self.fraction
                optimal_weights = np.maximum(optimal_weights, 0.0)
                
                # Apply leverage cap scaling
                total_lev = np.sum(optimal_weights)
                if total_lev > self.max_leverage:
                    optimal_weights = optimal_weights * (self.max_leverage / total_lev)

            # Trivial Solution Guard:
            # When all excess returns are negative ($\mu < r_f$), the optimal QP solution 
            # is the zero vector $w=0$ (holding 100% cash).
            # For a fully-invested long-only mandate, this is invalid. We fallback 
            # to Equal Weight to maintain market exposure if cash is not an option.
            total_invested = float(np.sum(np.maximum(optimal_weights, 0.0)))
            if total_invested < 1e-6:
                return self._equal_weight_fallback(tickers)

            # Build weight dict.
            # Note: Kelly weights are absolute. If sum(w) < 1, remainder is cash.
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