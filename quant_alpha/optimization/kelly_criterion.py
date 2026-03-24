"""
Kelly Criterion Portfolio Optimization
======================================

Derives optimal position sizes to maximize the geometric growth rate of capital
(Log-Utility Maximization).

Purpose
-------
This module implements a covariance-aware formulation of the Kelly Criterion. 
Unlike the simple "Edge/Odds" formula, this implementation solves a Quadratic 
Program (QP) to handle the correlation structure between assets. It effectively 
maps to a Mean-Variance optimization problem where the risk aversion coefficient 
is the reciprocal of the Kelly fraction.

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
-   **Growth Optimality**: Asymptotically maximizes wealth over infinite trials.
-   **Risk Control**: "Fractional Kelly" (f < 1.0) mitigates the extreme volatility
    and drawdown risks associated with "Full Kelly".
-   **Edge Case Resilience**: Includes defensive fallbacks for non-positive
    excess return regimes (where the theoretical Kelly weight is zero).

Mathematical Dependencies
-------------------------
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
    r"""
    Multi-Asset Kelly Criterion Position Sizing.

    Theoretical Foundation
    ----------------------
    Maximizes the expected geometric growth rate g. By Taylor expansion of the
    expected log-utility function E[ln(1 + r)]:

    .. math::
        g \approx r - \frac{1}{2}\sigma^2

    In a multi-asset context with correlations, this is equivalent to Mean-Variance
    optimization with specific risk aversion:

    .. math::
        \text{maximize} \quad w^T (\mu - r_f) - \frac{1}{2} \lambda w^T \Sigma w

    Where the risk aversion coefficient lambda = 1 / fraction.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_leverage: float = 1.0,
        use_solver: bool = True,
    ):
        """
        Initializes the Kelly Criterion optimization module.

        Args:
            fraction (float): The fractional Kelly multiplier (f). Recommended range is 
                between 0.1 and 0.5 to mitigate drawdown risks associated with "Full Kelly".
            max_leverage (float): The gross leverage constraint (sum of weights <= L) 
                enforced internally within the QP bounds.
            use_solver (bool): If True, utilizes CVXPY to enforce constraints via QP. 
                If False, applies an unconstrained closed-form optimization heuristic.
                
        Raises:
            ValueError: If the provided Kelly fraction is less than or equal to zero.
        """
        if fraction <= 0:
            raise ValueError(
                "Kelly fraction must be > 0. Recommended range: 0.1 to 1.0"
            )
        self.fraction     = fraction
        self.max_leverage = max_leverage
        self.use_solver   = use_solver

        logger.info(f"Kelly Criterion initialized: {fraction:.1%} of Full Kelly")

    def _prepare_data(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        """
        Aligns the expected returns vector and covariance matrix intersection.
        
        Args:
            expected_returns (Dict[str, float]): Expected annualized returns mapped by ticker.
            covariance_matrix (pd.DataFrame): Annualized asset covariance matrix.
            
        Returns:
            Tuple[List[str], np.ndarray, np.ndarray, int]: The intersection of tickers, 
                the returns array (mu), the covariance array (Sigma), and the asset count (n).
                
        Raises:
            ValueError: If no common identifiers exist between expected returns and the covariance frame.
        """
        available = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers   = [t for t in expected_returns if t in available]
        n         = len(tickers)

        if n == 0:
            raise ValueError(
                "No valid tickers found with both expected returns and covariance data."
            )

        mu    = np.array([expected_returns[t] for t in tickers])
        Sigma = covariance_matrix.loc[tickers, tickers].values

        return tickers, mu, Sigma, n

    def _equal_weight_fallback(self, tickers: List[str]) -> Dict[str, float]:
        """
        Generates a maximum entropy (equal weight) fallback allocation.
        
        Employed when the theoretical optimal allocation collapses to zero 
        (e.g., during regimes where all forecasted excess returns are <= 0).
        
        Args:
            tickers (List[str]): List of asset tickers to allocate.
            
        Returns:
            Dict[str, float]: Equal-weighted allocation dictionary summing exactly to 1.0.
        """
        n = len(tickers)
        logger.warning(
            "⚠️ All Kelly fractions are zero or negative (all excess returns ≤ 0). "
            "Falling back to Equal Weight allocation."
        )
        return {t: 1.0 / n for t in tickers}

    def _normalise_weights(
        self,
        raw_weights: np.ndarray,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Normalizes raw optimization output vectors to guarantee full investment.
        
        Mitigates floating-point inaccuracies introduced by the solver's 
        non-negativity constraints, ensuring that downstream portfolio management 
        modules map to a strictly fully invested assumption.
        
        Args:
            raw_weights (np.ndarray): The unnormalized allocation vector from the optimizer.
            tickers (List[str]): The corresponding list of asset tickers.
            
        Returns:
            Dict[str, float]: A mapped dictionary of tickers to normalized, positive weights.
        """
        clipped = np.maximum(raw_weights, 0.0)
        total   = float(clipped.sum())

        if total < 1e-6:
            return self._equal_weight_fallback(tickers)

        normalised = clipped / total
        return {
            tickers[i]: round(float(w), 8)
            for i, w in enumerate(normalised)
            if w > 1e-6
        }

    def calculate_portfolio(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, float]:
        """
        Calculates the covariance-aware optimal Kelly allocation vector.

        Args:
            expected_returns (Dict[str, float]): Expected annualized returns mapped by asset.
            covariance_matrix (pd.DataFrame): Annualized asset covariance matrix.
            risk_free_rate (float): Annualized baseline risk-free rate mapping (rf).

        Returns:
            Dict[str, float]: Optimal target position weights scaled out to sum to 1.0.
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(
                expected_returns, covariance_matrix
            )
        except ValueError as e:
            logger.error(str(e))
            return {}

        excess_returns = mu - risk_free_rate

        try:
            if self.use_solver:
                # Maps the Log-Utility maximization to a constrained Mean-Variance QP objective
                # by assigning the risk aversion coefficient to the inverse of the Kelly fraction.
                # Subject constraints: Weights must be >= 0 and capped by maximum strategy leverage.
                w              = cp.Variable(n)
                risk_aversion  = 1.0 / self.fraction

                objective   = cp.Maximize(
                    w @ excess_returns
                    - 0.5 * risk_aversion * cp.quad_form(w, Sigma)
                )
                constraints = [w >= 0, cp.sum(w) <= self.max_leverage]

                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.OSQP)

                if prob.status not in ("optimal", "optimal_inaccurate"):
                    raise ValueError(f"CVXPY solver failed: {prob.status}")

                raw_weights = w.value

            else:
                # Computes the unconstrained closed-form Kelly vector directly via the 
                # Moore-Penrose pseudo-inverse, ensuring matrix stability under collinearity.
                inv_sigma   = np.linalg.pinv(Sigma)
                full_kelly  = inv_sigma @ excess_returns
                raw_weights = full_kelly * self.fraction
                
                total = float(np.sum(np.maximum(raw_weights, 0.0)))
                if total > self.max_leverage:
                    raw_weights = raw_weights * (self.max_leverage / total)

            # Translates relative position sizing dynamically computed by the Kelly bounds 
            # to standard 1.0 normalization to satisfy capital execution assumptions.
            weight_dict = self._normalise_weights(raw_weights, tickers)

            logger.info(
                f"Kelly optimized: {len(weight_dict)} positions, "
                f"sum(w)={sum(weight_dict.values()):.4f}"
            )
            return weight_dict

        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}. Falling back to equal weight.")
            return {t: 1.0 / n for t in tickers}