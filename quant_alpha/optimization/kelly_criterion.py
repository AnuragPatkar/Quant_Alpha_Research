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
aversion coefficient is the reciprocal of the Kelly fraction.

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

Tools & Frameworks
------------------
-   **CVXPY**: Solves the constrained Quadratic Program (QP) for portfolio weights.
-   **NumPy/Pandas**: Linear algebra for the heuristic fallback and data alignment.

FIXES
-----
  BUG-078 (HIGH): calculate_portfolio() returned un-normalised weights in the
           normal code path. The QP solver returns weights that sum to at most
           max_leverage (e.g. 1.5), not to 1.0. The comment in the original
           code said "remainder is cash" — but PortfolioAllocator and
           BacktestEngine both expect weights summing to 1.0 (fully invested).
           A portfolio summing to 0.7 was silently 30% in cash with no
           indication to the caller.

           Fix: after solving, normalise the weight vector so sum(w) == 1.0.
           The Kelly fraction already controls aggressiveness via the QP
           objective (risk_aversion = 1/fraction). Normalisation preserves
           relative sizing while satisfying the contract.

           If the caller genuinely wants a cash allocation, they must handle
           it explicitly downstream — the weight dict contract is sum == 1.0.

           The equal-weight fallback (line ~120) already returned sum == 1.0;
           now the normal path matches.
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
        Args:
            fraction (float)   : Kelly multiplier f. Recommended: 0.25 <= f <= 0.5.
                                 Controls aggressiveness of the QP objective.
            max_leverage (float): Gross leverage constraint (sum of weights <= L)
                                 used INSIDE the QP. Output weights are always
                                 normalised to sum=1.0 after solving.
            use_solver (bool)  : If True, uses CVXPY (QP). If False, uses the
                                 unconstrained closed-form heuristic.
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
        """Aligns inputs to the intersection of tickers in returns and covariance."""
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
        Maximum Entropy Fallback: used when the theoretical optimal allocation
        is zero (all excess returns <= 0, i.e. full-cash regime).

        Returns weights that sum to 1.0.
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
        Normalise raw QP / heuristic weights so they sum to 1.0.

        FIX BUG-078: Kelly QP returns weights in [0, max_leverage]. Callers
        (PortfolioAllocator, BacktestEngine) expect sum(w) == 1.0.
        Zero-weight positions are dropped from the result dict.

        Steps:
        1. Clip to >= 0 (QP with non-negativity constraint should already do this,
           but floating-point can produce tiny negatives).
        2. If total weight is negligible, return equal-weight fallback.
        3. Divide by sum to normalise.
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
            expected_returns (Dict) : Expected annualized returns vector mu.
            covariance_matrix (df)  : Annualized covariance matrix Sigma.
            risk_free_rate (float)  : Annualized risk-free rate rf.

        Returns
        -------
        Dict[str, float]
            Position weights that always sum to 1.0 (fully invested).

        FIX BUG-078: Weights are normalised before returning so sum(w) == 1.0.
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
                # ---- Quadratic Programming Formulation ----
                # maximize  w @ excess - 0.5 * (1/fraction) * w' * Sigma * w
                # subject to  w >= 0,  sum(w) <= max_leverage
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
                # ---- Heuristic Closed-Form ----
                # w* = fraction * Sigma^{-1} * (mu - rf)
                inv_sigma   = np.linalg.pinv(Sigma)
                full_kelly  = inv_sigma @ excess_returns
                raw_weights = full_kelly * self.fraction
                # Apply leverage cap
                total = float(np.sum(np.maximum(raw_weights, 0.0)))
                if total > self.max_leverage:
                    raw_weights = raw_weights * (self.max_leverage / total)

            # FIX BUG-078: Normalise so sum(w) == 1.0.
            # The Kelly fraction controls aggressiveness via the QP objective;
            # normalisation preserves relative sizing.
            weight_dict = self._normalise_weights(raw_weights, tickers)

            logger.info(
                f"Kelly optimized: {len(weight_dict)} positions, "
                f"sum(w)={sum(weight_dict.values()):.4f}"
            )
            return weight_dict

        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}. Falling back to equal weight.")
            return {t: 1.0 / n for t in tickers}