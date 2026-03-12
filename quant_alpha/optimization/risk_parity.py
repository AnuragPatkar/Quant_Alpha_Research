"""
Risk Parity Portfolio Optimization Engine
=========================================
Convex optimization solver for Equal Risk Contribution (ERC) portfolio construction.

Purpose
-------
The `RiskParityOptimizer` constructs portfolios where each asset contributes
proportionally to the total portfolio risk, controlled by a risk budget vector $b$.
Unlike Mean-Variance optimization, Risk Parity does not require expected return
estimates ($\mu$), making it robust to estimation errors.

This implementation utilizes the **Spinu (2013)** convex formulation, which maps the
classic non-convex Risk Parity problem into a strictly convex minimization problem
using a log-barrier term. This guarantees a unique global optimum and avoids the
numerical instability (zero-gradient traps) associated with standard SQP solvers
on the marginal risk contribution metric.

Usage
-----
Intended for use via the `PortfolioAllocator` facade or standalone analysis.

.. code-block:: python

    # Standard Equal Risk Contribution (ERC)
    optimizer = RiskParityOptimizer()
    weights = optimizer.optimize(
        covariance_matrix=cov_df,
        tickers=['AAPL', 'MSFT', 'GOOG']
    )

    # Custom Risk Budgets (e.g., higher risk tolerance for Tech)
    optimizer = RiskParityOptimizer(target_risk={'AAPL': 0.4, 'MSFT': 0.4})

Importance
----------
-   **Convexity Guarantee**: The Spinu formulation transforms the problem into:
    .. math::
        \\min_{y} \\quad \\frac{1}{2} y^T \\Sigma y - \\sum_{i=1}^N b_i \\ln(y_i)
    This strictly convex objective ensures reliable convergence even with block-diagonal
    covariance matrices, where gradient-based methods (e.g., SLSQP) often encounter
    zero-gradient traps at the boundary ($w_i=0$).
-   **Numerical Stability**: Analytic gradients are supplied to the L-BFGS-B solver,
    reducing iterations and improving precision to $10^{-10}$.

Tools & Frameworks
------------------
-   **SciPy (optimize)**: L-BFGS-B solver for bound-constrained minimization.
-   **NumPy/Pandas**: Linear algebra operations and data alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

_WEIGHT_FLOOR = 1e-8  # Epsilon threshold to prune negligible weights post-normalization


class RiskParityOptimizer:
    """
    Risk Parity Optimization using the Spinu (2013) log-barrier formulation.

    Theoretical Foundation:
    -----------------------
    The problem of equating risk contributions:
    .. math::
        RC_i = w_i \\frac{(\\Sigma w)_i}{\\sqrt{w^T \\Sigma w}} = b_i

    Is isomorphic to the strictly convex minimization problem in variable $y$ (where $w = y / \\sum y_i$):
    .. math::
        \\text{minimize} \\quad \\frac{1}{2} y^T \\Sigma y - \\sum_{i=1}^N b_i \\ln(y_i)
        \\text{subject to} \\quad y > 0

    This formulation guarantees a unique, strictly positive solution vector $y^*$.
    """

    def __init__(self, target_risk: Optional[Dict[str, float]] = None):
        """
        Initializes the optimizer with specific risk budgets.

        Args:
            target_risk (Optional[Dict]): Target risk contribution per ticker (unnormalized).
                If None, defaults to Equal Risk Contribution (ERC), i.e., $b_i = 1/N$.
        """
        self.target_risk = target_risk

    # ==================== PUBLIC API ====================

    def optimize(
        self,
        covariance_matrix: pd.DataFrame,
        tickers: List[str],
    ) -> Dict[str, float]:
        """
        Executes the optimization routine to derive risk-balanced weights.

        Args:
            covariance_matrix (pd.DataFrame): Asset covariance matrix ($\Sigma$).
            tickers (List[str]): Universe of assets to optimize.

        Returns:
            Dict[str, float]: Normalized weight vector summing to 1.0.
        """
        # 1. Data Alignment: Intersect requested tickers with covariance data
        available_tickers = [t for t in tickers if t in covariance_matrix.index]
        if len(available_tickers) != len(tickers):
            missing = set(tickers) - set(available_tickers)
            logger.warning(
                f"⚠️ Missing covariance data for assets: {missing}. "
                "Optimizing available assets only."
            )

        if not available_tickers:
            return {}

        n = len(available_tickers)
        Sigma = covariance_matrix.loc[available_tickers, available_tickers].values

        # 2. Risk Budget Construction
        if self.target_risk is None:
            b = np.ones(n) / n  # Default: Equal Risk Contribution (ERC)
        else:
            b = np.array([self.target_risk.get(t, 1.0 / n) for t in available_tickers])
            b = b / b.sum()  # Normalize budgets to sum to 1.0

        # 3. Spinu (2013) Optimization
        # Solver: L-BFGS-B (Bound-constrained Quasi-Newton)
        # The log-barrier term naturally enforces y > 0, but explicit bounds aid stability.

        def _objective(y: np.ndarray) -> float:
            """
            Convex objective function.
            .. math:: J(y) = \\frac{1}{2} y^T \\Sigma y - b^T \\ln(y)
            """
            return -float(b @ np.log(y)) + 0.5 * float(y @ Sigma @ y)

        def _gradient(y: np.ndarray) -> np.ndarray:
            """
            Analytic gradient for L-BFGS-B.
            .. math:: \\nabla J(y) = \\Sigma y - b \\oslash y
            """
            return -b / y + Sigma @ y

        y0 = np.ones(n) / n          # Initial guess (Centroid)
        bounds = [(1e-8, None)] * n  # Bounds: y_i >= epsilon

        result = minimize(
            _objective,
            y0,
            jac=_gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10},
        )

        # 4. Weight Recovery: w = y / sum(y)
        if result.success and result.x is not None and result.x.sum() > 0:
            return self._pack_weights(result.x / result.x.sum(), available_tickers)

        # 5. Defensive Fallback: Inverse-Volatility
        # Triggered only on numerical convergence failure.
        logger.warning(
            f"⚠️ Spinu optimisation did not converge ({result.message}). "
            "Using Inverse Volatility fallback."
        )
        return self._inverse_vol_fallback(Sigma, available_tickers)

    # ==================== PRIVATE HELPERS ====================

    def _pack_weights(
        self, normalised_w: np.ndarray, tickers: List[str]
    ) -> Dict[str, float]:
        """
        Formatting utility: filters numerical noise and maps array to ticker dict.
        """
        weight_dict = {
            tickers[i]: float(w)
            for i, w in enumerate(normalised_w)
            if w > _WEIGHT_FLOOR
        }
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        
        logger.info(f"Risk parity optimised: {len(weight_dict)} positions")
        return weight_dict

    def _inverse_vol_fallback(
        self, Sigma: np.ndarray, tickers: List[str]
    ) -> Dict[str, float]:
        """
        Heuristic Fallback: Weights proportional to inverse standard deviation.
        .. math:: w_i \\propto \\frac{1}{\\sigma_i}
        Used when the primary solver fails to converge.
        """
        variances = np.diag(Sigma)
        volatilities = np.sqrt(np.maximum(variances, 1e-8))
        inv_vol = 1.0 / volatilities
        fallback_w = inv_vol / inv_vol.sum()

        weight_dict = {
            tickers[i]: float(w)
            for i, w in enumerate(fallback_w)
            if w > _WEIGHT_FLOOR
        }
        total = sum(weight_dict.values())
        if total > 0:
            weight_dict = {k: v / total for k, v in weight_dict.items()}
        return weight_dict