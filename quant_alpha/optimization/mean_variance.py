"""
Mean-Variance Portfolio Optimization Engine
===========================================
Convex optimization solver for modern portfolio theory (MPT) construction.

Purpose
-------
The `MeanVarianceOptimizer` translates financial objectives (maximize return, minimize risk)
into convex optimization problems. It supports two primary formulations:
1.  **Quadratic Programming (QP)**: Minimizes portfolio variance for a given risk aversion ($\lambda$).
2.  **Second-Order Cone Programming (SOCP)**: Maximizes the Sharpe Ratio directly via the
    Charnes-Cooper transformation, avoiding heuristic grid searches.

Usage
-----
Intended for use via the `PortfolioAllocator` facade.

.. code-block:: python

    optimizer = MeanVarianceOptimizer(risk_aversion=1.0)
    weights = optimizer.optimize(
        expected_returns={'AAPL': 0.12, ...},
        covariance_matrix=cov_df,
        constraints={'max_weight': 0.10}
    )

Importance
----------
-   **Convexity Guarantees**: Utilizes `cvxpy` to ensure that if a solution exists, it is the global optimum.
-   **Solver Robustness**: Implements a cascade of solvers (OSQP, ECOS, SCS, CLARABEL) to handle
    numerical instability in ill-conditioned covariance matrices.
-   **Defensive Feasibility**: Automatically relaxes weight caps in concentrated universes ($N < 1/w_{max}$)
    to prevent infeasibility errors.

Tools & Frameworks
------------------
-   **CVXPY**: Domain-specific language for convex optimization.
-   **Solvers**:
    -   **OSQP**: Operator Splitting Quadratic Program solver (Primary for QP).
    -   **ECOS/CLARABEL**: Embedded Conic Solver (Primary for SOCP).
    -   **SCS**: Splitting Conic Solver (Fallback).
"""

import pandas as pd
import numpy as np
import cvxpy as cp
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


def _get_available_solvers() -> List[str]:
    """Return CVXPY solvers available in the current environment."""
    return cp.installed_solvers()


class MeanVarianceOptimizer:
    """Implementation of Markowitz portfolio optimization via convex programming."""
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
        Aligns and vectorizes input maps for linear algebra operations.
        Pre-condition: Covariance matrix must be positive semi-definite (PSD).
        """
        available_tickers = set(covariance_matrix.index) & set(covariance_matrix.columns)
        tickers = [t for t in expected_returns.keys() if t in available_tickers]
        n = len(tickers)

        if n == 0:
            raise ValueError(
                "No valid tickers found in intersection of expected_returns and covariance_matrix."
            )

        mu = np.array([expected_returns[t] for t in tickers])
        Sigma = covariance_matrix.loc[tickers, tickers].values

        return tickers, mu, Sigma, n

    def _safe_normalize(self, optimized_w: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """
        Extracts weights and enforces the simplex constraint $\sum w_i = 1$.
        Filters numerical noise ($\epsilon < 1e-6$) and validates solver feasibility.
        """
        n = len(tickers)
        weight_dict = {
            tickers[i]: round(float(optimized_w[i]), 8)
            for i in range(n) if optimized_w[i] > 1e-6
        }

        total_sum = sum(weight_dict.values())

        if abs(total_sum - 1.0) < 1e-4:
            if total_sum > 0:
                return {k: v / total_sum for k, v in weight_dict.items()}
            return weight_dict
        else:
            raise ValueError(f"Solver returned infeasible weights summing to {total_sum:.4f}")

    def _resolve_max_weight(self, n: int, constraints: Optional[Dict]) -> float:
        """
        Dynamic Constraint Relaxation: Ensures the feasible set $\mathcal{F}$ is non-empty.
        
        Logic:
        If $N \cdot w_{max} < 1$, the budget constraint $\sum w_i = 1$ strictly cannot be satisfied.
        We enforce $w_i \le \max(w_{req}, 1/N)$ to guarantee solvability in small/concentrated
        universes while respecting user intent where possible.
        """
        if constraints is None:
            requested = 1.0  # unconstrained: let optimizer decide freely
        else:
            requested = constraints.get('max_weight', 1.0)

        infeasibility_floor = 1.0 / n
        return max(requested, infeasibility_floor)

    def optimize(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Solves the Standard Mean-Variance Quadratic Program (QP).

        .. math::
            \text{maximize} \quad \mu^T w - \lambda w^T \Sigma w \\
            \text{subject to} \quad \mathbf{1}^T w = 1, \quad w \ge 0, \quad w \le w_{max}
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

        # Feasibility Guard: Ensure max_weight doesn't break sum(w)=1
        effective_max_w = self._resolve_max_weight(n, constraints)
        constraints_list.append(w <= effective_max_w)

        prob = cp.Problem(objective, constraints_list)

        try:
            prob.solve(solver=cp.OSQP)

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                raise cp.SolverError(f"Solver status: {prob.status}")

            weight_dict = self._safe_normalize(w.value, tickers)
            logger.info(f"Optimization successful for {len(weight_dict)} assets.")
            return weight_dict

        except (cp.SolverError, ValueError) as e:
            logger.error(f"Optimization failed: {e}. Falling back to Equal Weight.")
            return {t: 1.0 / n for t in tickers}

    def solve_max_sharpe(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Direct Sharpe Ratio Maximization via Second-Order Cone Programming (SOCP).

        Utilizes the **Charnes-Cooper transformation** to linearize the fractional objective:
        Let $y = \kappa w$ where $\kappa > 0$.

        .. math::
            \text{maximize} \quad (\mu - r_f)^T y \\
            \text{subject to} \quad y^T \Sigma y \le 1 \\
            \quad \mathbf{1}^T y = \kappa, \quad y \ge 0, \quad \kappa \ge 0
            
        Recovered Weights: $w^* = y^* / \kappa^*$.
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        y = cp.Variable(n)
        kappa = cp.Variable()

        objective = cp.Maximize((mu - risk_free_rate) @ y)

        # Feasibility Guard: Scale constraints by variable kappa
        effective_max_w = self._resolve_max_weight(n, constraints)

        constraints_list = [
            cp.quad_form(y, Sigma) <= 1,   # Risk constraint (Unit Variance in transformed space)
            cp.sum(y) == kappa,
            y >= 0,
            y <= effective_max_w * kappa,
            kappa >= 0,
        ]

        prob = cp.Problem(objective, constraints_list)

        # Solver Cascade Strategy:
        # QCQP/SOCP problems require conic solvers. We attempt dispatch in order of reliability:
        # 1. ECOS: Best for SOCP.
        # 2. SCS: Robust first-order solver.
        # 3. CLARABEL: Modern interior-point solver (CVXPY default).
        qcqp_solvers = ["ECOS", "SCS", "CLARABEL"]
        installed = _get_available_solvers()
        solver_chain = [s for s in qcqp_solvers if s in installed]

        if not solver_chain:
            logger.error(
                "No QCQP-capable solver available (tried ECOS, SCS, CLARABEL). "
                "Install one via: pip install ecos  OR  pip install clarabel"
            )
            return {t: 1.0 / n for t in tickers}

        last_error = None
        for solver_name in solver_chain:
            try:
                solver_const = getattr(cp, solver_name)
                prob.solve(solver=solver_const)

                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    raise cp.SolverError(f"Solver {solver_name} status: {prob.status}")

                if y.value is None or kappa.value is None or float(kappa.value) <= 1e-6:
                    raise cp.SolverError(
                        f"Solver {solver_name} returned None or near-zero kappa "
                        f"(kappa={kappa.value})"
                    )

                weights = y.value / float(kappa.value)
                weight_dict = self._safe_normalize(weights, tickers)
                logger.info(
                    f"Sharpe optimization successful via {solver_name} "
                    f"for {len(weight_dict)} assets."
                )
                return weight_dict

            except (cp.SolverError, ValueError, ZeroDivisionError) as e:
                logger.warning(f"Solver {solver_name} failed: {e}. Trying next solver.")
                last_error = e
                continue

        logger.error(
            f"All QCQP solvers failed. Last error: {last_error}. "
            "Falling back to Equal Weight."
        )
        return {t: 1.0 / n for t in tickers}