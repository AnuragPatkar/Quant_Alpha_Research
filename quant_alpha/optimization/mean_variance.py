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
    """
    Evaluates mathematical boundaries dynamically mapping internal CVXPY compiler dependencies.
    
    Returns:
        List[str]: Sequence of directly accessible convex optimization backends.
    """
    return cp.installed_solvers()


class MeanVarianceOptimizer:
    """Implementation of Markowitz portfolio optimization via convex programming."""
    def __init__(self, risk_aversion: float = 1.0):
        """
        Initializes the explicit Mean-Variance limits mapped natively to CVXPY optimization logic.
        
        Args:
            risk_aversion (float): Structural parameter ($\lambda$) weighting portfolio variance against gross returns. 
                Defaults to 1.0.
        """
        self.risk_aversion = risk_aversion

    def _prepare_data(
        self,
        expected_returns: Dict[str, float],
        covariance_matrix: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        """
        Structurally aligns and standardizes internal tensors bridging covariance limits to returns.
        
        Constraint bound: Output covariance evaluation matrix must theoretically be positive semi-definite (PSD).
        
        Args:
            expected_returns (Dict[str, float]): Forecast return map matching independent asset configurations.
            covariance_matrix (pd.DataFrame): Primary dimensional constraint space matrix.
            
        Returns:
            Tuple[List[str], np.ndarray, np.ndarray, int]: Sequenced tickers, extracted array vectors ($\mu, \Sigma$), and count bounds.
            
        Raises:
            ValueError: If intersecting parameters result natively in empty dimension limits.
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
        Filters continuous mathematical noise resolving rigid standard simplex limits ($\sum w_i = 1$).
        
        Systematically prunes allocation coordinates bound below standard epsilon numerical floors ($\epsilon < 1e-6$).
        
        Args:
            optimized_w (np.ndarray): Direct structural optimization continuous mapping result.
            tickers (List[str]): Sequence of dimensional target identities.
            
        Returns:
            Dict[str, float]: Strictly validated positive limits correctly representing execution limits.
            
        Raises:
            ValueError: When non-convergent structural gaps trigger mass scale infeasibility failures.
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
        Executes dynamic mathematical constraint scaling guaranteeing structural continuous states $\mathcal{F}$ remain active.
        
        If $N \cdot w_{max} < 1$, the continuous matrix budget strictly prohibits optimal solutions.
        This function mathematically extracts bounds $w_i \le \max(w_{req}, 1/N)$ enforcing optimization bounds.
        
        Args:
            n (int): Cardinality vector defining absolute allocation states.
            constraints (Optional[Dict]): Structural limit override mapping target constraints.
            
        Returns:
            float: Valid upper limit boundary preserving simplex integrity calculations.
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
        Solves explicitly bounded Standard Mean-Variance configurations structurally evaluating a continuous Quadratic Program.

        .. math::
            \text{maximize} \quad \mu^T w - \lambda w^T \Sigma w \\
            \text{subject to} \quad \mathbf{1}^T w = 1, \quad w \ge 0, \quad w \le w_{max}
            
        Args:
            expected_returns (Dict[str, float]): Targeted alpha boundary vectors ($\mu$).
            covariance_matrix (pd.DataFrame): Systemic boundary limits mapping variance ($\Sigma$).
            constraints (Optional[Dict]): Hard structural restrictions limiting algorithmic cardinality.
            
        Returns:
            Dict[str, float]: Mapped portfolio definitions normalized across strict matrix constraints.
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

        Utilizes the structural Charnes-Cooper transformation to linearize fractional objective domains:
        Let $y = \kappa w$ where $\kappa > 0$.

        .. math::
            \text{maximize} \quad (\mu - r_f)^T y \\
            \text{subject to} \quad y^T \Sigma y \le 1 \\
            \quad \mathbf{1}^T y = \kappa, \quad y \ge 0, \quad \kappa \ge 0
            
        Final Result: $w^* = y^* / \kappa^*$.
        
        Args:
            expected_returns (Dict[str, float]): Targeted execution vectors ($\mu$).
            covariance_matrix (pd.DataFrame): Evaluated structural covariance definitions ($\Sigma$).
            risk_free_rate (float): Absolute mathematical barrier for ratio limits ($r_f$).
            constraints (Optional[Dict]): Internal overrides enforcing weight cardinality.
            
        Returns:
            Dict[str, float]: The maximal distribution structure fully invested against strict probability limits.
        """
        try:
            tickers, mu, Sigma, n = self._prepare_data(expected_returns, covariance_matrix)
        except ValueError as e:
            logger.error(str(e))
            return {}

        y = cp.Variable(n)
        kappa = cp.Variable()

        objective = cp.Maximize((mu - risk_free_rate) @ y)

        effective_max_w = self._resolve_max_weight(n, constraints)

        constraints_list = [
            cp.quad_form(y, Sigma) <= 1,   # Risk constraint (Unit Variance in transformed space)
            cp.sum(y) == kappa,
            y >= 0,
            y <= effective_max_w * kappa,
            kappa >= 0,
        ]

        prob = cp.Problem(objective, constraints_list)

        # Initiates hierarchical Solver Cascade Strategy prioritizing precise QCQP/SOCP solutions.
        # Attempts ordered structural dispatch (ECOS -> SCS -> CLARABEL) gracefully falling back 
        # upon localized interior-point or cone numerical convergence failures.
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